"""Lightning training core shared by the unified application entrypoint."""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Mapping, Sequence

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from .model import PeSToGaussianModel
from .objectives import semantic_patch_losses
from .config import ExperimentConfig, PESTO_MODEL_CONFIG
from .data import (
    CompactGaussianDirectDataset,
    collate_single,
    model_arguments_from_batch,
)
from .utils import (
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    read_csv_rows,
)


SEMANTIC_OBJECTIVE_ARGS = argparse.Namespace(
    semantic_patch_region_existence_threshold=0.05,
    semantic_patch_component_k=16,
)


class GlobalBatchSampler:
    """Split deterministic global batches into rank-balanced local batches."""

    def __init__(self, batches: Sequence[Sequence[int]], rank: int, world_size: int) -> None:
        self.batches = batches
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for global_batch in self.batches:
            local_size = math.ceil(len(global_batch) / self.world_size)
            start = self.rank * local_size
            yield list(global_batch[start : start + local_size])

    def __len__(self) -> int:
        return len(self.batches)


class GaussianDirectDataModule(LightningDataModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.dataset: CompactGaussianDirectDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage is not None and stage != "fit":
            return
        data = self.config.training.data
        rows = read_csv_rows(data.train_manifest)
        if len(rows) != data.expected_train_rows:
            raise ValueError(
                f"expected {data.expected_train_rows} train rows, got {len(rows)}"
            )
        self.dataset = CompactGaussianDirectDataset(
            data.train_manifest, data.train_compact_dir
        )

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("training dataset was not initialized")
        runtime = self.config.runtime
        optimizer = self.config.training.optimizer
        world_size = self._world_size()
        rank = int(os.environ.get("RANK", "0"))
        global_batch_size = optimizer.local_batch_size * world_size
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            runtime.seed * 1_000_003 + self.trainer.current_epoch
        )
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        global_batches = [
            order[offset : offset + global_batch_size]
            for offset in range(0, len(order), global_batch_size)
        ]
        if len(global_batches) > 1 and len(global_batches[-1]) < world_size:
            remainder = global_batches.pop()
            previous = global_batches.pop()
            tail = previous + remainder
            split = math.ceil(len(tail) / 2)
            global_batches.extend([tail[:split], tail[split:]])
        sampler = GlobalBatchSampler(global_batches, rank, world_size)
        return DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=runtime.num_workers,
            collate_fn=collate_single,
            pin_memory=True,
            persistent_workers=runtime.num_workers > 0,
            prefetch_factor=2 if runtime.num_workers > 0 else None,
        )

    def _world_size(self) -> int:
        if self.trainer is not None:
            return max(int(self.trainer.world_size), 1)
        return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


class GaussianDirectLightningModule(LightningModule):
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        steps_per_epoch: int,
        pos_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.steps_per_epoch = int(steps_per_epoch)
        architecture = config.architecture
        self.save_hyperparameters(config.to_dict())
        self.model = PeSToGaussianModel(
            pesto_config=PESTO_MODEL_CONFIG,
            gaussian_dim=architecture.gaussian_dim,
            co_update_rounds=architecture.co_update_rounds,
            dropout=architecture.dropout,
            zero_init_output=True,
            architecture=architecture.gaussian_architecture,
            direct_protein_residual=architecture.direct_protein_residual,
            direct_protein_head=architecture.direct_protein_head,
            bypass_gaussian=architecture.bypass_gaussian,
            parameter_matched_mlp=architecture.parameter_matched_mlp,
            ablation_no_surface_vector=architecture.ablation_no_surface_vector,
            ablation_no_residue_feedback=architecture.ablation_no_residue_feedback,
        )
        self.register_buffer("pos_weight", pos_weight.detach().clone(), persistent=False)
        self._epoch_loss_sum = 0.0
        self._epoch_batches = 0
        self._metrics: list[dict[str, object]] = []

    def configure_optimizers(self) -> dict[str, Any]:
        architecture = self.config.architecture
        optimizer_config = self.config.training.optimizer
        gaussian_parameters = (
            list(self.model.decoder_projection.parameters())
            if architecture.bypass_gaussian
            else list(self.model.gaussian.parameters())
        )
        if architecture.direct_protein_residual:
            gaussian_parameters.extend(self.model.protein_residual_head.parameters())
            gaussian_parameters.append(self.model.protein_residual_scale)
        if architecture.direct_protein_head:
            gaussian_parameters.extend(self.model.protein_output_head.parameters())
        groups: list[dict[str, Any]]
        if architecture.experiment == "frozen":
            for parameter in self.model.pesto.parameters():
                parameter.requires_grad_(False)
            groups = [
                {"params": gaussian_parameters, "lr": optimizer_config.gaussian_lr}
            ]
        else:
            groups = [
                {"params": list(self.model.pesto.parameters()), "lr": optimizer_config.pesto_lr},
                {"params": gaussian_parameters, "lr": optimizer_config.gaussian_lr},
            ]
        optimizer = torch.optim.AdamW(
            groups, weight_decay=optimizer_config.weight_decay
        )
        schedule_epochs = optimizer_config.schedule_epochs or optimizer_config.epochs
        total_steps = schedule_epochs * self.steps_per_epoch
        warmup_steps = int(optimizer_config.warmup_ratio * total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _scheduler_lambda(
                step,
                total_steps,
                warmup_steps,
                optimizer_config.min_lr_ratio,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_epoch_start(self) -> None:
        self._epoch_loss_sum = 0.0
        self._epoch_batches = 0
        if self.model.gaussian is not None:
            self.model.gaussian.set_semantic_epoch(self.current_epoch)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        logits = self.model(**model_arguments_from_batch(batch, self.device))
        labels = batch["y"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        world_size = max(int(self.trainer.world_size), 1)
        loss = _global_mean_loss(
            logits, labels, mask, self.pos_weight, world_size
        )
        semantic_loss = self._semantic_auxiliary_loss(logits, batch)
        if semantic_loss is not None:
            loss = loss + semantic_loss
        displayed_loss = loss.detach() / world_size
        self._epoch_loss_sum += float(displayed_loss)
        self._epoch_batches += 1
        self.log(
            "train_loss",
            displayed_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        return loss

    def _semantic_auxiliary_loss(
        self, logits: torch.Tensor, batch: Mapping[str, Any]
    ) -> torch.Tensor | None:
        semantic_outputs = getattr(self.model.gaussian, "last_semantic_outputs", [])
        if not semantic_outputs:
            return None
        objective_batch = {
            "y": (
                batch["y"].to(self.device)
                if batch["y"].ndim == 1
                else batch["y"].to(self.device)[:, 0]
            ),
            "mask": batch["mask"].to(self.device),
            "residue_counts": batch["residue_counts"],
        }
        losses = semantic_patch_losses(
            {"final_logits": logits, "semantic_patch_outputs": semantic_outputs},
            objective_batch,
            SEMANTIC_OBJECTIVE_ARGS,
        )
        return (
            0.03 * losses["amplitude"]
            + 0.03 * losses["set"]
            + 0.01 * losses["existence"]
            + 0.003 * losses["connectivity"]
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        del gradient_clip_val, gradient_clip_algorithm
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def on_train_epoch_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        optimizer = self.trainer.optimizers[0]
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        scaler = getattr(
            self.trainer.precision_plugin,
            "scaler",
            torch.cuda.amp.GradScaler(enabled=False),
        )
        result = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "learning_rate": float(optimizer.param_groups[-1]["lr"]),
            "train_loss": self._epoch_loss_sum / max(self._epoch_batches, 1),
        }
        self._metrics.append(result)
        checkpoint = {
            "schema_version": "pesto_gaussian_plugin_checkpoint_v1",
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": self._metrics,
            "running_loss": self._epoch_loss_sum,
            "epoch_batch_count": self._epoch_batches,
            "checkpoint_phase": "epoch_end",
            "args": self._checkpoint_args(),
            "app_config": self.config.to_dict(),
        }
        output_dir = self.config.training.output_dir
        atomic_torch_save(output_dir / f"model_epoch_{self.current_epoch:02d}.pt", checkpoint)
        atomic_torch_save(output_dir / "last.pt", checkpoint)
        atomic_write_json(output_dir / f"metrics_epoch_{self.current_epoch:02d}.json", result)
        atomic_write_csv(output_dir / "epoch_metrics.csv", self._metrics)

    def _checkpoint_args(self) -> dict[str, Any]:
        architecture = self.config.architecture
        optimizer = self.config.training.optimizer
        return {
            "experiment": architecture.experiment,
            "gaussian_dim": architecture.gaussian_dim,
            "gaussian_architecture": architecture.gaussian_architecture,
            "direct_protein_residual": architecture.direct_protein_residual,
            "direct_protein_head": architecture.direct_protein_head,
            "bypass_gaussian": architecture.bypass_gaussian,
            "parameter_matched_mlp": architecture.parameter_matched_mlp,
            "ablation_no_surface_vector": architecture.ablation_no_surface_vector,
            "ablation_no_residue_feedback": architecture.ablation_no_residue_feedback,
            "co_update_rounds": architecture.co_update_rounds,
            "dropout": architecture.dropout,
            "epochs": optimizer.epochs,
            "schedule_epochs": optimizer.schedule_epochs,
            "batch_size": optimizer.local_batch_size,
            "pesto_lr": optimizer.pesto_lr,
            "gaussian_lr": optimizer.gaussian_lr,
            "min_lr_ratio": optimizer.min_lr_ratio,
            "warmup_ratio": optimizer.warmup_ratio,
            "weight_decay": optimizer.weight_decay,
            "pos_weight_factor": optimizer.pos_weight_factor,
            "seed": self.config.runtime.seed,
            "output_dir": str(self.config.training.output_dir),
        }


def _scheduler_lambda(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    if warmup_steps and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    progress = min(
        max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _global_mean_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    current = mask > 0.5
    logit_targets = logits[:, 0] if logits.ndim == 2 else logits
    label_targets = labels[:, 0] if labels.ndim == 2 else labels
    loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
        logit_targets[current],
        label_targets[current],
        pos_weight=pos_weight,
        reduction="sum",
    )
    global_valid = current.sum()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_valid, op=torch.distributed.ReduceOp.SUM)
    return loss_sum * world_size / global_valid.clamp_min(1)


def lightning_ddp_strategy() -> DDPStrategy:
    return DDPStrategy(static_graph=True)
