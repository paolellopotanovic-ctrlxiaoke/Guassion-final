"""Unified Gaussian-direct128 application service."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data import (
    CompactGaussianDirectDataset,
    collate_single,
    load_checkpoint_model,
    load_compact_npz,
    model_arguments_from_batch,
    split_field,
)
from .lightning import (
    GaussianDirectDataModule,
    GaussianDirectLightningModule,
    lightning_ddp_strategy,
)
from .utils import (
    atomic_write_json,
    read_csv_rows,
    resolve_device,
    sha256_file,
)


CHANNEL_NAMES = ("protein", "nucleic", "ion", "ligand", "lipid")
ROOT = Path(__file__).resolve().parents[2]


class DirectNPZDataset(Dataset):
    def __init__(
        self, paths: Sequence[Path], residue_ids: Sequence[Sequence[str]] = ()
    ) -> None:
        self.paths = list(paths)
        self.residue_ids = list(residue_ids)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = load_compact_npz(self.paths[index])
        ids = self.residue_ids[index] if index < len(self.residue_ids) else (
            str(value) for value in range(item["y"].numel())
        )
        item["output_residue_ids"] = list(ids)
        return item


def collate_direct(batch: list[dict[str, Any]]) -> dict[str, Any]:
    base = collate_single(batch)
    base["output_residue_ids"] = [item["output_residue_ids"] for item in batch]
    base["direct_case_ids"] = [item["case_id"] for item in batch]
    return base


class GaussianDirectApp:
    """Single application class driven by one typed experiment configuration."""

    def __init__(self, config: ExperimentConfig | Path | str) -> None:
        self.config = (
            config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_file(config)
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "GaussianDirectApp":
        return cls(ExperimentConfig.from_file(path))

    def train(self) -> None:
        training = self.config.training
        data = training.data
        optimizer = training.optimizer
        runtime = self.config.runtime
        self._validate_path(data.train_manifest, "training manifest")
        self._validate_path(data.train_compact_dir, "training compact directory")
        rows = read_csv_rows(data.train_manifest)
        valid_residues, positive_residues = _protein_class_counts(rows)
        if not valid_residues:
            raise ValueError("training manifest has no valid labeled residues")
        positive_rate = positive_residues / valid_residues
        pos_weight = optimizer.pos_weight_factor * (1.0 - positive_rate) / max(
            positive_rate, 1e-6
        )
        global_batch_size = optimizer.local_batch_size * runtime.devices
        steps_per_epoch = math.ceil(len(rows) / global_batch_size)
        seed_everything(runtime.seed, workers=True)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = False
        datamodule = GaussianDirectDataModule(self.config)
        module = GaussianDirectLightningModule(
            self.config,
            steps_per_epoch=steps_per_epoch,
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32),
        )
        trainer = Trainer(
            accelerator=runtime.accelerator,
            devices=runtime.devices,
            strategy=lightning_ddp_strategy() if runtime.devices > 1 else "auto",
            precision=runtime.precision if runtime.accelerator != "cpu" else "32",
            max_epochs=optimizer.epochs,
            log_every_n_steps=runtime.log_every,
            default_root_dir=training.output_dir,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            benchmark=False,
        )
        contract = self.config.to_dict()
        contract["training_statistics"] = {
            "rows": len(rows),
            "valid_residues": valid_residues,
            "positive_residues": positive_residues,
            "positive_rate": positive_rate,
            "pos_weight": pos_weight,
            "global_batch_size": global_batch_size,
            "steps_per_epoch": steps_per_epoch,
        }
        training.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(training.output_dir / "run_contract.json", contract)
        trainer.fit(module, datamodule=datamodule)
        if trainer.is_global_zero:
            (training.output_dir / "TRAINING_COMPLETE").touch()

    def infer(self) -> Path:
        settings = self.config.inference
        device = resolve_device(settings.device)
        if settings.npz:
            dataset: Dataset = DirectNPZDataset(
                settings.npz,
                [str(settings.residue_ids or "").split(";")]
                if len(settings.npz) == 1 and settings.residue_ids
                else (),
            )
            collate = collate_direct
        else:
            self._validate_path(settings.manifest, "inference manifest")
            self._validate_path(settings.compact_dir, "inference compact directory")
            dataset = CompactGaussianDirectDataset(settings.manifest, settings.compact_dir)
            collate = collate_single
        model, payload = load_checkpoint_model(settings.checkpoint, device)
        loader = DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        rows: list[dict[str, Any]] = []
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(**model_arguments_from_batch(batch, device)).float().cpu()
                probability = torch.sigmoid(logits)
                probabilities.append(probability[:, 0].numpy())
                if "output_residue_ids" in batch:
                    residue_id_groups = batch["output_residue_ids"]
                else:
                    residue_id_groups = [
                        [str(value) for value in range(count)]
                        for count in batch["residue_counts"]
                    ]
                case_id_groups = batch.get("direct_case_ids") or [
                    row.get("apex_case_id", "") for row in batch["row"]
                ]
                offset = 0
                for case_id, sample_ids, count in zip(
                    case_id_groups, residue_id_groups, batch["residue_counts"]
                ):
                    for local_index, residue_id in enumerate(sample_ids):
                        index = offset + local_index
                        rows.append(
                            {
                                "case_id": case_id,
                                "residue_index": local_index,
                                "residue_id": residue_id,
                                "protein_probability": float(probability[index, 0]),
                                "predicted_interface": int(
                                    probability[index, 0] >= settings.threshold
                                ),
                                **{
                                    f"channel_{name}_probability": float(
                                        probability[index, channel]
                                    )
                                    for channel, name in enumerate(CHANNEL_NAMES)
                                },
                            }
                        )
                    offset += int(count)
        settings.output.parent.mkdir(parents=True, exist_ok=True)
        with settings.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        values = np.concatenate(probabilities) if probabilities else np.array([])
        atomic_write_json(
            settings.output.with_suffix(".summary.json"),
            {
                "checkpoint": str(settings.checkpoint),
                "checkpoint_sha256": sha256_file(settings.checkpoint),
                "checkpoint_epoch": payload.get("epoch"),
                "cases": len(dataset),
                "residues": int(values.size),
                "mean_protein_probability": float(values.mean()) if values.size else None,
                "output": str(settings.output),
            },
        )
        return settings.output

    def evaluate(self) -> Path:
        settings = self.config.evaluation
        device = resolve_device(settings.device)
        self._validate_path(settings.manifest, "evaluation manifest")
        self._validate_path(settings.compact_dir, "evaluation compact directory")
        dataset = CompactGaussianDirectDataset(settings.manifest, settings.compact_dir)
        loader = DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            collate_fn=collate_single,
            pin_memory=device.type == "cuda",
        )
        model, payload = load_checkpoint_model(settings.checkpoint, device)
        labels_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []
        case_metrics: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(**model_arguments_from_batch(batch, device)).float().cpu()
                scores = torch.sigmoid(logits[:, 0]).numpy()
                labels = batch["y"].numpy()
                mask = batch["mask"].numpy() > 0.5
                labels_all.append(labels[mask])
                scores_all.append(scores[mask])
                offset = 0
                for row, count in zip(batch["row"], batch["residue_counts"]):
                    local = slice(offset, offset + int(count))
                    local_labels = labels[local][mask[local]]
                    local_scores = scores[local][mask[local]]
                    case_metrics.append(
                        {
                            "case_id": row.get("apex_case_id", ""),
                            "input_state": row.get("input_state", ""),
                            "aupr": _safe_aupr(local_labels, local_scores),
                            "auroc": _safe_auroc(local_labels, local_scores),
                            "p10": _precision_at_k(local_labels, local_scores, 10),
                            "p20": _precision_at_k(local_labels, local_scores, 20),
                        }
                    )
                    offset += int(count)
        labels = np.concatenate(labels_all)
        scores = np.concatenate(scores_all)
        metrics = {
            "checkpoint": str(settings.checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "num_cases": len(dataset),
            "pooled_aupr": _safe_aupr(labels, scores),
            "pooled_auroc": _safe_auroc(labels, scores),
            "case_mean_aupr": _finite_mean(case_metrics, "aupr"),
            "case_mean_auroc": _finite_mean(case_metrics, "auroc"),
            "case_mean_p_at_10": _finite_mean(case_metrics, "p10"),
            "case_mean_p_at_20": _finite_mean(case_metrics, "p20"),
        }
        atomic_write_json(settings.output, metrics)
        if case_metrics:
            case_path = settings.output.with_name(
                settings.output.stem + ".case_metrics.csv"
            )
            with case_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(case_metrics[0]))
                writer.writeheader()
                writer.writerows(case_metrics)
        return settings.output

    def verify(self) -> dict[str, Any]:
        contract = json.loads(
            (ROOT / "data/data_contracts.json").read_text(encoding="utf-8")
        )["checkpoint"]
        checkpoint = ROOT / contract["path"]
        actual_hash = sha256_file(checkpoint)
        if actual_hash != contract["sha256"]:
            expected_hash = contract["sha256"]
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )

        manifest_path = ROOT / "RELEASE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise RuntimeError("release manifest must contain a non-empty files array")
        total_bytes = 0
        for record in records:
            relative = Path(str(record["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe release-manifest path: {relative}")
            path = ROOT / relative
            actual_size = path.stat().st_size
            actual_record_hash = sha256_file(path)
            if actual_size != int(record["bytes"]):
                raise RuntimeError(f"release size mismatch: {relative}")
            if actual_record_hash != str(record["sha256"]):
                raise RuntimeError(f"release SHA-256 mismatch: {relative}")
            total_bytes += actual_size
        if len(records) != int(manifest["file_count"]):
            raise RuntimeError("release file_count mismatch")
        if total_bytes != int(manifest["total_bytes"]):
            raise RuntimeError("release total_bytes mismatch")

        result = {
            "release_files": len(records),
            "release_total_bytes": total_bytes,
            "checkpoint": str(checkpoint),
            "sha256": actual_hash,
            "checkpoint_epoch": int(contract["epoch"]),
            "example_manifest": str(self.config.inference.manifest),
        }
        rows = read_csv_rows(self.config.inference.manifest)
        result["examples"] = len(rows)
        return result

    def preprocess(self) -> None:
        settings = self.config.preprocessing
        command = [
            sys.executable,
            "-m",
            "gaussian_direct.preprocessing",
            "--manifest",
            str(settings.manifest),
            "--output-dir",
            str(settings.output_dir),
            "--workers",
            str(settings.workers),
            "--torch-threads",
            str(settings.torch_threads),
            "--max-num-nn",
            str(settings.max_num_nn),
            "--payload",
            settings.payload,
            "--limit",
            str(settings.limit),
            "--progress-every",
            str(settings.progress_every),
            "--chunksize",
            str(settings.chunksize),
        ]
        if settings.trust_final_filter:
            command.append("--trust-final-filter")
        if settings.overwrite:
            command.append("--overwrite")
        environment = os.environ.copy()
        source_root = str(ROOT / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root
            if not existing_pythonpath
            else source_root + os.pathsep + existing_pythonpath
        )
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"preprocessing failed with exit code {completed.returncode}")

    @staticmethod
    def _validate_path(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def _protein_class_counts(rows: Sequence[Mapping[str, str]]) -> tuple[int, int]:
    positive = 0
    valid = 0
    for row in rows:
        labels = np.asarray(split_field(row["labels"], int), dtype=np.int64)
        mask = np.asarray(split_field(row["label_mask"], int), dtype=np.int64) > 0
        if labels.shape != mask.shape:
            raise ValueError(f"invalid labels for {row.get('apex_case_id', '')}")
        positive += int(labels[mask].sum())
        valid += int(mask.sum())
    return valid, positive


def _safe_aupr(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(average_precision_score(labels, scores)) if labels.size and labels.sum() else None


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None


def _precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    if not labels.size:
        return None
    selected = np.argsort(-scores, kind="stable")[: min(k, labels.size)]
    return float(labels[selected].mean())


def _finite_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None
