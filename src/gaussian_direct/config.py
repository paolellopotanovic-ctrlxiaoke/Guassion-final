"""Typed experiment configuration for Gaussian-direct128."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

PESTO_MODEL_CONFIG = {
    "em": {"N0": 30, "N1": 32},
    "sum": [
        *(
            {"Ns": 32, "Nh": 2, "Nk": 3, "nn": 8}
            for _ in range(8)
        ),
        *(
            {"Ns": 32, "Nh": 2, "Nk": 3, "nn": 16}
            for _ in range(8)
        ),
        *(
            {"Ns": 32, "Nh": 2, "Nk": 3, "nn": 32}
            for _ in range(8)
        ),
        *(
            {"Ns": 32, "Nh": 2, "Nk": 3, "nn": 64}
            for _ in range(8)
        ),
    ],
    "spl": {"N0": 32, "N1": 32, "Nh": 4},
    "dm": {"N0": 32, "N1": 32, "N2": 5},
}


@dataclass(frozen=True)
class ArchitectureConfig:
    experiment: str = "scratch"
    gaussian_dim: int = 128
    gaussian_architecture: str = "vg"
    direct_protein_head: bool = True
    direct_protein_residual: bool = False
    bypass_gaussian: bool = False
    parameter_matched_mlp: bool = False
    ablation_no_surface_vector: bool = False
    ablation_no_residue_feedback: bool = False
    co_update_rounds: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.gaussian_architecture not in {"base", "vg", "sem"}:
            raise ValueError("gaussian_architecture must be base, vg, or sem")
        if self.direct_protein_head and self.direct_protein_residual:
            raise ValueError(
                "direct_protein_head and direct_protein_residual are mutually exclusive"
            )
        if self.bypass_gaussian and not self.direct_protein_head:
            raise ValueError("bypass_gaussian requires direct_protein_head")
        if self.parameter_matched_mlp and not (
            self.bypass_gaussian and self.direct_protein_head
        ):
            raise ValueError(
                "parameter_matched_mlp requires bypass_gaussian and direct_protein_head"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    accelerator: str = "auto"
    devices: int = 2
    precision: str = "16-mixed"
    seed: int = 42
    num_workers: int = 2
    log_every: int = 50


@dataclass(frozen=True)
class TrainingDataConfig:
    train_manifest: Path = Path("${TRAIN_MANIFEST}")
    train_compact_dir: Path = Path("${TRAIN_COMPACT_DIR}")
    expected_train_rows: int = 20801


@dataclass(frozen=True)
class OptimizerConfig:
    epochs: int = 20
    schedule_epochs: int = 20
    local_batch_size: int = 4
    pesto_lr: float = 1e-4
    gaussian_lr: float = 1e-4
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    weight_decay: float = 1e-4
    pos_weight_factor: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path = Path("runs/gaussian_direct128")
    data: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class PreprocessingConfig:
    manifest: Path = Path("${PREPROCESS_MANIFEST}")
    output_dir: Path = Path("${PREPROCESS_COMPACT_DIR}")
    workers: int = 16
    torch_threads: int = 1
    max_num_nn: int = 64
    payload: str = "training"
    limit: int = 0
    progress_every: int = 1000
    chunksize: int = 8
    trust_final_filter: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: Path = Path("checkpoints/direct128_e11.pt")
    manifest: Path = Path("examples/example_manifest.csv")
    compact_dir: Path = Path("examples/compact_npz")
    npz: tuple[Path, ...] = ()
    residue_ids: str | None = None
    output: Path = Path("predictions/predictions.csv")
    device: str = "auto"
    batch_size: int = 1
    threshold: float = 0.5


@dataclass(frozen=True)
class EvaluationConfig(InferenceConfig):
    output: Path = Path("results/evaluation_metrics.json")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str = "gaussian_direct128_app_config_v1"
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, value: MappingLike, *, base_dir: Path | None = None) -> "ExperimentConfig":
        base = cls()
        architecture = _merge_dataclass(ArchitectureConfig, base.architecture, value.get("architecture"))
        runtime = _merge_dataclass(RuntimeConfig, base.runtime, value.get("runtime"))
        training_value = dict(value.get("training") or {})
        training_data = _merge_dataclass(
            TrainingDataConfig,
            base.training.data,
            training_value.pop("data", None),
            base_dir=base_dir,
        )
        training_optimizer = _merge_dataclass(
            OptimizerConfig,
            base.training.optimizer,
            training_value.pop("optimizer", None),
        )
        training = _merge_dataclass(
            TrainingConfig,
            replace(base.training, data=training_data, optimizer=training_optimizer),
            training_value,
            base_dir=base_dir,
        )
        preprocessing = _merge_dataclass(
            PreprocessingConfig,
            base.preprocessing,
            value.get("preprocessing"),
            base_dir=base_dir,
        )
        inference = _merge_dataclass(
            InferenceConfig,
            base.inference,
            value.get("inference"),
            base_dir=base_dir,
            tuple_fields={"npz"},
        )
        evaluation = _merge_dataclass(
            EvaluationConfig,
            replace(base.evaluation, checkpoint=inference.checkpoint),
            value.get("evaluation"),
            base_dir=base_dir,
        )
        return cls(
            schema_version=str(value.get("schema_version", base.schema_version)),
            architecture=architecture,
            runtime=runtime,
            training=training,
            preprocessing=preprocessing,
            inference=inference,
            evaluation=evaluation,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        return cls.from_dict(
            json.loads(config_path.read_text(encoding="utf-8")),
            base_dir=config_path.parent,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))

    def with_overrides(self, **changes: Any) -> "ExperimentConfig":
        return replace(self, **changes)


MappingLike = Any


def _expand_path(value: object, base_dir: Path | None = None) -> object:
    if not isinstance(value, (str, Path)):
        return value
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if base_dir is not None and not expanded.is_absolute():
        expanded = (base_dir / expanded).resolve()
    elif "${" not in str(expanded):
        expanded = expanded.resolve()
    return expanded


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    return value


def _merge_dataclass(
    dataclass_type: type,
    default: Any,
    value: Any,
    *,
    base_dir: Path | None = None,
    tuple_fields: set[str] | None = None,
) -> Any:
    if value is None:
        return default
    if not isinstance(value, dict):
        raise TypeError(f"{dataclass_type.__name__} must be a mapping")
    tuple_fields = tuple_fields or set()
    changes: dict[str, Any] = {}
    for key, current in value.items():
        if key.startswith("_") or key not in default.__dataclass_fields__:
            raise ValueError(f"unknown {dataclass_type.__name__} field: {key}")
        parsed = _expand_path(
            current,
            base_dir,
        ) if key in {
            "output_dir",
            "checkpoint",
            "manifest",
            "compact_dir",
            "output",
            "train_manifest",
            "train_compact_dir",
        } else current
        if key in tuple_fields:
            parsed = tuple(Path(item) for item in parsed) if isinstance(parsed, (list, tuple)) else (Path(parsed),)
        if key == "npz" and not isinstance(parsed, tuple):
            parsed = (Path(parsed),)
        changes[key] = parsed
    return replace(default, **changes)
