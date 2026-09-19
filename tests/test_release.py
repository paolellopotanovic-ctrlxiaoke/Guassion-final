from pathlib import Path
import csv
import hashlib
import json

from gaussian_direct.config import ExperimentConfig, PESTO_MODEL_CONFIG
from gaussian_direct.data import CompactGaussianDirectDataset
from gaussian_direct import GaussianDirectApp


ROOT = Path(__file__).resolve().parents[1]


def test_pesto_configuration() -> None:
    assert PESTO_MODEL_CONFIG["em"] == {"N0": 30, "N1": 32}
    assert len(PESTO_MODEL_CONFIG["sum"]) == 32
    assert PESTO_MODEL_CONFIG["dm"]["N2"] == 5


def test_example_manifest_loads() -> None:
    dataset = CompactGaussianDirectDataset(
        ROOT / "examples/example_manifest.csv",
        ROOT / "examples/compact_npz",
    )
    assert len(dataset) == 3
    assert {item["input_state"] for item in dataset} == {"holo", "apo", "af2"}


def test_default_config_is_typed_and_resolved() -> None:
    config = ExperimentConfig.from_file(ROOT / "configs/default.json")
    assert config.schema_version == "gaussian_direct128_app_config_v1"
    assert config.architecture.gaussian_dim == 128
    assert config.architecture.parameter_matched_mlp is False
    assert config.runtime.devices == 2
    assert config.training.optimizer.local_batch_size == 4
    assert config.preprocessing.payload == "training"
    assert config.inference.checkpoint == ROOT / "checkpoints/direct128_e11.pt"
    assert config.evaluation.manifest == ROOT / "examples/example_manifest.csv"


def test_smoke_config_overrides_nested_defaults() -> None:
    config = ExperimentConfig.from_file(ROOT / "configs/smoke.json")
    assert config.architecture.gaussian_dim == 32
    assert config.runtime.accelerator == "cpu"
    assert config.training.data.expected_train_rows == 3
    assert config.training.optimizer.epochs == 1


def test_runtime_uses_one_application_package() -> None:
    import gaussian_direct.model

    assert gaussian_direct.model.__name__ == "gaussian_direct.model"
    assert not (ROOT / "src/m4").exists()
    assert not (ROOT / "src/pesto_gaussian").exists()
    assert not (ROOT / "third_party").exists()
    assert not (ROOT / "scripts").exists()


def test_checkpoint_contract_is_immutable() -> None:
    contract = json.loads(
        (ROOT / "data/data_contracts.json").read_text(encoding="utf-8")
    )["checkpoint"]
    checkpoint = ROOT / contract["path"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert contract["epoch"] == 11
    assert digest == contract["sha256"]


def test_application_verifies_complete_release_manifest() -> None:
    report = GaussianDirectApp(ROOT / "configs/default.json").verify()
    assert report["release_files"] == 61
    assert report["checkpoint_epoch"] == 11
    assert report["examples"] == 3


def test_principal_result_tables_have_expected_contracts() -> None:
    with (ROOT / "results/three_clean_panel_model_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        panels = {row["panel"] for row in csv.DictReader(handle)}
    assert panels == {
        "bc30_holo_clean820",
        "mmseq_holo_clean1000",
        "mmseq_state_clean886",
    }

    with (ROOT / "results/direct128_epoch3_17_three_panel_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    epochs = {int(row["epoch"]) for row in rows}
    assert len(rows) == 48
    assert epochs == set(range(3, 20)) - {12}

    with (ROOT / "results/architecture_ablation_epoch7_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        variants = {row["variant"] for row in csv.DictReader(handle)}
    assert variants == {f"A{index}" for index in range(1, 8)}
