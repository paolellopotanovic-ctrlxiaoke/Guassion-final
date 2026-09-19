# Gaussian-direct128

Gaussian-direct128 is a geometry-aware, partner-free protein-binding residue predictor. It extends the official PeSTo residue representation with a 128-dimensional Gaussian residue/surface interaction path and a direct five-channel output head.

This repository is organized as one typed application rather than a collection of historical scripts: `main.py` dispatches every operation, `GaussianDirectApp` owns the use cases, and `configs/default.json` is the single experiment contract.

## Install

```bash
conda env create -f environment.yml
conda activate gaussian-direct128
```

## Unified entrypoint

```bash
python main.py --config configs/default.json show-config
python main.py --config configs/default.json verify
python main.py --config configs/default.json infer --device cpu
python main.py --config configs/default.json evaluate --device cpu
```

Train after setting the external data paths:

```bash
TRAIN_MANIFEST=/absolute/path/train.csv \
TRAIN_COMPACT_DIR=/absolute/path/semantic_npz \
python main.py --config configs/default.json train
```

Build compact NPZ inputs:

```bash
python main.py preprocess \
  --manifest /path/manifest.csv \
  --output-dir /path/semantic_npz \
  --workers 16 --payload training
```

## Configuration model

`src/gaussian_direct/config.py` defines typed dataclasses for:

- architecture;
- runtime and distributed training;
- training data and optimizer schedule;
- inference;
- evaluation.

Change `configs/default.json` and instantiate the application directly when embedding it in another project:

```python
from gaussian_direct import GaussianDirectApp

app = GaussianDirectApp("configs/default.json")
app.infer()
```

The same architecture section also carries the reproducible A3–A7 controls:
`bypass_gaussian`, `parameter_matched_mlp`, `ablation_no_surface_vector`,
`ablation_no_residue_feedback`, and `co_update_rounds`. A1 is the independent
PeSTo-scratch baseline rather than a Gaussian-direct128 variant.

## Repository layout

```text
main.py                    Unified train/infer/evaluate/preprocess entry
configs/default.json       Paper-scale typed config
configs/smoke.json         CPU smoke-test config
src/gaussian_direct/       App, config, data, Lightning runtime, shared utils
src/gaussian_direct/model.py
                           PeSTo-plus-Gaussian model
src/gaussian_direct/modules/
                           Gaussian and geometric components
src/gaussian_direct/pesto/
                           Minimal audited PeSTo-derived runtime
src/gaussian_direct/preprocessing.py
                           Compact-NPZ preprocessing worker
checkpoints/               Released E11 checkpoint and contract
examples/                  Three holo/APO/AF2 input fixtures
results/                   Principal machine-readable result tables
```

Historical scripts and reports are intentionally absent from Git; local-only work is ignored under `archive/`.

## Main case-AUPR results

| Panel | Gaussian-direct128 | PeSTo official | PeSTo-scratch | Direct−official | Direct−scratch |
|---|---:|---:|---:|---:|---:|
| MMseq holo 988 | **0.7915** | 0.7380 | 0.7769 | +0.0535 | +0.0146 |
| MMseq APO/AF2 874 | **0.7281** | 0.6770 | 0.7167 | +0.0511 | +0.0114 |

PeSTo-scratch belongs in architecture-control analyses, not the first deployed-model comparison.
The complete panel, epoch, stratified-gain, and ablation tables are under `results/`.

## Data and checkpoint policy

Full PINDER manifests, PDB coordinates, and compact NPZ caches are not redistributed. `data/data_contracts.json` records their identities and hashes. The released E11 checkpoint SHA-256 is:

```text
10ec32e0138c8150ee08f2b35e17115b4d90cdd1ad915f46e1e36134bc495a7c
```

## Third-party dependency

The baseline encoder and preprocessing runtime derive from [PeSTo](https://github.com/LBM-EPFL/PeSTo). See `docs/THIRD_PARTY.md` for the pinned upstream commit, local changes, CC BY-NC-SA 4.0 license, and citation:

```text
Krapp, L.F., Abriata, L.A., Cortés Rodriguez, F. et al. PeSTo: parameter-free geometric
deep learning for accurate prediction of protein binding interfaces. Nat Commun 14,
2175 (2023). https://doi.org/10.1038/s41467-023-37701-8
```
