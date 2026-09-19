# Reproduction guide

## Install and verify

```bash
conda env create -f environment.yml
conda activate gaussian-direct128
python main.py --config configs/default.json verify
python main.py --config configs/default.json infer --device cpu
```

The inference smoke test processes three bundled holo/APO/AF2 structures and 434 residues.

## Prepare data

Full manifests and semantic NPZ caches remain outside Git. Verify their hashes against `data/data_contracts.json`, then rebuild when needed:

```bash
python main.py preprocess \
  --manifest /path/manifest.csv \
  --output-dir /path/semantic_npz \
  --workers 16 \
  --payload training
```

## Train

The default paper-scale contract is two GPUs, local batch 4, global batch 8, 20 epochs, AdamW, warmup ratio 0.05, minimum LR ratio 0.1, and seed 42.

```bash
TRAIN_MANIFEST=/absolute/path/train.csv \
TRAIN_COMPACT_DIR=/absolute/path/semantic_npz \
python main.py --config configs/default.json train
```

Run the bundled CPU training contract:

```bash
python main.py --config configs/smoke.json train \
  --accelerator cpu --devices 1 --precision 32
```

The training runtime maps directly to the original audited semantics:

| Contract | Implementation |
|---|---|
| deterministic global-batch DDP sampler | `GaussianDirectDataModule` |
| PeSTo/Gaussian AdamW groups | `GaussianDirectLightningModule.configure_optimizers` |
| warmup-cosine step schedule | `GaussianDirectLightningModule.configure_optimizers` |
| global valid-residue BCE + semantic loss | `GaussianDirectLightningModule.training_step` |
| gradient norm 1.0 | `configure_gradient_clipping` |
| legacy epoch checkpoint | `model_epoch_XX.pt` |

The clean Lightning path is the engineering runtime. Full-scale bit-for-bit identity with the released native run is not claimed until an end-to-end same-seed replay is completed.

## Evaluate

```bash
python main.py --config configs/default.json evaluate \
  --manifest /path/panel.csv \
  --compact-dir /path/panel_npz \
  --checkpoint /path/model.pt \
  --output results/panel_metrics.json \
  --device cuda:0
```

## Selection boundary

E11 was selected on three clean evaluation panels. Prospective deployment should predefine an epoch or use a separate model-selection split.
