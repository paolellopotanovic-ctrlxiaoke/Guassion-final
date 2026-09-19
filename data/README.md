# Data and checkpoint contracts

This repository intentionally does not redistribute the full PINDER training set, the 20,801 compact NPZ structures, or the three complete evaluation panels. Those files are large and subject to upstream dataset terms.

`data_contracts.json` records:

- the exact training manifest path, row count, and SHA-256;
- the three clean evaluation panel manifests, row counts, and SHA-256 hashes;
- the E11 checkpoint path and SHA-256;
- the local cache paths used in the source laboratory.

To reproduce the data:

1. Obtain the contracted PINDER-derived manifests on the source machine.
2. Verify them against `data_contracts.json`.
3. Run `python main.py preprocess` with the training or panel manifest.
4. Train or evaluate using those local paths.

The files under `examples/` are three small holo/APO/AF2 fixtures for smoke tests only; they are not a benchmark and must not be used for model selection.
