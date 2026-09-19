# Audited PeSTo-derived runtime

## Source and license

`src/gaussian_direct/pesto/` contains the minimal PeSTo-derived files required
to replay the official residue encoder and compact-NPZ preprocessing contract
used by Gaussian-direct128. The runtime is integrated as a typed subpackage;
there is no separate `third_party/` import-path shell.

- Upstream project: <https://github.com/LBM-EPFL/PeSTo>
- Upstream commit used for verbatim files: `ba651aa29aaa839d0ee2c458dee90ca1f934d90a`
- Upstream license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
- Upstream reference: Krapp *et al.*, *Nature Communications* 14, 2175 (2023),
  <https://doi.org/10.1038/s41467-023-37701-8>

## Local provenance and changes

After moving the files into `gaussian_direct.pesto`, the following remain
byte-identical to the upstream commit above:

- `src/gaussian_direct/pesto/encoding.py`
- `src/gaussian_direct/pesto/structure.py`
- `src/gaussian_direct/pesto/structure_io.py`

`operations.py` starts from the same upstream commit and adds an optional
`edge_attr_dim` / `edge_attr_nn` input so the released model can replay edge-conditioned
attention without changing the original input contract.

`config.py` and `model.py` are the upstream files with only relative-package import
updates. `dataset.py` and `surface_cache.py` come from the laboratory PeSTo-main
snapshot used to build the compact training cache, with relative-package imports;
`dataset.py` also removes an unused `collate_batch_features` import, and
`surface_cache.py` records that optional historical atom-semantic adapters are absent.
These files are retained because they document the cache and dataset contract used by
the released training run. The public snapshot does not provide an independent Git
commit identifier for the laboratory snapshot.

No official PeSTo model weights are redistributed here. The checkpoint in the parent
repository is a Gaussian-direct128 model trained for this project.
