# Gaussian-direct128 architecture

Gaussian-direct128 is a plug-and-play extension of the official PeSTo protein-binding model.

## Forward path

1. Atom features and local topology are encoded by the official PeSTo atom encoder and state-update stack.
2. Atom states are pooled to residue states by PeSTo's state-pooling layer.
3. The decoder's penultimate residue representation is passed to the Gaussian interaction block.
4. The Gaussian block performs:
   - residue-to-surface Gaussian projection;
   - residue-pair Gaussian correction;
   - k16/k32 multigraph GVP surface encoding;
   - connected surface patch pooling;
   - surface-to-residue attention;
   - three residue/surface co-update rounds.
5. A direct five-channel protein/nucleic/ion/ligand/lipid head reads the final Gaussian residue state.

The released E11 checkpoint uses `architecture=vg`, `gaussian_dim=128`, `direct_protein_head=true`, `direct_protein_residual=false`, and `bypass_gaussian=false`.

## Parameter budget

The training contract records 3,355,001 total parameters: 1,474,957 in the PeSTo backbone and 1,880,044 in the Gaussian/direct path. The architecture-matched bypass control and parameter-matched MLP are therefore important for separating geometry from capacity.

## Causal route interpretation

The pair-Gaussian module exposes route diagnostics and matched interventions. The released result tables include top-route, random-route, long-route, and short-route perturbations. These support model-level causal dependence, not a wet-laboratory allosteric mechanism by themselves.

## Application architecture

The public engineering surface is deliberately small:

```text
ExperimentConfig
   └── GaussianDirectApp
         ├── train()    -> GaussianDirectDataModule + LightningModule
         ├── infer()    -> residue-level CSV and summary
         ├── evaluate() -> pooled/case metrics
         ├── preprocess() -> compact-NPZ cache
         └── verify()   -> checkpoint contract
```

`main.py` is only a dispatcher. New experiments should add a JSON config rather than another standalone training or inference script. Shared IO, hashing, device selection, and seeding utilities live in `src/gaussian_direct/utils.py`.
