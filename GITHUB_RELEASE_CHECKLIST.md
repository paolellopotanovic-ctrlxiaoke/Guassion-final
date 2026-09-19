# GitHub release checklist

1. Use a private remote until institutional and PINDER-derived fixture authorization is cleared.
2. Run `python main.py --config configs/default.json verify`.
3. Run `pytest -q`.
4. Run `python main.py --config configs/default.json infer --device cpu`.
5. Confirm no absolute machine paths, tenant identifiers, tokens, or private keys are present.
6. Confirm the E11 SHA-256 in `README.md`, `data/data_contracts.json`, and `SHA256SUMS` matches.
7. Commit only the typed application, released checkpoint, minimal fixtures, and machine-readable results.

```bash
git remote add origin git@github.com:OWNER/gaussian-direct128.git
git push -u origin main
```
