# Data splits

Split manifests for this project are committed here as reproducibility artifacts (see
`AGENTS.md` — cached images and raw downloads are gitignored, splits are not).

Populated in Phase 1 (`scripts/make_splits.py`):
- `aptos_train.csv`, `aptos_val.csv` — stratified by grade, seed 42
- `messidor2_test.csv` — untouched until the single Phase 2 evaluation run

To be documented here once written, per the domain-shift risk in `AGENTS.md`:
- The exact Messidor-2 label file used and its provenance
- The exact grade mapping applied to reconcile Messidor-2 labels with the 0-4 ICDR scale
  used for APTOS/IDRiD training
