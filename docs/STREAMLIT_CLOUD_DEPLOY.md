# Streamlit Cloud deploy checklist (~1 GB RAM)

## Required artifacts (Git LFS)

| File | Approx. size | Notes |
|------|----------------|-------|
| `artifacts/manuscript/independent_ligand/rf/model_seed42_cloud.pkl` | ~10 MB | **Required** — run `py -3 scripts/shrink_rf_for_cloud.py` |
| `artifacts/manuscript/ligand_feature_lookup.sqlite` | ~379 MB | **Required** for training-aligned ligands |
| `artifacts/manuscript/manifest.json` | small | Feature column list |

## Do NOT deploy (causes OOM on `git lfs pull` or predict)

Remove these from the **GitHub** repo (or stop tracking with LFS) if possible:

- `artifacts/manuscript/ligand_feature_lookup.joblib` (~346 MB)
- `artifacts/manuscript/independent_ligand/rf/model_seed42.pkl` (~166 MB, 1000 trees)
- `artifacts/manuscript/independent_ligand/ensemble/` (~213 MB)

Cloud uses **SQLite + `_cloud.pkl` only**.

## `requirements.txt`

Use the slim list (no xgboost, lightgbm, mordred, plotly, py3dmol) — see `requirements-cloud.txt`.

## Secrets

- `DATA_DRIVE_FILE_ID` → **~137 MB** slim zip (pockets + ML_code)
- `DATA_EXTRACTED_SUBDIR` = `GPCRtryagain-inference-slim`

## Rebuild cloud RF after export

```powershell
py -3 scripts/shrink_rf_for_cloud.py --trees 50
git add artifacts/manuscript/independent_ligand/rf/model_seed42_cloud.pkl
git lfs push
```
