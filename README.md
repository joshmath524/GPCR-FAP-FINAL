# GPCR-FAP GUI

Streamlit app for **Class A GPCR** receptor–ligand **multiclass** functional activity prediction (Agonist / Antagonist / Inactive), with optional **SMINA** docking and **py3Dmol** visualization.

- **Live app:** https://gpcrfap.streamlit.app/
- **Repository:** https://github.com/joshmath524/GPCR-FAP-FINAL

## Features

- Predict from SMILES or structure files (SDF, MOL, PDB, PDBQT, MOL2, CSV) for ~70 bundled receptors.
- Manuscript models: RF, LightGBM, XGBoost, stacking ensemble (`artifacts/manuscript/`, **6,633** features).
- Optional SMINA top-pose docking and 3D complex view after prediction.

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501 → choose **evaluation regime**, **model**, and **seed** → select receptor → enter SMILES → **Predict**.

## Manuscript models

Exports live under `artifacts/manuscript/` (`manifest.json`, `ligand_feature_lookup.sqlite`, and `model_seed*.pkl` per regime/model).

For local training/export, set (Windows example):

```bash
set GPCR_DATA_ROOT=C:\path\to\folder\with\Josh_Receptor_Features
set MANUSCRIPT_ML_ROOT=C:\path\to\ML_code
```

Re-export example:

```bash
py -3 scripts/export_manuscript_models.py --regime independent_ligand --model rf --seeds 42
```

Details: `docs/MANUSCRIPT_STREAMLIT_SETUP.md`. Streamlit Cloud: `docs/STREAMLIT_CLOUD_DEPLOY.md`.

## Requirements

- Python 3.10+ (3.10–3.12 tested)
- `requirements.txt` locally; Cloud uses `requirements-cloud.txt`
- Receptor data: `Josh_Receptor_Features/` in repo root, or set `GPCR_DATA_ROOT`
- Docking: SMINA in `docking_assets/` (Linux Cloud bundle: `docking_assets/smina_linux/`)

## Project layout

```
├── streamlit_app.py
├── artifacts/manuscript/
├── Josh_Receptor_Features/
├── docking_assets/
│   ├── receptor_pdbs/
│   ├── receptor_grid_boxes.json
│   └── smina_linux/          # Streamlit Cloud
└── src/gpcr/
```

## CLI

```bash
python -m src.gpcr.cli --receptor beta2 --ligand "CCO" --output out.csv
```

## Citation & contact

Cite the GPCR Class A functional activity prediction manuscript / Zenodo DOI when publishing.

**Dr. Sivanesan Dakshanamurthy** — sd233@georgetown.edu
