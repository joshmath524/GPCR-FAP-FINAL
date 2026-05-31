# GPCR-FAP GUI

GPCR-FAP is a Streamlit GUI for **Class A GPCR** receptor-ligand **multiclass** functional activity prediction (Agonist / Antagonist / Inactive), with optional **SMINA pose generation** and 3D receptor-ligand visualization.

Hosted app: https://gpcrfap.streamlit.app/

Repository: https://github.com/sivaGU/GPCR-FAP

## What This App Provides

- **Functional activity inference** from SMILES (or common structure files) for each bundled receptor target.
- **Models:** Manuscript Random Forest, LightGBM, XGBoost, and stacking ensemble under `artifacts/manuscript/` (**6,633** features per row).
- **Receptor assets:** `Josh_Receptor_Features/` — pocket CSVs, conservation summaries, and PDBs per target (~70 folders).
- **Post-prediction pose generation:** SMINA top-pose generation from the predicted ligand (SMILES input supported), using receptor-specific grid centers from each `*_ligand_only.pdb`.
- **3D docked complex view:** Receptor cartoon (tan) plus docked ligand pose in sticks (py3Dmol).

## Quick Start

1. Clone the repository and open the project folder.
2. Create a virtual environment and install dependencies (see **Run Locally**).
3. Launch Streamlit and open the prediction page.
4. Choose a model, select a receptor, enter SMILES or upload a structure file, and run **Predict**.
5. Review class probabilities.
6. Optionally click docking to generate and view a SMINA top pose.

## Run Locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Open `http://localhost:8501`.

## Manuscript models (required for the GUI)

The Streamlit app uses **only** manuscript exports in `artifacts/manuscript/` (not the legacy `artifacts/demo_*` bundles).

### Use manuscript models in Streamlit

Set environment variables in the same terminal before launching:

```bash
set GPCR_DATA_ROOT=C:\path\to\GPCRtryagain - Delete - Copy
set MANUSCRIPT_ML_ROOT=%GPCR_DATA_ROOT%\ML_code
```

Then launch:

```bash
streamlit run streamlit_app.py
```

In the app choose:
- **Evaluation regime:** Independent ligand, Scaffold split, or LORO (whichever you exported)
- **Model:** RF / XGBoost / LightGBM / Ensemble
- **Seed:** 42

### Re-export Manuscript Ensemble

Canonical command:

```bash
py -3 scripts/export_manuscript_models.py --regime ensemble --model ensemble --seeds 42
```

Resume behavior:
- OOF folds and final model stages are checkpointed in `artifacts/manuscript/independent_ligand/ensemble/checkpoint_seed42.joblib`.
- Re-running the same command resumes completed stages.
- `--fresh` forces a full restart from scratch.

## Requirements

- Python **3.10+** recommended (3.10–3.12 tested).
- Dependencies in `requirements.txt` (RDKit, scikit-learn, LightGBM, XGBoost, Streamlit, py3Dmol, etc.).
- **Trained models:** `artifacts/manuscript/<regime>/<model>/model_seed42.pkl`, plus `manifest.json` and `ligand_feature_lookup.sqlite`. See `docs/MANUSCRIPT_STREAMLIT_SETUP.md`.
- **Docking engine:** SMINA binary available in `docking_assets/` or system `PATH`.

## Supported Inputs

- **Ligands:** SMILES string, or upload **SDF, MOL, PDB, PDBQT, MOL2**, or **CSV** (first SMILES column).
- **Receptors:** Selected from folders under `Josh_Receptor_Features/`.

## Receptor Data Path

If `Josh_Receptor_Features/` is in the **repository root**, no configuration is needed. If you keep data under a sibling `GUI_Folder/` next to this project, the app can auto-detect that layout when `GPCR_DATA_ROOT` is unset.

To point elsewhere:

```bash
set GPCR_DATA_ROOT=C:\path\to\folder\that\contains\Josh_Receptor_Features
```

## Project Structure

```
GPCR-FAP/
├── streamlit_app.py
├── README.md
├── requirements.txt
├── data/
│   └── gpcr_class_a_receptors.txt
├── src/
│   └── gpcr/
│       ├── predict.py
│       ├── manuscript_features.py
│       ├── cloud_predict.py
│       ├── structure_view.py
│       └── docking.py
├── artifacts/
│   └── manuscript/                    # regime / model / seed exports
├── docking_assets/
│   ├── smina / smina.exe
│   └── receptor_grid_boxes.json
├── docking_results/                   # generated at runtime
└── Josh_Receptor_Features/
    └── <receptor_name>/
        ├── *_receptor_only.pdb
        ├── *_ligand_only.pdb
        ├── *_pocket_residues_with_conservation.csv
        └── ...
```

## Large Repositories

`Josh_Receptor_Features/` contains many PDBs. If GitHub size limits are an issue, distribute data via **Git LFS**, a **release ZIP**, or an external download and set `GPCR_DATA_ROOT`.

## Notes on Validation Scope

- Outputs include **multiclass functional activity** predictions (agonist / antagonist / inactive).
- Docking output is a **SMINA-generated top pose** intended for screening visualization and ranking, not a substitute for full physics-based validation.

## Citation

Cite the **GPCR-FAP / Class A functional activity prediction** manuscript, Zenodo release, or DOI when publishing.

## Contact

Questions, issues, or collaboration requests: **Dr. Sivanesan Dakshanamurthy** — sd233@georgetown.edu
