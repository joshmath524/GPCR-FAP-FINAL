# Manuscript-faithful Streamlit setup

The Streamlit GUI uses **only** manuscript exports under `artifacts/manuscript/` (independent ligand, scaffold, LORO, stacking ensemble). Legacy `artifacts/demo_*` bundles (**2,103** features) are **not** exposed in the app.

## Manuscript vs legacy demo (reference only)

| Item | Manuscript (S9–S21) | Legacy `artifacts/demo_*` (not in GUI) |
|------|---------------------|----------------------------------------|
| Training rows | 40,611 pairs | ~1,500 (demo export) |
| Ligand features | RDKit + **Mordred** from enriched CSVs (~1,600 cols) | 10 RDKit + 2048 Morgan only (**2,103** total) |
| Independent ligand | RF/XGB/LGB fit on **dev80** | Unclear / mixed seed files |
| Scaffold | Separate train split | Not bundled |
| LORO | **68 models** (one per held-out receptor) | Not bundled |
| Ensemble | **Stacking** (3 bases → logistic meta on 9 probs) | Partial `StackingClassifier` + wrong averages |

Export manuscript models into `artifacts/manuscript/` before running the app (see below).

---

## What you need (checklist)

### 1. Training data (on disk)

Per receptor folder (68 names like `beta2`, `D2`, `5-HT1A`):

- `{receptor}_agonist_enriched.csv`
- `{receptor}_antagonist_enriched.csv`
- `{receptor}_non_active_compounds.csv`

These are your Supp. Files 4–6 style tables (RDKit + Mordred columns), e.g. `M2_agonist_enriched.csv` in Downloads.

Parent folder = **`MANUSCRIPT_DATA_ROOT`** (e.g. `C:\Users\Piano\Downloads` if each receptor is a subfolder).

### 2. Training code

- `shared_utilities.py` (imported by Code S9–S21)
- `Code S9` – RF independent ligand  
- `Code S10` / `S11` – XGB / LightGBM  
- `Code S15`–`S17` – scaffold  
- `Code S18`–`S20` – LORO  
- `Code S21` – stacking ensemble  

Set **`MANUSCRIPT_ML_ROOT`** to the folder containing `shared_utilities.py`.

### 3. Receptor pocket files

`Josh_Receptor_Features/<receptor>/` in the Streamlit repo (already present).

### 4. Split file (for independent ligand / ensemble)

`splits/final_independent_ligand_80_20_canonical_smiles.json` from your training run.  
Set **`MANUSCRIPT_SPLIT_DIR`** if it is not under the data root.

---

## Step 1 — Export models (run once, on your training PC)

```powershell
cd "C:\path\to\GPCR-FAP-main"

$env:MANUSCRIPT_DATA_ROOT = "C:\Users\joshmatchem\Downloads"   # parent of beta2\, M2\, ...
$env:MANUSCRIPT_ML_ROOT    = "C:\path\to\ML_code"              # shared_utilities.py
$env:GPCR_EXPORT_ROOT       = "C:\Users\joshmatchem\Music\GPCR-FAP-main (2)\GPCR-FAP-main"

py -3 scripts\export_manuscript_models.py --regime independent_ligand --model rf --seeds 42,43,44,45,46,47,48,49,50,51
```

Repeat for:

- `--model lightgbm` / `--model xgboost`
- `--regime scaffold` (extend script using S15–S17 logic)
- `--regime loro` (68 × seeds — large)
- `--regime ensemble` (Code S21 stacking bundle)

### Expected output layout

```text
artifacts/manuscript/
  manifest.json                 # feature_columns + metadata (REQUIRED)
  independent_ligand/
    rf/model_seed42.pkl …
    lightgbm/…
    xgboost/…
  scaffold/
    rf/…
  loro/
    rf/beta2/model_seed42.pkl
    rf/D2/…
    …
  ensemble/
    independent_ligand/
      stacking_seed42.pkl       # StackingEnsemblePredictor or sklearn Pipeline
```

---

## Inference features (at predict time)

Manuscript models expect **~6,600** columns from `manifest.json` (union of ligand columns across receptors + 31 receptor + 14 interactions).

**Training ligand columns come from `*_NEW.xlsx` only** (~3,300 cols per workbook), not the slimmer `*_enriched.csv` files.

For each SMILES + receptor, the app loads:

- **Ligand:** row from `{receptor}_*_enriched_NEW.xlsx` when SMILES is in that receptor’s workbooks (`src/gpcr/new_workbook_ligand.py`); otherwise RDKit + Mordred fallback for novel SMILES
- **Receptor:** 31 pocket features from `Josh_Receptor_Features/<folder>/` under **`GPCR_DATA_ROOT`**
- **Interactions:** 14 products (ligand × receptor), same as Code S9

Set **`GPCR_DATA_ROOT`** to **Delete-Copy** (must contain per-receptor folders + `Josh_Receptor_Features/`).

Optional: build a global lookup (slow) with `py -3 scripts/build_manuscript_ligand_lookup.py`.

**Streamlit Cloud (~1 GB RAM):** do not `joblib.load` the full lookup. After building joblib, create SQLite:

```powershell
py -3 scripts/build_ligand_lookup_sqlite.py
git add artifacts/manuscript/ligand_feature_lookup.sqlite artifacts/manuscript/ligand_feature_lookup_meta.json
git lfs push
```

Commit `ligand_feature_lookup.sqlite` via **Git LFS** (~380 MB on disk; only **one SMILES row** is loaded into RAM per Predict).

Verify locally:

```powershell
$env:GPCR_DATA_ROOT = "C:\...\GPCRtryagain - Delete - Copy"
$env:MANUSCRIPT_ML_ROOT = "C:\...\GPCRtryagain - Delete - Copy\ML_code"
py -3 scripts/smoke_test_manuscript_features.py
```

You should see **inference nonzero ~1200+** (not ~60) and **0 ligand column mismatches** vs an enriched CSV row.

---

## Step 2 — Run Streamlit locally

```powershell
cd "C:\Users\joshmatchem\Music\GPCR-FAP-main (2)\GPCR-FAP-main"
pip install -r requirements.txt
pip install mordred                    # required for manuscript ligand columns
$env:GPCR_DATA_ROOT = "C:\...\GPCRtryagain - Delete - Copy"
$env:MANUSCRIPT_ML_ROOT = "C:\...\GPCRtryagain - Delete - Copy\ML_code"
py -3 -m streamlit run streamlit_app.py
```

In the app:

1. **Evaluation regime:** Independent ligand / Scaffold / LORO  
2. **Model:** RF, LightGBM, XGBoost, or Ensemble  
3. **Seed:** 42–51 (match paper seeds)  
4. Receptor + SMILES → Predict  

---

## Step 3 — Update the public site (gpcrfap.streamlit.app)

1. Commit and push `artifacts/manuscript/` + code changes to GitHub.  
2. Large files may need **Git LFS** or a release ZIP + download on first run.  
3. Streamlit Cloud rebuilds from the repo (see share.streamlit.io).

You need **write access** to https://github.com/sivaGU/GPCR-FAP or deploy your own fork.

---

## Regime behavior in the app

| Regime | Manuscript meaning | Model used at predict time |
|--------|-------------------|----------------------------|
| **Independent ligand** | Trained on dev80; tested on held-out 20% SMILES | One global model per seed |
| **Scaffold** | Trained without test scaffolds | One global model per seed |
| **LORO** | Trained without that receptor’s ligands | **Per-receptor** model (select receptor in UI) |
| **Ensemble** | Stacking on independent ligand | Single stacking object per seed |

**LORO + Ensemble:** The paper’s stacking table uses the **independent ligand** split, not LORO. For LORO, pick RF/LightGBM/XGBoost only.

---

## Verify before claiming “matches paper”

1. `manifest.json` has `n_features` ≈ 6600 (not 2103).  
2. `py -3 scripts/smoke_test_manuscript_features.py` passes (ligand descriptors match enriched CSV).  
3. Independent 20% test: re-run Code S9 metrics and compare to Streamlit probabilities on the same pairs.

---

## Contact / extend export script

`scripts/export_manuscript_models.py` currently implements **RF + independent_ligand** fully. Extend it by copying the final `model.fit(...)` blocks from S10, S11, S15–S17, S18–S20, and S21 into the same script so all regimes export automatically.
