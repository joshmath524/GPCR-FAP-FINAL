#!/usr/bin/env python3
"""
Export trained models for the GPCR-FAP Streamlit app (manuscript-faithful).

Example (PowerShell):
  $env:MANUSCRIPT_DATA_ROOT = "C:\\...\\GPCRtryagain - Delete - Copy"
  $env:MANUSCRIPT_ML_ROOT    = "C:\\...\\GPCRtryagain - Delete - Copy\\ML_code"
  $env:MANUSCRIPT_SPLIT_DIR  = "C:\\...\\MLcodes\\pan_gpcr_results\\splits"
  $env:GPCR_EXPORT_ROOT       = "C:\\...\\GPCR-FAP-main"
  py -3 scripts/export_manuscript_models.py --regime independent_ligand --model rf --seeds 42
  py -3 scripts/export_manuscript_models.py --regime ensemble --model ensemble --seeds 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Code S9 trial-76 / GUI prompt (n_estimators=1000)
RF_PARAMS_S9 = {
    "n_estimators": 1000,
    "max_depth": 30,
    "min_samples_split": 9,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "criterion": "entropy",
    "bootstrap": False,
    "class_weight": None,
    "n_jobs": -1,
    "oob_score": False,
}

XGB_PARAMS_S10 = {
    "n_estimators": 1000,
    "learning_rate": 0.011729508579500059,
    "max_depth": 10,
    "min_child_weight": 1.0084867831102238,
    "subsample": 0.9804440329670632,
    "colsample_bytree": 0.7518705540721746,
    "gamma": 0.2959731631374173,
    "reg_alpha": 1.5981756177121895,
    "reg_lambda": 1.8565747985457355,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 0,
}

LGB_PARAMS_S11 = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "None",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.022675095411901637,
    "num_leaves": 278,
    "max_depth": -1,
    "min_child_samples": 81,
    "subsample": 0.8458649196412246,
    "colsample_bytree": 0.9888514643504509,
    "lambda_l1": 1.8080951629101958,
    "lambda_l2": 0.8806303532495801,
    "n_jobs": -1,
}


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def _export_root() -> Path:
    env = os.environ.get("GPCR_EXPORT_ROOT", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


def _import_training_stack():
    ml_root = os.environ.get("MANUSCRIPT_ML_ROOT", "").strip()
    if not ml_root:
        raise SystemExit("Set MANUSCRIPT_ML_ROOT to the folder containing shared_utilities.py.")
    p = Path(ml_root)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import shared_utilities as su  # noqa: F401

    return su


def _import_stacking_class(export_root: Path):
    if str(export_root) not in sys.path:
        sys.path.insert(0, str(export_root))
    from src.gpcr.manuscript_bundle import StackingEnsemblePredictor

    return StackingEnsemblePredictor


def _training_data_root() -> Path:
    p = _env_path("MANUSCRIPT_DATA_ROOT")
    if p is None or not p.is_dir():
        raise SystemExit("Set MANUSCRIPT_DATA_ROOT to the parent folder of receptor subfolders.")
    return p


def _split_dir(data_root: Path) -> Path:
    p = _env_path("MANUSCRIPT_SPLIT_DIR")
    if p is not None:
        return p
    candidates = [
        data_root / "MLcodes" / "pan_gpcr_results" / "splits",
        data_root / "ML_code" / "outputs" / "splits",
    ]
    for c in candidates:
        if (c / "final_independent_ligand_80_20_canonical_smiles.json").exists():
            return c
    return candidates[0]


def _lgb_fit_matrix(X) -> np.ndarray:
    """NumPy matrix for LightGBM (column names may contain JSON-special chars)."""
    if hasattr(X, "to_numpy"):
        return np.asarray(X.to_numpy(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _align_proba(probs: np.ndarray, model_classes: np.ndarray, n_classes: int = 3) -> np.ndarray:
    out = np.zeros((probs.shape[0], n_classes), dtype=np.float64)
    for i in range(n_classes):
        if i in model_classes:
            col = int(np.where(model_classes == i)[0][0])
            out[:, i] = probs[:, col]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


RECEPTORS = [
    "5-HT1A", "5-HT1B", "5-HT1D", "5-HT1F", "5-HT2A", "5-HT2B", "5-HT2C",
    "5-HT4", "5-HT5A", "5-HT6", "5-HT7", "A1", "A2A", "A2B", "A3",
    "alpha1A", "alpha1B", "alpha2C", "beta1", "beta2", "beta3",
    "BLT1", "CB1", "CB2", "CysLT1", "CysLT2", "D1", "D2", "D3", "D4", "D5",
    "EP2", "EP3", "EP4", "FFA1", "FFA2", "FFA3", "FFA4", "FP",
    "GPBA", "GPR119", "GPR139", "GPR174", "GPR35", "H1", "H2", "H3", "H4", "HCA2", "IP",
    "LPA1", "M1", "M2", "M3", "M4", "M5", "MT1", "MT2", "P2Y1", "P2Y12", "PAF",
    "S1P1", "S1P2", "S1P3", "S1P5", "succinate", "TA1", "TP",
]


def load_table_new_only(su, base_path: Path) -> pd.DataFrame | None:
    """Load ligand table from *_NEW.xlsx only (manuscript workbooks)."""
    for suffix in ("_NEW.csv", "_NEW.xlsx"):
        path = base_path.with_name(base_path.stem + suffix)
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        return pd.read_excel(path)
    return None


def build_training_matrix(su, data_root: Path):
    """Same feature matrix as Code S9 (union of ligand cols across *_NEW workbooks)."""
    receptor_feats = {}
    for rec in RECEPTORS:
        feats = su.aggregate_receptor_features(rec)
        if feats:
            receptor_feats[rec] = feats
    receptor_df = pd.DataFrame(receptor_feats).T.fillna(0) if receptor_feats else pd.DataFrame()
    if len(receptor_df) > 0:
        receptor_df.index.name = "Receptor"
    print(f"  [OK] Receptor feature matrix: {len(receptor_df)} × {len(receptor_df.columns)}")

    rows = []
    all_ligand_cols: set[str] = set()
    receptors_to_load = list(receptor_df.index) if len(receptor_df) else RECEPTORS

    for rec in receptors_to_load:
        folder = data_root / rec
        if not folder.is_dir():
            continue
        for label, fname in [
            ("agonist", f"{rec}_agonist_enriched.csv"),
            ("antagonist", f"{rec}_antagonist_enriched.csv"),
            ("inactive", f"{rec}_non_active_compounds.csv"),
        ]:
            table = load_table_new_only(su, folder / fname)
            if table is None:
                print(f"  [SKIP] missing _NEW workbook for {rec} / {label}", flush=True)
                continue
            if os.environ.get("GPCR_EXPORT_VERBOSE", "").strip() == "1":
                print(f"  [OK] {rec} {label} from _NEW ({len(table):,} rows)", flush=True)
            table["Receptor"] = rec
            table["Label_Type"] = label
            all_ligand_cols.update(c for c in table.columns if not su.should_exclude_column(c))
            rows.append(table)

    if not rows:
        raise SystemExit(f"No ligand tables under {data_root}.")

    data_df = pd.concat(rows, ignore_index=True)
    if len(receptor_df) > 0:
        merged = data_df.merge(receptor_df.reset_index(), on="Receptor", how="left")
        merged[list(receptor_df.columns)] = merged[list(receptor_df.columns)].fillna(0)
    else:
        merged = data_df

    merged, interaction_cols = su.build_interaction_terms(merged, receptor_df)
    merged, smiles_series = su.canonicalize_smiles_series(merged)

    ligand_cols = sorted(all_ligand_cols)
    feature_cols = (
        ligand_cols + list(receptor_df.columns) + interaction_cols
        if len(receptor_df) > 0
        else ligand_cols + interaction_cols
    )
    feature_cols = [c for c in feature_cols if c in merged.columns]
    merged = merged.loc[:, ~merged.columns.duplicated()]
    feature_cols = [c for c in feature_cols if c in merged.columns]

    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
    X = X.select_dtypes(include=[np.number])
    final_cols = X.columns.tolist()
    X = X[final_cols]

    y, _, class_names = su.encode_labels(merged)
    print(f"  [OK] Training matrix: {len(X):,} rows × {len(final_cols)} features")
    return X, y, smiles_series, final_cols, class_names


def _load_train_split(su, X, y, smiles, data_root: Path) -> np.ndarray:
    split_dir = _split_dir(data_root)
    split_file = split_dir / "final_independent_ligand_80_20_canonical_smiles.json"
    if not split_file.exists():
        raise SystemExit(f"Missing split file: {split_file}")
    print(f"  [OK] Split: {split_file}")
    train_idx, test_idx = su.load_or_create_global_split(X, y, smiles, split_dir, random_state=42)
    train_idx = np.array(train_idx)
    print(f"  [OK] Fit on dev80: {len(train_idx):,} rows (test held out: {len(test_idx):,})")
    return train_idx


def export_independent_ligand(
    su,
    out_root: Path,
    model_kind: str,
    seeds: list[int],
    skip_existing: bool = False,
) -> list[str]:
    from sklearn.ensemble import RandomForestClassifier

    data_root = _training_data_root()
    X, y, smiles, final_cols, _ = build_training_matrix(su, data_root)
    train_idx = _load_train_split(su, X, y, smiles, data_root)
    X_train = X.iloc[train_idx]
    y_train = y[train_idx]

    dest = out_root / "independent_ligand" / model_kind
    dest.mkdir(parents=True, exist_ok=True)

    if model_kind == "rf":
        for seed in seeds:
            out_path = dest / f"model_seed{seed}.pkl"
            if skip_existing and out_path.exists() and out_path.stat().st_size > 50_000:
                print(f"  [SKIP] {out_path} exists")
                continue
            params = {**RF_PARAMS_S9, "random_state": seed}
            clf = RandomForestClassifier(**params)
            print(f"  [FIT] RF seed {seed} …", flush=True)
            clf.fit(X_train, y_train)
            joblib.dump(clf, out_path, compress=3)
            print(f"  [OK] {out_path} ({out_path.stat().st_size // 1024} KB)")
    elif model_kind == "xgboost":
        import xgboost as xgb

        for seed in seeds:
            out_path = dest / f"model_seed{seed}.pkl"
            if skip_existing and out_path.exists() and out_path.stat().st_size > 50_000:
                print(f"  [SKIP] {out_path}")
                continue
            params = {**XGB_PARAMS_S10, "random_state": seed}
            clf = xgb.XGBClassifier(**params)
            print(f"  [FIT] XGB seed {seed} …", flush=True)
            clf.fit(X_train, y_train)
            joblib.dump(clf, out_path, compress=3)
            print(f"  [OK] {out_path}")
    elif model_kind == "lightgbm":
        import lightgbm as lgb

        for seed in seeds:
            out_path = dest / f"model_seed{seed}.pkl"
            if skip_existing and out_path.exists() and out_path.stat().st_size > 50_000:
                print(f"  [SKIP] {out_path}")
                continue
            params = {**LGB_PARAMS_S11, "random_state": seed}
            clf = lgb.LGBMClassifier(**params)
            print(f"  [FIT] LightGBM seed {seed} …", flush=True)
            clf.fit(_lgb_fit_matrix(X_train), y_train)
            joblib.dump(clf, out_path, compress=3)
            print(f"  [OK] {out_path}")
    else:
        raise SystemExit(f"Unknown model_kind={model_kind}")

    return final_cols


ENSEMBLE_N_FOLDS = 5


def _ensemble_checkpoint_path(dest: Path, seed: int) -> Path:
    return dest / f"checkpoint_seed{seed}.joblib"


def _load_ensemble_checkpoint(
    path: Path, seed: int, n_train: int, n_meta: int, n_features: int
) -> dict | None:
    if not path.is_file():
        return None
    try:
        ckpt = joblib.load(path)
    except Exception as exc:
        print(f"  [WARN] Could not load checkpoint ({exc}); starting fresh")
        return None
    if (
        ckpt.get("seed") != seed
        or ckpt.get("n_train") != n_train
        or ckpt.get("n_meta") != n_meta
        or ckpt.get("n_features") != n_features
    ):
        print("  [WARN] Checkpoint metadata mismatch; starting fresh")
        return None
    meta_x = ckpt.get("meta_x_train")
    if meta_x is None or np.asarray(meta_x).shape != (n_train, n_meta):
        print("  [WARN] Checkpoint OOF matrix shape mismatch; starting fresh")
        return None
    completed = {int(f) for f in ckpt.get("completed_folds", [])}
    if completed:
        print(f"  [RESUME] Checkpoint: OOF folds already done {sorted(completed)}", flush=True)
    ckpt["meta_x_train"] = np.asarray(meta_x, dtype=np.float64)
    ckpt["completed_folds"] = completed
    return ckpt


def _save_ensemble_checkpoint(path: Path, payload: dict, note: str) -> None:
    out = dict(payload)
    if "completed_folds" in out and isinstance(out["completed_folds"], set):
        out["completed_folds"] = sorted(out["completed_folds"])
    joblib.dump(out, path, compress=3)
    print(f"  [CHKPT] {note} -> {path.name}", flush=True)


def export_ensemble_independent_ligand(
    su,
    out_root: Path,
    export_root: Path,
    seeds: list[int],
    skip_existing: bool = False,
    fresh: bool = False,
) -> list[str]:
    """
    Code S21-style stacking: 5-fold OOF base probabilities → meta LR;
    final RF/XGB/LGB refit on full dev80 for deployment (XGB/LGB use StandardScaler).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import xgboost as xgb

    StackingEnsemblePredictor = _import_stacking_class(export_root)

    data_root = _training_data_root()
    X, y, smiles, final_cols, _ = build_training_matrix(su, data_root)
    train_idx = _load_train_split(su, X, y, smiles, data_root)
    X_train = np.asarray(X.iloc[train_idx], dtype=np.float64)
    y_train = y[train_idx]
    n_train = len(y_train)
    n_meta = 3 * 3

    dest = out_root / "independent_ligand" / "ensemble"
    dest.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        out_path = dest / f"stacking_seed{seed}.pkl"
        if skip_existing and out_path.exists() and out_path.stat().st_size > 100_000:
            print(f"  [SKIP] {out_path}")
            continue

        checkpoint_path = _ensemble_checkpoint_path(dest, seed)
        if fresh and checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  [CHKPT] Removed {checkpoint_path.name} (--fresh)", flush=True)

        print(
            f"  [FIT] Ensemble seed {seed}: {ENSEMBLE_N_FOLDS}-fold OOF meta + full dev80 bases …",
            flush=True,
        )
        meta_x_train = np.zeros((n_train, n_meta), dtype=np.float64)
        completed_folds: set[int] = set()
        n_features = int(X_train.shape[1])
        ckpt_state: dict = {
            "seed": seed,
            "n_train": n_train,
            "n_meta": n_meta,
            "n_features": n_features,
            "meta_x_train": meta_x_train,
            "completed_folds": completed_folds,
            "meta_model": None,
            "tree_scaler": None,
            "final_rf": None,
            "final_xgb": None,
            "final_lgb": None,
        }

        if not fresh:
            loaded = _load_ensemble_checkpoint(
                checkpoint_path, seed, n_train, n_meta, n_features
            )
            if loaded is not None:
                ckpt_state.update(loaded)
                meta_x_train = ckpt_state["meta_x_train"]
                completed_folds = ckpt_state["completed_folds"]

        skf = StratifiedKFold(n_splits=ENSEMBLE_N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            if fold in completed_folds:
                print(f"    [SKIP] fold {fold}/{ENSEMBLE_N_FOLDS} (checkpoint)", flush=True)
                continue
            print(f"    fold {fold}/{ENSEMBLE_N_FOLDS} …", flush=True)
            X_tr = X_train[tr_idx]
            X_val = X_train[val_idx]
            y_tr = y_train[tr_idx]

            scaler_fold = StandardScaler()
            X_tr_s = scaler_fold.fit_transform(X_tr)
            X_val_s = scaler_fold.transform(X_val)

            rf_fold = RandomForestClassifier(**{**RF_PARAMS_S9, "random_state": seed})
            xgb_fold = xgb.XGBClassifier(**{**XGB_PARAMS_S10, "random_state": seed})
            lgb_fold = lgb.LGBMClassifier(**{**LGB_PARAMS_S11, "random_state": seed})

            rf_fold.fit(X_tr, y_tr)
            xgb_fold.fit(X_tr_s, y_tr)
            lgb_fold.fit(_lgb_fit_matrix(X_tr_s), y_tr)

            p_rf = _align_proba(rf_fold.predict_proba(X_val), rf_fold.classes_)
            p_xgb = _align_proba(xgb_fold.predict_proba(X_val_s), xgb_fold.classes_)
            p_lgb = _align_proba(
                lgb_fold.predict_proba(_lgb_fit_matrix(X_val_s)), lgb_fold.classes_
            )
            meta_x_train[val_idx] = np.hstack([p_rf, p_xgb, p_lgb])
            completed_folds.add(fold)
            ckpt_state["meta_x_train"] = meta_x_train
            ckpt_state["completed_folds"] = completed_folds
            _save_ensemble_checkpoint(
                checkpoint_path,
                ckpt_state,
                f"Saved OOF progress (folds {sorted(completed_folds)})",
            )

        meta = ckpt_state.get("meta_model")
        if meta is not None:
            print("  [SKIP] Meta-model (checkpoint)", flush=True)
        else:
            meta = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=3000,
                random_state=seed,
            )
            meta.fit(meta_x_train, y_train)
            print("  [OK] Meta-model fit on OOF features", flush=True)
            ckpt_state["meta_model"] = meta
            _save_ensemble_checkpoint(checkpoint_path, ckpt_state, "Saved meta-model")

        tree_scaler = ckpt_state.get("tree_scaler")
        if tree_scaler is not None:
            print("  [SKIP] Tree scaler (checkpoint)", flush=True)
            X_train_s = tree_scaler.transform(X_train)
        else:
            tree_scaler = StandardScaler()
            X_train_s = tree_scaler.fit_transform(X_train)
            ckpt_state["tree_scaler"] = tree_scaler
            _save_ensemble_checkpoint(checkpoint_path, ckpt_state, "Saved tree scaler")

        rf = ckpt_state.get("final_rf")
        if rf is not None:
            print("  [SKIP] Final RF (checkpoint)", flush=True)
        else:
            print("  [FIT] Final RF on full dev80 …", flush=True)
            rf = RandomForestClassifier(**{**RF_PARAMS_S9, "random_state": seed})
            rf.fit(X_train, y_train)
            print("  [OK] Final RF fit", flush=True)
            ckpt_state["final_rf"] = rf
            _save_ensemble_checkpoint(checkpoint_path, ckpt_state, "Saved final RF")

        xgb_clf = ckpt_state.get("final_xgb")
        if xgb_clf is not None:
            print("  [SKIP] Final XGB (checkpoint)", flush=True)
        else:
            print("  [FIT] Final XGB on full dev80 …", flush=True)
            xgb_clf = xgb.XGBClassifier(**{**XGB_PARAMS_S10, "random_state": seed})
            xgb_clf.fit(X_train_s, y_train)
            print("  [OK] Final XGB fit", flush=True)
            ckpt_state["final_xgb"] = xgb_clf
            _save_ensemble_checkpoint(checkpoint_path, ckpt_state, "Saved final XGB")

        lgb_clf = ckpt_state.get("final_lgb")
        if lgb_clf is not None:
            print("  [SKIP] Final LGB (checkpoint)", flush=True)
        else:
            print("  [FIT] Final LGB on full dev80 …", flush=True)
            lgb_clf = lgb.LGBMClassifier(**{**LGB_PARAMS_S11, "random_state": seed})
            lgb_clf.fit(_lgb_fit_matrix(X_train_s), y_train)
            print("  [OK] Final LGB fit", flush=True)
            ckpt_state["final_lgb"] = lgb_clf
            _save_ensemble_checkpoint(checkpoint_path, ckpt_state, "Saved final LGB")

        stack = StackingEnsemblePredictor(
            rf, xgb_clf, lgb_clf, meta, tree_scaler=tree_scaler
        )
        joblib.dump(stack, out_path, compress=3)
        print(f"  [OK] {out_path} ({out_path.stat().st_size // 1024} KB)")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  [CHKPT] Removed {checkpoint_path.name} (export complete)", flush=True)

    return final_cols


def write_manifest(out_root: Path, feature_columns: list, extra: dict):
    manifest = {
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "class_names": ["Agonist", "Antagonist", "Inactive"],
        **extra,
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_root / 'manifest.json'} ({len(feature_columns)} features)")


def main():
    parser = argparse.ArgumentParser(description="Export manuscript models for Streamlit")
    parser.add_argument("--regime", required=True, choices=["independent_ligand", "scaffold", "loro", "ensemble"])
    parser.add_argument("--model", default="rf", choices=["rf", "lightgbm", "xgboost", "ensemble"])
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds (default: 42 only)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip seeds whose .pkl already exists")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ensemble only: delete OOF checkpoint and re-run all folds from scratch",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    export_root = _export_root()
    out_root = export_root / "artifacts" / "manuscript"
    out_root.mkdir(parents=True, exist_ok=True)

    su = _import_training_stack()

    if args.regime == "independent_ligand" and args.model != "ensemble":
        cols = export_independent_ligand(su, out_root, args.model, seeds, skip_existing=args.skip_existing)
        write_manifest(
            out_root,
            cols,
            {
                "evaluation_regime": "independent_ligand",
                "source_scripts": ["Code S9", "Code S10", "Code S11"],
                "training_rows_note": "fit on dev80; feature columns = union across all receptor enriched tables",
                "seeds": seeds,
            },
        )
    elif args.regime == "ensemble" or (args.regime == "independent_ligand" and args.model == "ensemble"):
        cols = export_ensemble_independent_ligand(
            su, out_root, export_root, seeds, skip_existing=args.skip_existing, fresh=args.fresh
        )
        manifest_path = out_root / "manifest.json"
        extra = {
            "evaluation_regime": "independent_ligand",
            "source_scripts": ["Code S21"],
            "training_rows_note": "Code S21: 5-fold OOF meta on 9 probs; RF/XGB/LGB refit on full dev80 (XGB/LGB scaled)",
            "seeds": seeds,
        }
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                old = json.load(f)
            if len(old.get("feature_columns", [])) == len(cols):
                print("  [OK] manifest.json feature count matches ensemble export")
            else:
                write_manifest(out_root, cols, extra)
        else:
            write_manifest(out_root, cols, extra)
    else:
        print(f"Regime '{args.regime}' not implemented yet (scaffold / LORO).")
        sys.exit(1)


if __name__ == "__main__":
    main()
