#!/usr/bin/env python3
"""Check RF / XGBoost / LightGBM predictions vs _NEW workbook labels."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_data_root = os.environ.get("MANUSCRIPT_DATA_ROOT", "").strip()
if not _data_root:
    raise SystemExit(
        "Set MANUSCRIPT_DATA_ROOT before running validation "
        "(folder containing receptor *_NEW.xlsx workbooks)."
    )
DATA = Path(_data_root)
if not DATA.is_dir():
    raise SystemExit(f"MANUSCRIPT_DATA_ROOT is not a directory: {DATA}")

os.environ.setdefault("GPCR_DATA_ROOT", str(DATA))
os.environ.setdefault("MANUSCRIPT_ML_ROOT", str(DATA / "ML_code"))
os.environ.setdefault("MANUSCRIPT_DATA_ROOT", str(DATA))

from src.gpcr.manuscript_features import build_manuscript_feature_row, load_feature_columns
from src.gpcr.predict import _canonicalize_smiles, load_predictor, predict_single

LABEL_TO_ID = {"agonist": 0, "antagonist": 1, "inactive": 2}
ID_TO_NAME = ["Agonist", "Antagonist", "Inactive"]

RECEPTORS = [
    ("beta2", 15),
    ("D2", 10),
    ("5-HT2A", 10),
]


def _sample_smiles(receptor: str, label: str, n: int) -> list[tuple[str, str]]:
    folder = DATA / receptor
    fname = {
        "agonist": f"{receptor}_agonist_enriched_NEW.xlsx",
        "antagonist": f"{receptor}_antagonist_enriched_NEW.xlsx",
        "inactive": f"{receptor}_non_active_compounds_NEW.xlsx",
    }[label]
    path = folder / fname
    if not path.exists():
        return []
    df = pd.read_excel(path)
    if "SMILES" not in df.columns:
        return []
    out: list[tuple[str, str]] = []
    for smi in df["SMILES"].dropna().astype(str):
        canon = _canonicalize_smiles(smi)
        if not canon:
            continue
        out.append((canon, label))
        if len(out) >= n:
            break
    return out


def _eval_receptor(receptor: str, n_per_class: int, cols: list[str]) -> dict:
    cases: list[tuple[str, str]] = []
    for label in ("agonist", "antagonist", "inactive"):
        cases.extend(_sample_smiles(receptor, label, n_per_class))

    rf_pred = load_predictor(
        str(ROOT), model_type="rf", evaluation_regime="independent_ligand", seed=42
    )
    xgb_pred = load_predictor(
        str(ROOT), model_type="xgboost", evaluation_regime="independent_ligand", seed=42
    )
    lgb_pred = load_predictor(
        str(ROOT), model_type="lightgbm", evaluation_regime="independent_ligand", seed=42
    )

    stats = {
        "receptor": receptor,
        "n": len(cases),
        "rf_correct": 0,
        "xgb_correct": 0,
        "lgb_correct": 0,
        "sparse_infer": 0,
        "by_label": {},
    }

    for canon, true_label in cases:
        vec = build_manuscript_feature_row(ROOT, receptor, canon, feature_columns=cols)
        nz = int(np.count_nonzero(vec)) if vec is not None else 0
        if vec is None or nz < 1000:
            stats["sparse_infer"] += 1

        rr = predict_single(receptor, canon, predictor=rf_pred)
        xr = predict_single(receptor, canon, predictor=xgb_pred)
        lr = predict_single(receptor, canon, predictor=lgb_pred)
        true_id = LABEL_TO_ID[true_label]
        rf_ok = rr.class_id == true_id
        xgb_ok = xr.class_id == true_id
        lgb_ok = lr.class_id == true_id
        stats["rf_correct"] += int(rf_ok)
        stats["xgb_correct"] += int(xgb_ok)
        stats["lgb_correct"] += int(lgb_ok)

        bl = stats["by_label"].setdefault(
            true_label,
            {"n": 0, "rf_ok": 0, "xgb_ok": 0, "lgb_ok": 0, "lgb_inactive": 0},
        )
        bl["n"] += 1
        bl["rf_ok"] += int(rf_ok)
        bl["xgb_ok"] += int(xgb_ok)
        bl["lgb_ok"] += int(lgb_ok)
        if lr.class_id == 2:
            bl["lgb_inactive"] += 1

    return stats


def main() -> None:
    cols = load_feature_columns(ROOT)
    print(f"manifest: {len(cols)} features\n")

    for receptor, n_per in RECEPTORS:
        s = _eval_receptor(receptor, n_per, cols)
        n = s["n"]
        if n == 0:
            print(f"{receptor}: no _NEW samples found")
            continue
        print(f"=== {receptor} ({n} ligands from _NEW) ===")
        print(f"  RF  accuracy:  {s['rf_correct']}/{n} ({100 * s['rf_correct'] / n:.1f}%)")
        print(f"  XGB accuracy:  {s['xgb_correct']}/{n} ({100 * s['xgb_correct'] / n:.1f}%)")
        print(f"  LGB accuracy:  {s['lgb_correct']}/{n} ({100 * s['lgb_correct'] / n:.1f}%)")
        if s["sparse_infer"]:
            print(f"  WARN: {s['sparse_infer']} rows with <1000 nonzero features")
        for label, bl in s["by_label"].items():
            print(
                f"    {label:10s} n={bl['n']:2d}  "
                f"RF {bl['rf_ok']}/{bl['n']}  XGB {bl['xgb_ok']}/{bl['n']}  "
                f"LGB {bl['lgb_ok']}/{bl['n']}  "
                f"(LGB->Inactive {bl['lgb_inactive']}/{bl['n']})"
            )
        print()

    # Spot-check: first beta2 agonist — compare probs
    cases = _sample_smiles("beta2", "agonist", 3)
    if cases:
        canon, _ = cases[0]
        lgb = load_predictor(
            str(ROOT), model_type="lightgbm", evaluation_regime="independent_ligand", seed=42
        )
        r = predict_single("beta2", canon, predictor=lgb)
        vec = build_manuscript_feature_row(ROOT, "beta2", canon, feature_columns=cols)
        nz = int(np.count_nonzero(vec)) if vec is not None else 0
        print("Spot check (first beta2 agonist from _NEW):")
        print(f"  SMILES: {canon[:70]}...")
        print(f"  nonzero features: {nz}")
        print(
            f"  LGB: {r.predicted_class}  "
            f"P=[{r.prob_agonist:.3f}, {r.prob_antagonist:.3f}, {r.prob_inactive:.3f}]"
        )


if __name__ == "__main__":
    main()
