#!/usr/bin/env python3
"""Verify manuscript feature rows match training-scale descriptor coverage."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpcr.manuscript_features import build_manuscript_feature_row, load_feature_columns
from gpcr.predict import INTERACTION_PAIRS

DATA_ROOT = Path(
    os.environ.get(
        "MANUSCRIPT_DATA_ROOT",
        r"C:\Users\joshmatchem\Music\GPCR-FAP-main (2)\GPCRtryagain - Delete - Copy",
    )
)
ML_ROOT = Path(
    os.environ.get(
        "MANUSCRIPT_ML_ROOT",
        DATA_ROOT / "ML_code",
    )
)
os.environ.setdefault("GPCR_DATA_ROOT", str(DATA_ROOT))
os.environ.setdefault("MANUSCRIPT_ML_ROOT", str(ML_ROOT))


def _nonzero(vec: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(vec) > 1e-12))


def main() -> None:
    cols = load_feature_columns(ROOT)
    csv_path = DATA_ROOT / "beta2" / "beta2_agonist_enriched.csv"
    row = pd.read_csv(csv_path, nrows=1).iloc[0]
    smiles = str(row["SMILES"])
    receptor = "beta2"

    train_vec = (
        row.reindex(cols)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .astype(np.float32)
        .to_numpy()
    )
    infer_vec = build_manuscript_feature_row(ROOT, receptor, smiles, feature_columns=cols)
    if infer_vec is None:
        print("FAIL: build_manuscript_feature_row returned None")
        sys.exit(1)

    nz_train = _nonzero(train_vec)
    nz_infer = _nonzero(infer_vec)
    diff = np.abs(train_vec - infer_vec)
    nz_diff = int(np.count_nonzero(diff > 1e-3))
    max_diff = float(diff.max()) if diff.size else 0.0

    print(f"SMILES: {smiles[:60]}...")
    print(f"manifest features: {len(cols)}")
    print(f"training nonzero: {nz_train}")
    print(f"inference nonzero: {nz_infer}")
    print(f"columns |diff|>1e-3: {nz_diff}, max diff: {max_diff:.4g}")

    model_path = ROOT / "artifacts" / "manuscript" / "independent_ligand" / "rf" / "model_seed42.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        pred_train = model.predict([train_vec])[0]
        pred_infer = model.predict([infer_vec])[0]
        proba_infer = model.predict_proba([infer_vec])[0]
        names = ["Agonist", "Antagonist", "Inactive"]
        print(f"RF on training row: {names[pred_train]}")
        print(f"RF on inference row: {names[pred_infer]} probs={proba_infer.round(3)}")
    else:
        print(f"(skip RF) missing {model_path}")

    if nz_infer < 1000:
        print("WARN: inference still sparse (<1000 nonzero)")
        sys.exit(2)
    if nz_diff > 200:
        print("WARN: large mismatch vs training row")
        sys.exit(3)
    print("OK")


if __name__ == "__main__":
    main()
