"""
Minimal manuscript RF inference for Streamlit Cloud (~1 GB RAM).

Avoids GPCRPredictor / load_predictor overhead; loads only model_seed*_cloud.pkl.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .manuscript_bundle import _align_proba
from .manuscript_features import build_manuscript_feature_row, load_feature_columns
from .predict import PredictResult, _canonicalize_smiles


def _cloud_rf_path(project_root: Path, regime: str, seed: int) -> Path:
    return (
        project_root
        / "artifacts"
        / "manuscript"
        / regime
        / "rf"
        / f"model_seed{seed}_cloud.pkl"
    )


def predict_cloud_rf(
    project_root: Path,
    receptor: str,
    ligand_smiles: str,
    evaluation_regime: str = "independent_ligand",
    seed: int = 42,
) -> PredictResult:
    root = Path(project_root)
    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return PredictResult(
            is_valid=False,
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canonical_smiles="",
            predicted_class="Unknown",
            class_id=-1,
            prob_agonist=0.0,
            prob_antagonist=0.0,
            prob_inactive=0.0,
            error="Invalid SMILES",
        )

    model_path = _cloud_rf_path(root, evaluation_regime, seed)
    if not model_path.is_file() or model_path.stat().st_size < 50_000:
        return PredictResult(
            is_valid=False,
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canonical_smiles=canon,
            predicted_class="Unknown",
            class_id=-1,
            prob_agonist=0.0,
            prob_antagonist=0.0,
            prob_inactive=0.0,
            error=(
                f"Missing {model_path.name}. Run scripts/shrink_rf_for_cloud.py and deploy via Git LFS."
            ),
        )

    cols = load_feature_columns(root)
    vec = build_manuscript_feature_row(root, receptor, canon, feature_columns=cols)
    if vec is None:
        return PredictResult(
            is_valid=False,
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canonical_smiles=canon,
            predicted_class="Unknown",
            class_id=-1,
            prob_agonist=0.0,
            prob_antagonist=0.0,
            prob_inactive=0.0,
            error="Could not build feature row (receptor pockets or ligand lookup).",
        )

    model = None
    try:
        model = joblib.load(model_path)
        X = np.ascontiguousarray(vec.reshape(1, -1), dtype=np.float64)
        raw = model.predict_proba(X)
        probs = _align_proba(raw, model.classes_, n_classes=3)[0]
        class_id = int(np.argmax(probs))
        names = ["Agonist", "Antagonist", "Inactive"]
        return PredictResult(
            is_valid=True,
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canonical_smiles=canon,
            predicted_class=names[class_id],
            class_id=class_id,
            prob_agonist=float(probs[0]),
            prob_antagonist=float(probs[1]),
            prob_inactive=float(probs[2]),
            prob_std_error=0.0,
        )
    finally:
        if model is not None:
            if hasattr(model, "estimators_"):
                model.estimators_ = []
            del model
        del vec
        gc.collect()
