"""
Minimal manuscript inference for Streamlit Cloud (~1 GB RAM).

Loads one model at a time (RF cloud / XGB / LGB), predicts, then frees memory.
Does not use GPCRPredictor. Ensemble is not supported on Cloud.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .manuscript_bundle import (
    _align_proba,
    _joblib_load_model,
    manuscript_model_pick_path,
)
from .manuscript_features import build_manuscript_feature_row, load_feature_columns
from .predict import PredictResult, _canonicalize_smiles
from .uncertainty import aggregate_base_model_probs, uncertainty_on_predicted_class

CloudModelType = Literal["rf", "lightgbm", "xgboost"]
CLOUD_MODEL_TYPES = ("rf", "lightgbm", "xgboost")


def cloud_model_path(
    project_root: Path,
    evaluation_regime: str,
    model_type: str,
    seed: int = 42,
) -> Path:
    mt = model_type.strip().lower()
    folder = {"rf": "rf", "lightgbm": "lightgbm", "xgboost": "xgboost"}.get(mt)
    if folder is None:
        raise ValueError(f"Unsupported cloud model_type: {model_type}")
    pick = manuscript_model_pick_path(project_root, evaluation_regime, mt, seed)
    if pick is not None:
        return pick
    name = f"model_seed{seed}_cloud.pkl" if mt == "rf" else f"model_seed{seed}.pkl"
    return project_root / "artifacts" / "manuscript" / evaluation_regime / folder / name


def cloud_model_ready(
    project_root: Path,
    evaluation_regime: str,
    model_type: str,
    seed: int = 42,
    *,
    min_bytes: int = 50_000,
) -> bool:
    if model_type.strip().lower() not in CLOUD_MODEL_TYPES:
        return False
    path = manuscript_model_pick_path(project_root, evaluation_regime, model_type, seed)
    if path is None:
        path = cloud_model_path(project_root, evaluation_regime, model_type, seed)
    return path.is_file() and path.stat().st_size >= min_bytes


def _invalid(
    *,
    receptor: str,
    ligand_smiles: str,
    canon: str,
    error: str,
) -> PredictResult:
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
        error=error,
    )


def _release_model(model: object) -> None:
    if hasattr(model, "estimators_"):
        model.estimators_ = []  # type: ignore[attr-defined]
    del model


def predict_cloud_manuscript(
    project_root: Path,
    receptor: str,
    ligand_smiles: str,
    evaluation_regime: str = "independent_ligand",
    seed: int = 42,
    model_type: str = "rf",
) -> PredictResult:
    root = Path(project_root)
    mt = model_type.strip().lower()
    if mt not in CLOUD_MODEL_TYPES:
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon="",
            error=f"Cloud supports rf, lightgbm, xgboost only (not {model_type}).",
        )

    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon="",
            error="Invalid SMILES",
        )

    model_path = manuscript_model_pick_path(root, evaluation_regime, mt, seed)
    if model_path is None or not cloud_model_ready(root, evaluation_regime, mt, seed):
        model_path = cloud_model_path(root, evaluation_regime, mt, seed)
    if not cloud_model_ready(root, evaluation_regime, mt, seed):
        if mt == "rf":
            hint = "Run scripts/shrink_rf_for_cloud.py and deploy model_seed*_cloud.pkl via Git LFS."
        else:
            hint = f"Deploy {model_path.name} under artifacts/manuscript/{evaluation_regime}/{mt}/."
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon=canon,
            error=f"Missing {model_path.name}. {hint}",
        )

    cols = load_feature_columns(root)
    vec = build_manuscript_feature_row(root, receptor, canon, feature_columns=cols)
    if vec is None:
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon=canon,
            error="Could not build feature row (receptor pockets or ligand lookup).",
        )

    model: Optional[object] = None
    try:
        model = _joblib_load_model(model_path)
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
            prob_std_error=None,  # single model — no cross-model ensemble variance
        )
    finally:
        if model is not None:
            _release_model(model)
        del vec
        gc.collect()


def _proba_from_loaded_model(model: object, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1, 2]))
    return _align_proba(raw, classes, n_classes=3)[0]


def predict_cloud_manuscript_with_uncertainty(
    project_root: Path,
    receptor: str,
    ligand_smiles: str,
    evaluation_regime: str = "independent_ligand",
    seed: int = 42,
) -> PredictResult:
    """
    Run RF, XGBoost, and LightGBM sequentially (one in RAM at a time).

    Class probabilities are the mean across available base models; standard error
    reflects disagreement among those models (same idea as local Ensemble diagnostics).
    """
    root = Path(project_root)
    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon="",
            error="Invalid SMILES",
        )

    cols = load_feature_columns(root)
    vec = build_manuscript_feature_row(root, receptor, canon, feature_columns=cols)
    if vec is None:
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon=canon,
            error="Could not build feature row (receptor pockets or ligand lookup).",
        )

    X = np.ascontiguousarray(vec.reshape(1, -1), dtype=np.float64)
    all_probs: list[np.ndarray] = []
    try:
        for mt in CLOUD_MODEL_TYPES:
            if not cloud_model_ready(root, evaluation_regime, mt, seed):
                continue
            path = manuscript_model_pick_path(root, evaluation_regime, mt, seed)
            if path is None:
                continue
            model: Optional[object] = None
            try:
                model = _joblib_load_model(path)
                all_probs.append(_proba_from_loaded_model(model, X))
            finally:
                if model is not None:
                    _release_model(model)
                gc.collect()

        if not all_probs:
            return _invalid(
                receptor=receptor,
                ligand_smiles=ligand_smiles,
                canon=canon,
                error="No cloud models (RF / XGB / LGB) found for uncertainty run.",
            )

        if len(all_probs) == 1:
            probs = all_probs[0]
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
                prob_std_error=None,
            )

        agg = aggregate_base_model_probs(all_probs)
        if agg[0] is None:
            return _invalid(
                receptor=receptor,
                ligand_smiles=ligand_smiles,
                canon=canon,
                error="Could not aggregate base-model probabilities.",
            )
        mean_probs, std_probs, std_error = agg
        class_id, se, sd = uncertainty_on_predicted_class(mean_probs, std_probs, std_error)
        names = ["Agonist", "Antagonist", "Inactive"]
        return PredictResult(
            is_valid=True,
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canonical_smiles=canon,
            predicted_class=names[class_id],
            class_id=class_id,
            prob_agonist=float(mean_probs[0]),
            prob_antagonist=float(mean_probs[1]),
            prob_inactive=float(mean_probs[2]),
            prob_std_error=se,
            prob_std_dev=sd,
        )
    finally:
        del vec, X, all_probs
        gc.collect()


def predict_cloud_rf(
    project_root: Path,
    receptor: str,
    ligand_smiles: str,
    evaluation_regime: str = "independent_ligand",
    seed: int = 42,
) -> PredictResult:
    """Backward-compatible wrapper."""
    return predict_cloud_manuscript(
        project_root,
        receptor,
        ligand_smiles,
        evaluation_regime=evaluation_regime,
        seed=seed,
        model_type="rf",
    )
