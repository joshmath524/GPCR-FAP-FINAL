"""
Build manuscript feature rows (full ligand + receptor + interaction columns).

Requires artifacts/manuscript/manifest.json with feature_columns produced by export script.
Ligand columns come from *_NEW.xlsx (training); use ligand_feature_lookup.joblib when built,
otherwise fall back to on-the-fly RDKit + Mordred (fewer columns — biased toward Inactive).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from rdkit import Chem

from .ligand_enrichment import build_ligand_descriptor_dict
from .predict import _canonicalize_smiles
from .receptor_names import resolve_receptor_folder

_LIGAND_LOOKUP_CACHE: Optional[Tuple[Dict[str, Dict[str, float]], str]] = None


def _manuscript_root(project_root: Path) -> Path:
    return project_root / "artifacts" / "manuscript"


def load_feature_columns(project_root: Path) -> List[str]:
    manifest_path = _manuscript_root(project_root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run scripts/export_manuscript_models.py from your training repo."
        )
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    cols = data.get("feature_columns")
    if not cols:
        raise ValueError("manifest.json must contain feature_columns list.")
    return list(cols)


def _try_import_shared_utilities():
    ml_root = os.environ.get("MANUSCRIPT_ML_ROOT", "").strip()
    if not ml_root:
        return None
    import sys

    p = Path(ml_root)
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
    try:
        import shared_utilities as su  # type: ignore

        return su
    except ImportError:
        return None


def _load_ligand_lookup(project_root: Path) -> Tuple[Dict[str, Dict[str, float]], str]:
    """Load SMILES → ligand descriptor dict built from *_NEW.xlsx only."""
    global _LIGAND_LOOKUP_CACHE
    if _LIGAND_LOOKUP_CACHE is not None:
        return _LIGAND_LOOKUP_CACHE

    path = _manuscript_root(project_root) / "ligand_feature_lookup.joblib"
    if not path.exists():
        _LIGAND_LOOKUP_CACHE = ({}, "missing")
        return _LIGAND_LOOKUP_CACHE

    try:
        payload = joblib.load(path)
        lookup = payload.get("lookup", {})
        _LIGAND_LOOKUP_CACHE = (lookup, str(payload.get("source", "_NEW.xlsx")))
    except Exception:
        _LIGAND_LOOKUP_CACHE = ({}, "corrupt_or_incomplete")
    return _LIGAND_LOOKUP_CACHE


def _ligand_row_for_smiles(
    project_root: Path,
    receptor_input: str,
    canon: str,
    mol: Chem.Mol,
) -> Dict[str, float]:
    from .new_workbook_ligand import ligand_dict_from_new_workbooks

    row = ligand_dict_from_new_workbooks(receptor_input, canon)
    if not row:
        lookup, _ = _load_ligand_lookup(project_root)
        if canon in lookup and lookup[canon]:
            row = dict(lookup[canon])
    computed = build_ligand_descriptor_dict(canon, mol=mol) or {}
    for col, val in computed.items():
        if col not in row:
            row[col] = val
    return row

def _receptor_features(receptor_input: str, data_root: Path) -> Optional[Dict[str, float]]:
    folder = resolve_receptor_folder(receptor_input, data_root)
    if folder is None:
        return None
    su = _try_import_shared_utilities()
    if su is not None:
        rec = su.aggregate_receptor_features(folder)
        return rec if rec else None
    from .predict import _aggregate_receptor_feature_dict

    return _aggregate_receptor_feature_dict(receptor_input)


def _fill_row_from_parts(
    feature_columns: List[str],
    lig_row: Dict[str, float],
    rec_feats: Dict[str, float],
) -> np.ndarray:
    row_dict: Dict[str, float] = {}
    for col in feature_columns:
        if col in lig_row:
            row_dict[col] = lig_row[col]
        elif col in rec_feats:
            row_dict[col] = float(rec_feats[col])
        elif col.startswith("INT_") and "_X_" in col:
            lig_k, rec_k = col[4:].split("_X_", 1)
            row_dict[col] = float(lig_row.get(lig_k, 0.0)) * float(rec_feats.get(rec_k, 0.0))
        else:
            row_dict[col] = 0.0
    return np.array([float(row_dict.get(c, 0.0)) for c in feature_columns], dtype=np.float32)


def build_manuscript_feature_row(
    project_root: Path,
    receptor_input: str,
    ligand_smiles: str,
    feature_columns: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Build one row aligned to manuscript training columns (from manifest).
    """
    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return None

    if feature_columns is None:
        feature_columns = load_feature_columns(project_root)

    data_root = Path(os.environ.get("GPCR_DATA_ROOT", "").strip() or project_root)
    rec_feats = _receptor_features(receptor_input, data_root)
    if not rec_feats:
        return None

    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return None
    lig_row = _ligand_row_for_smiles(project_root, receptor_input, canon, mol)
    if not lig_row:
        return None

    return _fill_row_from_parts(feature_columns, lig_row, rec_feats)


def build_demo_2103_features(receptor_input: str, ligand_smiles: str) -> Optional[np.ndarray]:
    """Legacy 2103-dim demo bundle (10 RDKit + Morgan + 31 receptor + 14 interaction)."""
    from .predict import _compute_full_features

    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return None
    return _compute_full_features(receptor_input, canon)
