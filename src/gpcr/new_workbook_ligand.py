"""
Load ligand descriptor rows from *_NEW.xlsx only (manuscript training workbooks).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .predict import _canonicalize_smiles

_EXCLUDE = {
    "Receptor", "Label", "ChEMBL_ID", "AID", "CID", "Activity", "SMILES",
    "Label_Type", "Label_Type_clean",
}

_RECEPTOR_TABLE_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}


def _data_root() -> Path:
    raw = os.environ.get("GPCR_DATA_ROOT", "").strip() or os.environ.get(
        "MANUSCRIPT_DATA_ROOT", ""
    ).strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2]


def _numeric_row_dict(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in row.index:
        if col in _EXCLUDE:
            continue
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(val):
            continue
        out[str(col)] = float(val)
    return out


def _load_receptor_smiles_index(receptor: str) -> Dict[str, Dict[str, float]]:
    if receptor in _RECEPTOR_TABLE_CACHE:
        return _RECEPTOR_TABLE_CACHE[receptor]

    folder = _data_root() / receptor
    index: Dict[str, Dict[str, float]] = {}
    if not folder.is_dir():
        _RECEPTOR_TABLE_CACHE[receptor] = index
        return index

    for name in [
        f"{receptor}_agonist_enriched_NEW.xlsx",
        f"{receptor}_antagonist_enriched_NEW.xlsx",
        f"{receptor}_non_active_compounds_NEW.xlsx",
    ]:
        path = folder / name
        if not path.exists():
            continue
        df = pd.read_excel(path)
        if "SMILES" not in df.columns:
            continue
        for _, row in df.iterrows():
            smi = row.get("SMILES")
            if pd.isna(smi) or not str(smi).strip():
                continue
            canon = _canonicalize_smiles(str(smi))
            if not canon:
                continue
            prev = index.get(canon, {})
            prev.update(_numeric_row_dict(row))
            index[canon] = prev

    _RECEPTOR_TABLE_CACHE[receptor] = index
    return index


def ligand_dict_from_new_workbooks(receptor: str, canonical_smiles: str) -> Dict[str, float]:
    """Return ligand columns from _NEW files for this receptor + SMILES, or {}."""
    folder = resolve_receptor_folder_name(receptor)
    if folder is None:
        return {}
    index = _load_receptor_smiles_index(folder)
    return dict(index.get(canonical_smiles, {}))


def resolve_receptor_folder_name(receptor_input: str) -> Optional[str]:
    from .receptor_names import resolve_receptor_folder

    return resolve_receptor_folder(receptor_input, _data_root())
