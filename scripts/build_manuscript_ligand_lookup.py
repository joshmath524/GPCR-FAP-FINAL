#!/usr/bin/env python3
"""
Build canonical-SMILES → ligand descriptor lookup from *_NEW.xlsx only.

Training/manifest features come from these workbooks (not the slimmer *_enriched.csv).
Streamlit inference uses this lookup so rows match what the RF was trained on.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "MANUSCRIPT_DATA_ROOT",
        r"C:\Users\joshmatchem\Music\GPCR-FAP-main (2)\GPCRtryagain - Delete - Copy",
    )
)

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

EXCLUDE = {
    "Receptor", "Label", "ChEMBL_ID", "AID", "CID", "Activity", "SMILES",
    "Label_Type", "Label_Type_clean",
}


def _canon_smiles(smi: str) -> str | None:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_new_xlsx(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_excel(path)


def main() -> None:
    manifest_path = ROOT / "artifacts" / "manuscript" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest_cols = json.load(f).get("feature_columns", [])
        ligand_cols = [
            c
            for c in manifest_cols
            if not c.startswith("INT_")
            and c
            not in {
                "num_residues", "num_aromatic", "num_acidic", "num_basic",
                "num_charge_positive", "num_charge_negative", "num_charge_neutral",
                "num_polar", "num_nonpolar", "num_size_small", "num_size_medium",
                "num_size_large", "num_sulfur", "num_hydroxyl", "num_amide",
                "aromatic_ratio", "basic_ratio", "acidic_ratio",
                "charge_positive_ratio", "charge_negative_ratio", "charge_neutral_ratio",
                "polar_ratio", "nonpolar_ratio", "size_small_ratio", "size_medium_ratio",
                "size_large_ratio", "sulfur_ratio", "hydroxyl_ratio", "amide_ratio",
                "avg_distance", "avg_conservation",
            }
        ]
    else:
        ligand_cols = None

    lookup: dict[str, dict[str, float]] = {}
    files_loaded = 0
    rows_indexed = 0

    for rec in RECEPTORS:
        folder = DATA_ROOT / rec
        if not folder.is_dir():
            continue
        for stem in [
            f"{rec}_agonist_enriched_NEW.xlsx",
            f"{rec}_antagonist_enriched_NEW.xlsx",
            f"{rec}_non_active_compounds_NEW.xlsx",
        ]:
            path = folder / stem
            df = load_new_xlsx(path)
            if df is None or "SMILES" not in df.columns:
                continue
            files_loaded += 1
            num_cols = [c for c in df.columns if c not in EXCLUDE]
            for _, row in df.iterrows():
                smi = row.get("SMILES")
                if pd.isna(smi) or not str(smi).strip():
                    continue
                canon = _canon_smiles(str(smi))
                if not canon or canon in ("", "nan"):
                    continue
                vec: dict[str, float] = lookup.get(canon, {})
                for col in num_cols:
                    val = pd.to_numeric(row.get(col), errors="coerce")
                    if pd.isna(val) or np.isinf(val):
                        continue
                    vec[col] = float(val)
                lookup[canon] = vec
                rows_indexed += 1

    out_dir = ROOT / "artifacts" / "manuscript"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ligand_feature_lookup.joblib"
    payload = {
        "lookup": lookup,
        "ligand_columns": ligand_cols,
        "source": "_NEW.xlsx only",
        "n_smiles": len(lookup),
        "files_loaded": files_loaded,
    }
    joblib.dump(payload, out_path, compress=3)
    print(f"Loaded {files_loaded} _NEW workbooks, indexed {rows_indexed} rows")
    print(f"Unique canonical SMILES: {len(lookup)}")
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
    if ligand_cols:
        sample = next(iter(lookup.values()), {})
        hit = sum(1 for c in ligand_cols if c in sample)
        print(f"Manifest ligand columns matched in sample row: {hit}/{len(ligand_cols)}")


if __name__ == "__main__":
    main()
