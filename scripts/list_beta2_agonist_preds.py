#!/usr/bin/env python3
"""List beta2 agonist-workbook compounds predicted as Agonist."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gpcr.predict import _canonicalize_smiles, load_predictor, predict_single

DATA = Path(os.environ["GPCR_DATA_ROOT"])
MODEL = os.environ.get("MODEL", "ensemble")


def main() -> None:
    p = DATA / "beta2" / "beta2_agonist_enriched_NEW.xlsx"
    df = pd.read_excel(p)
    cols = [c for c in ("AID", "CID", "ChEMBL_ID", "SMILES") if c in df.columns]
    seen: set[str] = set()
    rows: list[dict] = []
    for _, r in df.iterrows():
        smi = r.get("SMILES")
        if pd.isna(smi):
            continue
        c = _canonicalize_smiles(str(smi))
        if not c or c in seen:
            continue
        seen.add(c)
        row = {k: r.get(k) for k in cols}
        row["SMILES"] = c
        rows.append(row)

    pred = load_predictor(
        str(ROOT), model_type=MODEL, evaluation_regime="independent_ligand", seed=42
    )
    for row in rows:
        res = predict_single("beta2", row["SMILES"], predictor=pred)
        row["pred"] = res.predicted_class
        row["p_ag"] = res.prob_agonist
        row["p_ant"] = res.prob_antagonist
        row["p_in"] = res.prob_inactive

    agonist = [r for r in rows if r["pred"] == "Agonist"]
    print(f"{MODEL.upper()} agonist: {len(agonist)} / {len(rows)}")
    for r in agonist:
        aid = r.get("AID", "")
        cid = r.get("CID", "")
        chembl = r.get("ChEMBL_ID", "")
        print(
            f"AID={aid}\tCID={cid}\tChEMBL={chembl}\t"
            f"P=[{r['p_ag']:.3f},{r['p_ant']:.3f},{r['p_in']:.3f}]\t"
            f"SMILES={r['SMILES']}"
        )


if __name__ == "__main__":
    main()
