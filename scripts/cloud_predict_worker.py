#!/usr/bin/env python3
"""Run one manuscript RF prediction in a subprocess (frees RAM when process exits)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--regime", default="independent_ligand")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("GPCR_CLOUD_LITE", "1")
    os.environ.setdefault("GPCR_JOBLIB_MMAP", "1")

    from gpcr.predict import load_predictor, predict_single

    predictor = load_predictor(
        root,
        model_type="rf",
        evaluation_regime=args.regime,
        seed=args.seed,
    )
    result = predict_single(args.receptor, args.smiles, predictor=predictor)
    payload = {
        "is_valid": result.is_valid,
        "receptor": result.receptor,
        "ligand_smiles": result.ligand_smiles,
        "canonical_smiles": result.canonical_smiles,
        "predicted_class": result.predicted_class,
        "class_id": int(result.class_id),
        "prob_agonist": float(result.prob_agonist),
        "prob_antagonist": float(result.prob_antagonist),
        "prob_inactive": float(result.prob_inactive),
        "prob_std_error": float(result.prob_std_error) if result.prob_std_error is not None else None,
        "error": result.error,
    }
    print(json.dumps(payload))
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
