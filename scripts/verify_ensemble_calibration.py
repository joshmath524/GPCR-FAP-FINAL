#!/usr/bin/env python3
"""Quick check that ensemble probs are not pinned at ~0.97 inactive."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = Path(
    os.environ.get(
        "MANUSCRIPT_DATA_ROOT",
        r"C:\Users\joshmatchem\Music\GPCR-FAP-main (2)\GPCRtryagain - Delete - Copy",
    )
)
os.environ.setdefault("GPCR_DATA_ROOT", str(DATA))
os.environ.setdefault("MANUSCRIPT_ML_ROOT", str(DATA / "ML_code"))

from gpcr.predict import load_predictor, predict_single

CASES = [
    ("salbutamol", "CC(C)NCC(O)c1cc(O)c(O)cc1O"),
    ("ethanol", "CCO"),
]


def main() -> None:
    pkl = ROOT / "artifacts" / "manuscript" / "independent_ligand" / "ensemble" / "stacking_seed42.pkl"
    if not pkl.exists():
        print(f"Missing {pkl} — run scripts/run_export_ensemble_oof.ps1 first.")
        sys.exit(1)

    pred = load_predictor(
        str(ROOT), model_type="ensemble", evaluation_regime="independent_ligand", seed=42
    )
    has_scaler = getattr(pred.models[0], "tree_scaler", None) is not None
    print(f"tree_scaler present: {has_scaler}")

    max_inactive = 0.0
    for name, smi in CASES:
        r = predict_single("beta2", smi, predictor=pred)
        max_inactive = max(max_inactive, r.prob_inactive)
        print(
            f"{name}: {r.predicted_class} "
            f"[{r.prob_agonist:.3f}, {r.prob_antagonist:.3f}, {r.prob_inactive:.3f}]"
        )

    if max_inactive > 0.95:
        print("WARN: inactive prob still very high (>0.95) on all cases")
        sys.exit(2)
    print("OK: ensemble probabilities look less extreme than ~0.97 pinned")


if __name__ == "__main__":
    main()
