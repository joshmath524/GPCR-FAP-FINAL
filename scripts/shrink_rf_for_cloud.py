#!/usr/bin/env python3
"""
Build model_seed42_cloud.pkl — RF with fewer trees for Streamlit Cloud (~1 GB RAM).

Uses the first N trees from the full exported forest (inference-only; not retrained).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "artifacts" / "manuscript" / "independent_ligand" / "rf" / "model_seed42.pkl"
DEFAULT_OUT = ROOT / "artifacts" / "manuscript" / "independent_ligand" / "rf" / "model_seed42_cloud.pkl"


def shrink_rf(full: RandomForestClassifier, n_trees: int) -> RandomForestClassifier:
    n = min(int(n_trees), len(full.estimators_))
    small = RandomForestClassifier(n_estimators=n, n_jobs=1)
    small.estimators_ = full.estimators_[:n]
    small.classes_ = full.classes_
    for attr in ("n_features_in_", "n_outputs_", "n_classes_"):
        if hasattr(full, attr):
            setattr(small, attr, getattr(full, attr))
    if hasattr(full, "_n_classes"):
        small._n_classes = full._n_classes  # type: ignore[attr-defined]
    return small


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trees", type=int, default=150)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Missing {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input} ({args.input.stat().st_size // (1024 * 1024)} MB)...", flush=True)
    full = joblib.load(args.input)
    if not hasattr(full, "estimators_"):
        print("Not a sklearn RandomForestClassifier", file=sys.stderr)
        sys.exit(1)

    small = shrink_rf(full, args.trees)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(small, args.output, compress=3)
    mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({mb:.1f} MB, {len(small.estimators_)} trees)")


if __name__ == "__main__":
    main()
