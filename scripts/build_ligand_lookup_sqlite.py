#!/usr/bin/env python3
"""
Build ligand_feature_lookup.sqlite for low-RAM inference (Streamlit Cloud).

Reads existing ligand_feature_lookup.joblib if present, else builds from *_NEW.xlsx
(same sources as build_manuscript_ligand_lookup.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpcr.ligand_lookup_store import (  # noqa: E402
    bulk_insert,
    create_writable,
    encode_feature_dict,
    sqlite_lookup_path,
)

MANUSCRIPT = ROOT / "artifacts" / "manuscript"
JOBLIB_PATH = MANUSCRIPT / "ligand_feature_lookup.joblib"
META_PATH = MANUSCRIPT / "ligand_feature_lookup_meta.json"
BATCH = 500


def _from_joblib() -> tuple[dict[str, dict[str, float]], str]:
    import joblib

    if not JOBLIB_PATH.is_file():
        raise FileNotFoundError(JOBLIB_PATH)
    print(f"Loading {JOBLIB_PATH} ({JOBLIB_PATH.stat().st_size // (1024 * 1024)} MB)...", flush=True)
    payload = joblib.load(JOBLIB_PATH)
    lookup = payload.get("lookup", {})
    source = str(payload.get("source", "_NEW.xlsx"))
    return lookup, source


def main() -> None:
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    out_path = sqlite_lookup_path(MANUSCRIPT)
    tmp_path = out_path.with_suffix(".sqlite.tmp")

    if not JOBLIB_PATH.is_file():
        print(f"Missing {JOBLIB_PATH}. Run: py -3 scripts/build_manuscript_ligand_lookup.py")
        sys.exit(1)
    lookup, source = _from_joblib()

    if not lookup:
        print("FAIL: empty lookup")
        sys.exit(1)

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Writing {len(lookup):,} rows to {tmp_path}...", flush=True)
    con = create_writable(tmp_path)
    batch: list[tuple[str, bytes]] = []
    try:
        for i, (smi, feats) in enumerate(lookup.items(), start=1):
            batch.append((smi, encode_feature_dict(feats)))
            if len(batch) >= BATCH:
                bulk_insert(con, batch)
                con.commit()
                batch.clear()
            if i % 5000 == 0:
                print(f"  {i:,} rows...", flush=True)
        if batch:
            bulk_insert(con, batch)
            con.commit()
    finally:
        con.close()

    if out_path.exists():
        out_path.unlink()
    tmp_path.replace(out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    meta = {
        "n_smiles": len(lookup),
        "source": source,
        "storage": "sqlite",
        "sqlite_bytes": out_path.stat().st_size,
        "joblib_bytes": JOBLIB_PATH.stat().st_size if JOBLIB_PATH.is_file() else None,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")
    print(f"Updated {META_PATH}")


if __name__ == "__main__":
    main()
