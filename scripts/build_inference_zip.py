#!/usr/bin/env python3
"""
Build a Streamlit Cloud–friendly data zip: Josh_Receptor_Features + ML_code + *_NEW.xlsx only.

Usage:
  set MANUSCRIPT_DATA_ROOT=C:\\path\\to\\GPCRtryagain - Delete - Copy
  py -3 scripts/build_inference_zip.py
  py -3 scripts/build_inference_zip.py --output C:\\Users\\you\\GPCR-inference-cloud.zip
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(
    os.environ.get(
        "MANUSCRIPT_DATA_ROOT",
        r"C:\Users\joshmatchem\Music\GPCR-FAP-main (2)\GPCRtryagain - Delete - Copy",
    )
)
WORKBOOK_NAMES = (
    "agonist_enriched_NEW.xlsx",
    "antagonist_enriched_NEW.xlsx",
    "non_active_compounds_NEW.xlsx",
)


def _add_tree(zf: zipfile.ZipFile, src: Path, arc_prefix: str) -> int:
    n = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        arc = f"{arc_prefix}/{path.relative_to(src).as_posix()}"
        zf.write(path, arc)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inference zip for Streamlit Cloud")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "GPCR-inference-cloud.zip")
    args = parser.parse_args()
    data = args.data_root.resolve()
    if not (data / "Josh_Receptor_Features").is_dir():
        raise SystemExit(f"Missing Josh_Receptor_Features under {data}")
    if not (data / "ML_code").is_dir():
        raise SystemExit(f"Missing ML_code under {data}")

    out = args.output.resolve()
    if out.exists():
        out.unlink()
    staging = out.with_suffix(".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copytree(data / "Josh_Receptor_Features", staging / "Josh_Receptor_Features")
    shutil.copytree(data / "ML_code", staging / "ML_code")

    workbook_files = 0
    for child in sorted(data.iterdir()):
        if not child.is_dir() or child.name in ("Josh_Receptor_Features", "ML_code"):
            continue
        dest_dir = staging / child.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for suffix in WORKBOOK_NAMES:
            src = child / f"{child.name}_{suffix}"
            if src.is_file():
                shutil.copy2(src, dest_dir / src.name)
                workbook_files += 1

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        files = 0
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())
                files += 1

    shutil.rmtree(staging)
    mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({mb:.1f} MB, {files} files, {workbook_files} workbooks)")
    print("Upload to Google Drive and set DATA_DRIVE_FILE_ID in Streamlit secrets.")
    print("DATA_EXTRACTED_SUBDIR can stay empty if zip unpacks with Josh_Receptor_Features at top level.")


if __name__ == "__main__":
    main()
