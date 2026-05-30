#!/usr/bin/env python3
"""Copy real *_receptor_only.pdb files into docking_assets/receptor_pdbs/ for Cloud SMINA."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpcr.docking import _is_lfs_pointer_file  # noqa: E402


def _find_josh_root(project_root: Path) -> Optional[Path]:
    for base in (
        project_root,
        project_root.parent / "GUI_Folder",
        Path(__file__).resolve().parents[2] / "GPCRtryagain - Delete - Copy",
    ):
        p = base / "Josh_Receptor_Features"
        if p.is_dir():
            return p
    return None


def main() -> None:
    josh = _find_josh_root(ROOT)
    if josh is None:
        print("Josh_Receptor_Features not found — set GPCR_DATA_ROOT or place it beside the project.")
        sys.exit(1)

    out_root = ROOT / "docking_assets" / "receptor_pdbs"
    out_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for folder in sorted(p for p in josh.iterdir() if p.is_dir()):
        src_files = sorted(folder.glob("*_receptor_only.pdb"))
        if not src_files:
            continue
        src = src_files[0]
        if _is_lfs_pointer_file(src):
            print(f"skip LFS {src}")
            continue
        dest_dir = out_root / folder.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        n += 1
    print(f"Bundled {n} receptor-only PDB(s) under {out_root}")


if __name__ == "__main__":
    main()
