#!/usr/bin/env python3
"""Bundle Linux SMINA + conda shared libraries for Streamlit Cloud."""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docking_assets" / "smina_linux"
BIN = OUT / "bin" / "smina"
LIB = OUT / "lib"

PACKAGES = {
    "smina": "https://conda.anaconda.org/conda-forge/linux-64/smina-2020.12.10-hecca717_2.conda",
    "libboost": "https://conda.anaconda.org/conda-forge/linux-64/libboost-1.82.0-h6fcfa73_6.conda",
    "openbabel": "https://conda.anaconda.org/conda-forge/linux-64/openbabel-3.1.1-py311h7c3e0e0_5.tar.bz2",
}

NEEDED = (
    "libboost_filesystem.so.1.82.0",
    "libboost_iostreams.so.1.82.0",
    "libboost_program_options.so.1.82.0",
    "libboost_serialization.so.1.82.0",
    "libboost_thread.so.1.82.0",
    "libboost_timer.so.1.82.0",
    "libopenbabel.so.7",
)


def _ensure_zstd():
    try:
        import zstandard  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zstandard", "-q"])


def _extract_conda(url: str, dest: Path) -> None:
    import bz2
    import zstandard as zstd

    print(f"Fetching {url} ...")
    data = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "GPCR-FAP/1.0"}),
        timeout=180,
    ).read()
    if url.endswith(".tar.bz2"):
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(bz2.decompress(data))) as tf:
            for member in tf.getmembers():
                if member.name.startswith("lib/"):
                    tf.extract(member, dest)
        return
    outer = zipfile.ZipFile(io.BytesIO(data))
    pkg_name = next(n for n in outer.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
    tar_bytes = zstd.ZstdDecompressor().decompress(outer.read(pkg_name))
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for member in tf.getmembers():
            name = member.name
            if name.startswith("bin/smina") or name.startswith("lib/"):
                tf.extract(member, dest)


def main() -> None:
    _ensure_zstd()
    if OUT.exists():
        shutil.rmtree(OUT)
    stage = OUT / "_stage"
    stage.mkdir(parents=True)
    for url in PACKAGES.values():
        _extract_conda(url, stage)

    LIB.mkdir(parents=True)
    BIN.parent.mkdir(parents=True)
    shutil.copy2(stage / "bin" / "smina", BIN)

    copied = set()
    for lib_dir in sorted(stage.glob("**/lib")):
        if not lib_dir.is_dir():
            continue
        for name in NEEDED:
            src = lib_dir / name
            if src.is_file() and name not in copied:
                shutil.copy2(src, LIB / name)
                copied.add(name)

    missing = [n for n in NEEDED if n not in copied]
    if missing:
        print("Missing libraries:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)

    shutil.rmtree(stage)
    # Legacy single-file path used by older code paths.
    legacy = ROOT / "docking_assets" / "smina"
    shutil.copy2(BIN, legacy)
    print(f"Bundled SMINA runtime under {OUT}")
    print(f"  binary: {BIN} ({BIN.stat().st_size} bytes)")
    print(f"  libs:   {len(list(LIB.iterdir()))} files")


if __name__ == "__main__":
    main()
