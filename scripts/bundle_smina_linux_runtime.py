#!/usr/bin/env python3
"""Bundle Linux SMINA + conda shared libraries for Streamlit Cloud."""
from __future__ import annotations

import io
import re
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

LIB_PREFIXES = ("libboost", "libopenbabel", "libinchi")


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
                    tf.extract(member, dest, filter="data")
        return
    outer = zipfile.ZipFile(io.BytesIO(data))
    pkg_name = next(n for n in outer.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
    tar_bytes = zstd.ZstdDecompressor().decompress(outer.read(pkg_name))
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for member in tf.getmembers():
            name = member.name
            if name.startswith("bin/smina") or name.startswith("lib/"):
                tf.extract(member, dest, filter="data")


def _elf_needed_libs(path: Path) -> set[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return set()
    if not data.startswith(b"\x7fELF"):
        return set()
    return {m.group(0).decode("ascii") for m in re.finditer(rb"lib[a-zA-Z0-9_+\-.]+\.so(?:\.[0-9]+)*", data)}


def _copy_runtime_libs(stage: Path, lib_out: Path) -> None:
    """Copy all Boost/OpenBabel libs, then close over transitive Boost deps."""
    lib_out.mkdir(parents=True, exist_ok=True)
    pool: dict[str, Path] = {}
    for lib_root in sorted(stage.glob("**/lib")):
        if not lib_root.is_dir():
            continue
        for src in sorted(lib_root.iterdir()):
            if not src.is_file():
                continue
            if not src.name.startswith(LIB_PREFIXES):
                continue
            if src.stat().st_size <= 0:
                continue
            pool[src.name] = src

    required = set(_elf_needed_libs(stage / "bin" / "smina"))
    while True:
        added = False
        for lib_name in list(required):
            if not lib_name.startswith(("libboost", "libopenbabel", "libinchi")):
                continue
            src = pool.get(lib_name)
            if src is None:
                continue
            for dep in _elf_needed_libs(src):
                if dep.startswith(("libboost", "libopenbabel", "libinchi")) and dep not in required:
                    required.add(dep)
                    added = True
        if not added:
            break

    missing = sorted(
        name
        for name in required
        if name.startswith(("libboost", "libopenbabel", "libinchi")) and name not in pool
    )
    if missing:
        print("Missing runtime libraries:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)

    copied = 0
    for name in sorted(required):
        if not name.startswith(("libboost", "libopenbabel", "libinchi")):
            continue
        src = pool[name]
        dst = lib_out / name
        shutil.copy2(src, dst)
        copied += 1

    # Ship the full Boost 1.82 stack (small enough) to avoid future missing .so errors.
    for name, src in sorted(pool.items()):
        if not name.startswith("libboost"):
            continue
        dst = lib_out / name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    print(f"  copied {copied} shared library file(s)")


def main() -> None:
    _ensure_zstd()
    if OUT.exists():
        shutil.rmtree(OUT)
    stage = OUT / "_stage"
    stage.mkdir(parents=True)
    for url in PACKAGES.values():
        _extract_conda(url, stage)

    BIN.parent.mkdir(parents=True)
    shutil.copy2(stage / "bin" / "smina", BIN)
    _copy_runtime_libs(stage, LIB)

    shutil.rmtree(stage)
    legacy = ROOT / "docking_assets" / "smina"
    shutil.copy2(BIN, legacy)
    print(f"Bundled SMINA runtime under {OUT}")
    print(f"  binary: {BIN} ({BIN.stat().st_size} bytes)")
    print(f"  libs:   {len(list(LIB.iterdir()))} files")


if __name__ == "__main__":
    main()
