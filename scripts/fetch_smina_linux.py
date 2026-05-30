#!/usr/bin/env python3
"""Fetch Linux SMINA from conda-forge and install as docking_assets/smina."""
from __future__ import annotations

import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docking_assets" / "smina"
URL = (
    "https://conda.anaconda.org/conda-forge/linux-64/smina-2020.12.10-hecca717_2.conda"
)


def main() -> None:
    try:
        import zstandard as zstd
    except ImportError:
        print("Install zstandard: py -3 -m pip install zstandard", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {URL} ...")
    data = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "GPCR-FAP/1.0"}),
        timeout=180,
    ).read()
    outer = zipfile.ZipFile(io.BytesIO(data))
    pkg_name = next(n for n in outer.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
    tar_bytes = zstd.ZstdDecompressor().decompress(outer.read(pkg_name))
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("bin/smina"))
        blob = tf.extractfile(member)
        if blob is None:
            raise RuntimeError("bin/smina missing from conda package")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(blob.read())
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
