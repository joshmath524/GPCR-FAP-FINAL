"""
Post-prediction docking helpers for the GPCR Streamlit GUI.

This module keeps the docking trigger explicit (user-clicked) and uses receptor-specific
grid centers from each receptor folder's `<id>_ligand_only.pdb`.
"""
from __future__ import annotations

import math
import os
import re
import json
import stat
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .structure_view import resolve_receptor_structure_paths

try:
    import py3Dmol
except ImportError:
    py3Dmol = None


# Match MBind defaults for consistency.
DEFAULT_EXHAUSTIVENESS = 64
DEFAULT_NUM_MODES = 10
DEFAULT_SEED = 42
DEFAULT_TIMEOUT_S = 300

# Tan cartoon used elsewhere; post-docking view uses white receptor + emphasized ligand.
RECEPTOR_CARTOON_HEX = "d2b48c"
DOCKED_RECEPTOR_CARTOON_HEX = "ffffff"
DOCKED_LIGAND_STICK_HEX = "00c853"
DOCKED_VIEW_BG = "#e8eef5"

# Closest-residue contact visualization (MBind-style dashed cylinders)
_SOLVENT_RESN = {"HOH", "WAT", "SOL", "TIP", "TIP3", "HO4"}
_METAL_RESN = {"ZN", "MG", "FE", "CU", "FE2", "MN", "NI", "CO", "CD", "NA", "CL", "CA"}
_AROM_RESN = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"}
_POLAR_ELEM = {"O", "N", "S", "P", "F", "CL", "BR", "I"}


class _ViewAtom(NamedTuple):
    chain: str
    resi: int
    resn: str
    atom_name: str
    x: float
    y: float
    z: float
    elem: str  # element symbol (best-effort)


def _elem_from_atom_line(line: str) -> str:
    if len(line) >= 78:
        e = line[76:78].strip().upper()
        if e:
            if len(e) == 2 and e in ("BR", "CL", "FE", "ZN", "MG", "NA", "MN", "CU", "CO", "NI", "SE"):
                return e
            return e[0]
    name = line[12:16].strip().upper() if len(line) >= 16 else ""
    if not name:
        return "C"
    if name[:2] in ("BR", "CL"):
        return name[:2]
    return name[0]


def _parse_receptor_pdb_heavy_atoms(pdb_text: str) -> List[_ViewAtom]:
    """Protein ATOMs only (standard residues), heavy atoms."""
    from .structure_view import _STANDARD_PROTEIN_RESN

    atoms: List[_ViewAtom] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        resn = line[17:20].strip().upper()
        if resn not in _STANDARD_PROTEIN_RESN:
            continue
        elem = _elem_from_atom_line(line)
        if elem in ("H", "D"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        chain = line[21:22].strip() if len(line) > 21 else "A"
        if not chain:
            chain = "A"
        try:
            resi = int(line[22:26].strip())
        except ValueError:
            continue
        atom_name = line[12:16].strip()
        atoms.append(_ViewAtom(chain, resi, resn, atom_name, x, y, z, elem))
    return atoms


def _parse_pose_heavy_atoms(pose_block: str) -> List[_ViewAtom]:
    """Ligand pose from PDBQT/PDB-like ATOM/HETATM lines."""
    atoms: List[_ViewAtom] = []
    for line in pose_block.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        parts = line.split()
        ad4_or_elem = (parts[-1] if parts else "").strip()
        if ad4_or_elem and ad4_or_elem[0] in ("H", "h"):
            continue
        elem = _elem_from_atom_line(line)
        if elem in ("H", "D"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        chain = line[21:22].strip() if len(line) > 21 else "L"
        if not chain:
            chain = "L"
        try:
            resi = int(line[22:26].strip())
        except ValueError:
            resi = 1
        resn = line[17:20].strip().upper() if len(line) > 20 else "UNL"
        atom_name = line[12:16].strip()
        atoms.append(_ViewAtom(chain, resi, resn, atom_name, x, y, z, elem))
    return atoms


def _dist_atoms(a: _ViewAtom, b: _ViewAtom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _closest_protein_residues(
    rec_atoms: List[_ViewAtom], lig_atoms: List[_ViewAtom], n: int = 3
) -> List[Tuple[str, int]]:
    if not lig_atoms:
        return []
    best: Dict[Tuple[str, int], float] = {}
    for ra in rec_atoms:
        ru = ra.resn.upper()
        if ru in _SOLVENT_RESN or ru in _METAL_RESN:
            continue
        d_min = min(_dist_atoms(ra, la) for la in lig_atoms)
        key = (ra.chain, ra.resi)
        if key not in best or d_min < best[key]:
            best[key] = d_min
    ranked = sorted(best.items(), key=lambda kv: kv[1])
    return [k for k, _ in ranked[:n]]


def _best_contact_pair(
    rec_atoms: List[_ViewAtom],
    lig_atoms: List[_ViewAtom],
    chain: str,
    resi: int,
) -> Optional[Tuple[_ViewAtom, _ViewAtom, float]]:
    r_atoms = [a for a in rec_atoms if a.chain == chain and a.resi == resi]
    if not r_atoms or not lig_atoms:
        return None
    best: Optional[Tuple[_ViewAtom, _ViewAtom, float]] = None
    for ra in r_atoms:
        for la in lig_atoms:
            d = _dist_atoms(ra, la)
            if best is None or d < best[2]:
                best = (ra, la, d)
    return best


def _interaction_line_color(ra: _ViewAtom, la: _ViewAtom, d: float) -> str:
    """Teal (polar), green (aromatic C–C), slate otherwise — aligned with MBind logic."""
    if ra.elem in _POLAR_ELEM or la.elem in _POLAR_ELEM:
        return "0x2a9d9d"
    if (
        d < 4.8
        and ra.elem == "C"
        and la.elem == "C"
        and (ra.resn.upper() in _AROM_RESN or la.resn.upper() in _AROM_RESN)
    ):
        return "0x3d8f3d"
    return "0x7a8fa3"


def _add_dashed_line(
    view,
    a: _ViewAtom,
    b: _ViewAtom,
    color: str,
    dash_len: float = 0.45,
    gap_len: float = 0.12,
    radius: float = 0.05,
) -> None:
    dx = b.x - a.x
    dy = b.y - a.y
    dz = b.z - a.z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-6:
        return
    ux, uy, uz = dx / dist, dy / dist, dz / dist
    pos = 0.0
    min_seg = 0.08
    while pos < dist:
        seg_end = min(pos + dash_len, dist)
        if seg_end - pos >= min_seg:
            view.addCylinder(
                {
                    "start": {"x": a.x + ux * pos, "y": a.y + uy * pos, "z": a.z + uz * pos},
                    "end": {"x": a.x + ux * seg_end, "y": a.y + uy * seg_end, "z": a.z + uz * seg_end},
                    "radius": radius,
                    "color": color,
                    "fromCap": True,
                    "toCap": True,
                }
            )
        pos = seg_end + gap_len


def _apply_closest_residue_highlights_and_contacts(
    view,
    rec_atoms: List[_ViewAtom],
    lig_atoms: List[_ViewAtom],
    *,
    receptor_cartoon_hex: str,
    contact_max_dist: float = 5.0,
    n_residues: int = 3,
) -> None:
    """Highlight closest residues (cartoon + sticks) and dashed lines to ligand (MBind-style)."""
    tan_hex = receptor_cartoon_hex.lstrip("#").lower()
    closest = _closest_protein_residues(rec_atoms, lig_atoms, n=n_residues)
    for chain, resi in closest:
        view.setStyle(
            {"model": 0, "chain": chain, "resi": resi},
            {
                "cartoon": {"color": f"0x{tan_hex}", "opacity": 0.38},
                "stick": {"radius": 0.13, "color": "0xb0bec5"},
            },
        )
    for chain, resi in closest:
        pair = _best_contact_pair(rec_atoms, lig_atoms, chain, resi)
        if pair is None:
            continue
        ra, la, d = pair
        if d > contact_max_dist:
            continue
        _add_dashed_line(view, ra, la, _interaction_line_color(ra, la, d))


def _contact_type_label(ra: _ViewAtom, la: _ViewAtom, d: float) -> str:
    col = _interaction_line_color(ra, la, d)
    if col == "0x2a9d9d":
        return "polar / H-bond–like"
    if col == "0x3d8f3d":
        return "aromatic (C–C)"
    return "van der Waals / hydrophobic"


def build_closest_contact_summary(
    receptor_pdb_text: str, pose_pdb_text: str, n: int = 3, contact_max_dist: float = 5.0
) -> List[str]:
    """Human-readable lines for the n closest protein residues to the docked ligand (MBind-style geometry)."""
    rec_atoms = _parse_receptor_pdb_heavy_atoms(receptor_pdb_text)
    lig_atoms = _parse_pose_heavy_atoms(pose_pdb_text)
    if not rec_atoms or not lig_atoms:
        return []
    lines: List[str] = []
    for chain, resi in _closest_protein_residues(rec_atoms, lig_atoms, n=n):
        pair = _best_contact_pair(rec_atoms, lig_atoms, chain, resi)
        if pair is None:
            continue
        ra, la, d = pair
        if d > contact_max_dist:
            continue
        label = _contact_type_label(ra, la, d)
        lines.append(f"{chain} {ra.resn} {resi} — {d:.2f} Å ({label})")
    return lines


@dataclass
class DockingResult:
    ok: bool
    message: str
    receptor_name: str
    canonical_smiles: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    score_kcal_mol: Optional[float]
    engine: str
    command: str
    out_pose_path: Optional[str]
    log_path: Optional[str]
    html: Optional[str]
    contact_summary: Optional[List[str]] = None


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mbind_source_files_dir(project_root: Path) -> Path:
    # Workspace layout in this project: <Final GUI>/<repo>, with MBind-main as sibling.
    return project_root.parent / "MBind-main" / "Files_for_GUI"


def ensure_docking_files_folder(project_root: Optional[Path] = None) -> Tuple[Path, str]:
    """
    Ensure a local docking assets folder exists inside this GUI project.
    Syncs SMINA binaries from sibling MBind-main/Files_for_GUI when available.
    """
    root = project_root or _resolve_project_root()
    dst = root / "docking_assets"
    src = _mbind_source_files_dir(root)
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        _write_receptor_grid_manifest(root, dst)
        _ensure_local_binaries_executable(dst)
        return dst, f"docking_assets ready at {dst}."
    copied = 0
    # Keep synced assets minimal and purpose-specific for SMINA pose generation.
    for name in ("smina", "smina.exe"):
        path = src / name
        if not path.is_file():
            continue
        out = dst / name
        try:
            src_size = path.stat().st_size
            needs_copy = (not out.exists()) or (out.stat().st_size != src_size)
        except OSError:
            needs_copy = True
        if needs_copy:
            shutil.copy2(path, out)
            copied += 1
    _write_receptor_grid_manifest(root, dst)
    _ensure_local_binaries_executable(dst)
    return dst, f"docking_assets synced from MBind source ({copied} updated file(s))."


def _write_receptor_grid_manifest(project_root: Path, docking_files_dir: Path) -> None:
    receptor_root = project_root / "Josh_Receptor_Features"
    if not receptor_root.is_dir():
        return
    manifest = {}
    for folder in sorted(p.name for p in receptor_root.iterdir() if p.is_dir()):
        rec, lig = resolve_receptor_structure_paths(folder)
        if rec is None or lig is None or not lig.is_file():
            continue
        coords = _ligand_heavy_coords_from_pdb(lig.read_text(encoding="utf-8", errors="ignore"))
        if coords is None:
            continue
        center, size = _grid_from_ligand_coords(coords)
        pdb_id = lig.name.replace("_ligand_only.pdb", "")
        manifest[folder] = {
            "pdb_id": pdb_id,
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
        }
    out = docking_files_dir / "receptor_grid_boxes.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _is_executable_on_platform(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name == "nt":
        return path.suffix.lower() == ".exe"
    return os.access(path, os.X_OK)


def _try_make_executable(path: Path) -> bool:
    if not path.is_file() or os.name == "nt":
        return _is_executable_on_platform(path)
    try:
        mode = path.stat().st_mode
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        return False
    return os.access(path, os.X_OK)


def _ensure_local_binaries_executable(files_dir: Path) -> None:
    """
    Linux deployments often copy binaries without +x bit. Set execute bits proactively.
    """
    if os.name == "nt":
        return
    for name in ("vina", "smina", "gnina", "autogrid4", "autodock4"):
        p = files_dir / name
        if p.is_file():
            _try_make_executable(p)


def _download_smina_linux(target_path: Path) -> Tuple[bool, str]:
    """
    Download a Linux SMINA binary when not bundled.
    Uses public SourceForge endpoint for the static build.
    """
    if os.name == "nt":
        return False, "Auto-download is only enabled for Linux deployments."
    urls = [
        "https://sourceforge.net/projects/smina/files/smina.static/download",
    ]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            if not data or len(data) < 1024:
                continue
            target_path.write_bytes(data)
            if _try_make_executable(target_path):
                return True, f"Downloaded SMINA to {target_path}."
        except Exception:
            continue
    return False, "Could not auto-download SMINA binary from public mirror."


def _pdb_line_element_symbol(line: str) -> str:
    elem = ""
    if len(line) >= 78:
        elem = line[76:78].strip().upper()
    if elem:
        return elem
    if len(line) < 16:
        return ""
    name = line[12:16].strip().upper()
    if not name:
        return ""
    two = name[:2]
    if two in ("BR", "CL", "FE", "ZN", "MG", "CA", "NA", "MN", "CU", "CO", "NI", "SE"):
        return two
    if name[0].isdigit() and len(name) > 1 and name[1].isalpha():
        return name[1:2]
    return name[0]


def _ligand_heavy_coords_from_pdb(pdb_text: str) -> Optional[np.ndarray]:
    coords = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        elem = _pdb_line_element_symbol(line)
        if elem in ("H", "D"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append([x, y, z])
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64)


def _grid_from_ligand_coords(coords: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    center = coords.mean(axis=0)
    span = np.ptp(coords, axis=0)
    # Keep box dimensions in requested range (15-20 A), with small pad around the ligand extent.
    size = np.clip(span + 10.0, 15.0, 20.0)
    return (float(center[0]), float(center[1]), float(center[2])), (
        float(size[0]),
        float(size[1]),
        float(size[2]),
    )


def compute_receptor_grid_params(receptor_folder: str) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], str]:
    rec_path, lig_path = resolve_receptor_structure_paths(receptor_folder)
    if rec_path is None or not rec_path.is_file():
        return None, None, "Missing receptor-only PDB for this target."
    if lig_path is None or not lig_path.is_file():
        return None, None, "Missing ligand-only PDB for this target."
    lig_text = lig_path.read_text(encoding="utf-8", errors="ignore")
    coords = _ligand_heavy_coords_from_pdb(lig_text)
    if coords is None:
        return None, None, "Could not parse heavy-atom coordinates from ligand-only PDB."
    center, size = _grid_from_ligand_coords(coords)
    return center, size, "ok"


def _select_docking_engine(files_dir: Path) -> Tuple[Optional[str], Optional[Path]]:
    """
    Pose generation is SMINA-only by design.
    Prefer local docking_assets binary, then PATH.
    """
    is_windows = os.name == "nt"
    local_name = "smina.exe" if is_windows else "smina"
    p = files_dir / local_name
    if p.is_file() and (_is_executable_on_platform(p) or _try_make_executable(p)):
        return "smina", p

    found = shutil.which("smina")
    if found:
        return "smina", Path(found)
    if is_windows:
        found_exe = shutil.which("smina.exe")
        if found_exe:
            return "smina", Path(found_exe)

    # Last resort on Linux deployments: auto-provision SMINA in docking_assets.
    if not is_windows:
        ok, _ = _download_smina_linux(files_dir / "smina")
        p2 = files_dir / "smina"
        if ok and p2.is_file() and (_is_executable_on_platform(p2) or _try_make_executable(p2)):
            return "smina", p2
    return None, None


def _sanitize_receptor_pdb_for_view(pdb_text: str) -> str:
    protein_resn = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE", "HID", "HIP",
        "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
    out = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM"):
            continue
        if line.startswith("ATOM") and len(line) > 20:
            if line[17:20].strip().upper() not in protein_resn:
                continue
        out.append(line)
    return "\n".join(out)


def _extract_top_pose_pdb_block(docked_path: Path) -> Optional[str]:
    try:
        lines = docked_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    in_model = False
    saw_model = False
    pose_lines = []
    for line in lines:
        if line.startswith("MODEL"):
            if saw_model:
                break
            saw_model = True
            in_model = True
            continue
        if line.startswith("ENDMDL"):
            if in_model:
                break
            continue
        if saw_model and not in_model:
            continue
        if line.startswith(("ATOM", "HETATM")):
            pose_lines.append(line)
    if not pose_lines:
        return None
    return "\n".join(pose_lines) + "\n"


def _extract_best_score_kcal_mol(docked_path: Path) -> Optional[float]:
    try:
        txt = docked_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    # Vina/SMINA/GNINA often emit: REMARK VINA RESULT: -7.2 ...
    m = re.search(r"REMARK\s+VINA\s+RESULT:\s+(-?\d+(?:\.\d+)?)", txt)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_best_score_from_stdout(stdout_text: str) -> Optional[float]:
    # Vina/smina table usually has a first row: "1   -7.2 ..."
    for line in (stdout_text or "").splitlines():
        m = re.match(r"^\s*1\s+(-?\d+(?:\.\d+)?)\b", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _build_docked_complex_html(receptor_pdb_text: str, pose_pdb_text: str, width: int = 720, height: int = 520) -> Optional[str]:
    if py3Dmol is None:
        return None
    if not receptor_pdb_text.strip() or not pose_pdb_text.strip():
        return None
    try:
        rec_atoms = _parse_receptor_pdb_heavy_atoms(receptor_pdb_text)
        lig_atoms = _parse_pose_heavy_atoms(pose_pdb_text)
        view = py3Dmol.view(width=width, height=height)
        # Light background; white receptor reads clearly with AO shading on folds.
        view.setBackgroundColor(DOCKED_VIEW_BG)
        view.enableFog(False)
        try:
            view.setViewStyle({"style": "ambientOcclusion", "strength": 0.42, "radius": 3.0})
        except Exception:
            pass
        view.addModel(receptor_pdb_text, "pdb")
        view.setStyle(
            {"model": 0},
            {"cartoon": {"color": f"0x{DOCKED_RECEPTOR_CARTOON_HEX}", "style": "rectangle", "thickness": 0.35}},
        )
        if rec_atoms and lig_atoms:
            _apply_closest_residue_highlights_and_contacts(
                view,
                rec_atoms,
                lig_atoms,
                receptor_cartoon_hex=DOCKED_RECEPTOR_CARTOON_HEX,
            )
        try:
            view.addModel(pose_pdb_text, "pdbqt")
        except Exception:
            view.addModel(pose_pdb_text, "pdb")
        view.setStyle(
            {"model": 1},
            {"stick": {"radius": 0.22, "color": f"0x{DOCKED_LIGAND_STICK_HEX}"}},
        )
        view.zoomTo()
        return view._make_html()
    except Exception:
        return None


def _smiles_to_sdf_file(smiles: str, out_path: Path) -> Tuple[bool, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Could not parse canonical SMILES for docking."
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = DEFAULT_SEED
        emb = AllChem.EmbedMolecule(mol, params)
    except Exception:
        emb = AllChem.EmbedMolecule(mol, randomSeed=DEFAULT_SEED)
    if emb != 0:
        return False, "RDKit failed to generate a 3D conformer for docking."
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return True, "ok"


def run_single_receptor_docking(
    receptor_folder: str,
    canonical_smiles: str,
    base_exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    base_num_modes: int = DEFAULT_NUM_MODES,
    grid_center: Optional[Tuple[float, float, float]] = None,
    grid_size: Optional[Tuple[float, float, float]] = None,
) -> DockingResult:
    project_root = _resolve_project_root()
    files_dir, sync_msg = ensure_docking_files_folder(project_root)
    engine, engine_path = _select_docking_engine(files_dir)
    if engine_path is None:
        return DockingResult(
            ok=False,
            message=(
                f"{sync_msg} No SMINA binary is available. "
                "The app searched `docking_assets`, PATH, and attempted Linux auto-download. "
                f"Add `smina` to `{files_dir}` or install it in PATH. "
                "On Linux, ensure execute permission (`chmod +x`)."
            ),
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            score_kcal_mol=None,
            engine="none",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    rec_path, lig_path = resolve_receptor_structure_paths(receptor_folder)
    if rec_path is None or not rec_path.is_file():
        return DockingResult(
            ok=False,
            message="No receptor-only PDB found for selected receptor.",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )
    if lig_path is None or not lig_path.is_file():
        return DockingResult(
            ok=False,
            message="No ligand-only PDB found for selected receptor.",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    lig_coords = _ligand_heavy_coords_from_pdb(lig_path.read_text(encoding="utf-8", errors="ignore"))
    if lig_coords is None:
        return DockingResult(
            ok=False,
            message="Could not parse ligand-only PDB coordinates to build a docking grid.",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )
    center, size = _grid_from_ligand_coords(lig_coords)
    if grid_center is not None:
        center = (float(grid_center[0]), float(grid_center[1]), float(grid_center[2]))
    if grid_size is not None:
        sx, sy, sz = float(grid_size[0]), float(grid_size[1]), float(grid_size[2])
        if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
            return DockingResult(
                ok=False,
                message="Grid size must be positive on every axis (size_x, size_y, size_z).",
                receptor_name=receptor_folder,
                canonical_smiles=canonical_smiles,
                center=center,
                size=size,
                score_kcal_mol=None,
                engine=engine or "unknown",
                command="",
                out_pose_path=None,
                log_path=None,
                html=None,
            )
        size = (sx, sy, sz)

    runs_dir = project_root / "docking_results"
    run_dir = runs_dir / f"{receptor_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ligand_sdf = run_dir / "query_ligand.sdf"
    docked_out = run_dir / "docked_topposes.pdbqt"
    log_path = run_dir / "docking.log"

    ok_prep, prep_msg = _smiles_to_sdf_file(canonical_smiles, ligand_sdf)
    if not ok_prep:
        return DockingResult(
            ok=False,
            message=prep_msg,
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command="",
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    cmd = [
        str(engine_path),
        "-r",
        str(rec_path),
        "-l",
        str(ligand_sdf),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--num_modes",
        str(int(base_num_modes)),
        "--exhaustiveness",
        str(int(base_exhaustiveness)),
        "--seed",
        str(DEFAULT_SEED),
        "--scoring",
        "vina",
        "-o",
        str(docked_out),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return DockingResult(
            ok=False,
            message=f"Docking timed out after {DEFAULT_TIMEOUT_S}s.",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command=" ".join(cmd),
            out_pose_path=None,
            log_path=None,
            html=None,
        )
    except Exception as e:
        return DockingResult(
            ok=False,
            message=f"Docking failed to launch: {e}",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command=" ".join(cmd),
            out_pose_path=None,
            log_path=None,
            html=None,
        )

    log_path.write_text(
        "\n".join(
            [
                f"Engine: {engine_path}",
                f"Command: {' '.join(cmd)}",
                f"Return code: {proc.returncode}",
                "",
                "---- STDOUT ----",
                proc.stdout or "",
                "",
                "---- STDERR ----",
                proc.stderr or "",
            ]
        ),
        encoding="utf-8",
    )

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "").strip()
        return DockingResult(
            ok=False,
            message=(
                "Docking engine returned a non-zero exit code. "
                f"See log: {log_path}. "
                f"Engine output: {err_tail[:220]}"
            ),
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command=" ".join(cmd),
            out_pose_path=None,
            log_path=str(log_path),
            html=None,
        )
    if not docked_out.exists() or docked_out.stat().st_size == 0:
        return DockingResult(
            ok=False,
            message=f"Docking completed but no output pose was written: {docked_out}",
            receptor_name=receptor_folder,
            canonical_smiles=canonical_smiles,
            center=center,
            size=size,
            score_kcal_mol=None,
            engine=engine or "unknown",
            command=" ".join(cmd),
            out_pose_path=None,
            log_path=str(log_path),
            html=None,
        )

    score = _extract_best_score_from_stdout(proc.stdout or "")
    if score is None:
        score = _extract_best_score_kcal_mol(docked_out)
    pose_text = _extract_top_pose_pdb_block(docked_out)
    rec_text = _sanitize_receptor_pdb_for_view(rec_path.read_text(encoding="utf-8", errors="ignore"))
    html = _build_docked_complex_html(rec_text, pose_text or "")
    contact_lines: Optional[List[str]] = None
    if pose_text and rec_text.strip():
        contact_lines = build_closest_contact_summary(rec_text, pose_text)

    return DockingResult(
        ok=True,
        message=f"{sync_msg} Docking completed successfully.",
        receptor_name=receptor_folder,
        canonical_smiles=canonical_smiles,
        center=center,
        size=size,
        score_kcal_mol=score,
        engine=engine or "unknown",
        command=" ".join(cmd),
        out_pose_path=str(docked_out),
        log_path=str(log_path),
        html=html,
        contact_summary=contact_lines,
    )

