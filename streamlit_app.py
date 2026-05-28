"""
GPCR Class A Functional Activity Prediction Streamlit GUI.

Run from this folder (project root):
  streamlit run streamlit_app.py
"""
import http.cookiejar
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# Ensure project root (this folder) is on path for src.gpcr
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _artifact_tree_has_models(artifacts_dir: Path) -> bool:
    if not artifacts_dir.is_dir():
        return False
    for sub in artifacts_dir.glob("demo_*"):
        if sub.is_dir():
            if list(sub.glob("model_seed*.pkl")) or list(sub.glob("model_seed*.joblib")):
                return True
            if list(sub.glob("*.pkl")) or list(sub.glob("*.joblib")):
                return True
    if list(artifacts_dir.glob("model_seed*.pkl")) or list(artifacts_dir.glob("model_seed*.joblib")):
        return True
    return bool(list(artifacts_dir.glob("*.pkl")) or list(artifacts_dir.glob("*.joblib")))


def _resolve_handoff_dir() -> Path:
    """
    Resolve the directory passed to load_predictor() (may be .../artifacts or a flat demo bundle).

    Order: ./artifacts in this repo → sibling **artifact sahith** (updated handoff) → ../artifacts → cwd.
    """
    local_art = PROJECT_ROOT / "artifacts"
    if _artifact_tree_has_models(local_art):
        return PROJECT_ROOT
    sahith = PROJECT_ROOT.parent / "artifact sahith"
    if _artifact_tree_has_models(sahith):
        return sahith
    parent_art = PROJECT_ROOT.parent / "artifacts"
    if _artifact_tree_has_models(parent_art):
        return PROJECT_ROOT.parent
    return PROJECT_ROOT


HANDOFF_DIR = _resolve_handoff_dir()


def _is_valid_gpcr_data_root(path: Path) -> bool:
    return (path / "Josh_Receptor_Features").is_dir()


def _is_manuscript_ready_gpcr_root(path: Path) -> bool:
    """Full training tree: pocket CSVs plus ML_code (shared_utilities, *_NEW.xlsx)."""
    return _is_valid_gpcr_data_root(path) and (path / "ML_code").is_dir()


def _apply_gpcr_data_root(root: Path) -> None:
    """Set GPCR_DATA_ROOT and MANUSCRIPT_ML_ROOT when layout is recognized."""
    root = root.resolve()
    os.environ["GPCR_DATA_ROOT"] = str(root)
    ml_code = root / "ML_code"
    if ml_code.is_dir():
        os.environ["MANUSCRIPT_ML_ROOT"] = str(ml_code.resolve())


def _find_gpcr_data_root(base: Path, subdir_hint: str = "") -> Optional[Path]:
    """Locate folder containing Josh_Receptor_Features under base (or nested one level)."""
    if subdir_hint:
        hinted = base / subdir_hint
        if _is_valid_gpcr_data_root(hinted):
            return hinted.resolve()
    if _is_valid_gpcr_data_root(base):
        return base.resolve()
    if not base.is_dir():
        return None
    for child in sorted(base.iterdir()):
        if child.is_dir() and _is_valid_gpcr_data_root(child):
            return child.resolve()
    return None


def _ensure_default_gpcr_data_root() -> None:
    """
    Default **GPCR_DATA_ROOT** to the folder that contains Josh_Receptor_Features (pocket CSVs).

    Prefer sibling **GUI_Folder** (training layout) when present; else project-local bundle.
    Cloud download runs later (after Streamlit import) via _bootstrap_cloud_gpcr_data().
    """
    existing = os.environ.get("GPCR_DATA_ROOT", "").strip()
    if existing and _is_valid_gpcr_data_root(Path(existing)):
        _apply_gpcr_data_root(Path(existing))
        return
    gui = PROJECT_ROOT.parent / "GUI_Folder"
    if _is_manuscript_ready_gpcr_root(gui):
        _apply_gpcr_data_root(gui)
        return
    if _is_manuscript_ready_gpcr_root(PROJECT_ROOT):
        _apply_gpcr_data_root(PROJECT_ROOT)
        return
    # Pocket CSVs only (no ML_code): do not set GPCR_DATA_ROOT here — cloud bootstrap
    # must download the full zip when DATA_ZIP_URL is configured.


_ensure_default_gpcr_data_root()

import streamlit as st


def _read_deploy_cfg(key: str, default: str = "") -> str:
    """Read Streamlit secret or environment variable."""
    val = os.environ.get(key, "").strip()
    if val:
        return val
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


def _google_drive_file_id(url: str) -> str:
    """Parse file id from common Google Drive share / uc URLs."""
    url = url.strip()
    for pattern in (
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
    ):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not parse Google Drive file id from: {url}")


def _google_drive_download_url(url_or_id: str) -> str:
    raw = url_or_id.strip()
    if raw.startswith("http"):
        file_id = _google_drive_file_id(raw)
    else:
        file_id = raw
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _looks_like_zip_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with open(path, "rb") as fh:
        head = fh.read(512)
    lower = head.lower()
    if b"<html" in lower or b"<!doctype" in lower:
        return False
    return head[:2] == b"PK" or path.stat().st_size > 50_000_000


def _stream_url_to_file(opener: urllib.request.OpenerDirector, download_url: str, dest: Path) -> None:
    with opener.open(download_url, timeout=3600) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _download_google_drive_with_cookies(download_url: str, dest_zip: Path) -> None:
    """urllib + cookies fallback (large Drive files need confirm token)."""
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]

    resp = opener.open(download_url, timeout=3600)
    peek = resp.read(65536)

    confirm: Optional[str] = None
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            confirm = cookie.value
            break

    text = peek.decode("utf-8", errors="ignore")
    needs_confirm = b"<html" in peek.lower() or "virus scan" in text.lower()
    if confirm is None and needs_confirm:
        for pattern in (
            r"confirm=([0-9A-Za-z_-]+)",
            r"uuid=([0-9a-fA-F-]{36})",
        ):
            match = re.search(pattern, text)
            if match:
                confirm = match.group(1)
                break
        if confirm is None:
            confirm = "t"

    if confirm:
        resp.close()
        sep = "&" if "?" in download_url else "?"
        final_url = f"{download_url}{sep}confirm={confirm}"
        _stream_url_to_file(opener, final_url, dest_zip)
        return

    with open(dest_zip, "wb") as out:
        out.write(peek)
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    resp.close()


def _download_google_drive(url: str, dest_zip: Path) -> None:
    """Download a Google Drive file (large zip: virus-scan confirm handled)."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    download_url = _google_drive_download_url(url)
    errors: list[str] = []

    try:
        import gdown

        gdown.download(download_url, str(dest_zip), quiet=False, fuzzy=True)
        if _looks_like_zip_file(dest_zip):
            return
        errors.append("gdown saved a non-zip response")
        dest_zip.unlink(missing_ok=True)
    except ImportError:
        errors.append("gdown not installed")
    except Exception as exc:
        errors.append(f"gdown: {exc}")
        dest_zip.unlink(missing_ok=True)

    try:
        _download_google_drive_with_cookies(download_url, dest_zip)
        if _looks_like_zip_file(dest_zip):
            return
        errors.append("cookie downloader saved HTML or a tiny file")
        dest_zip.unlink(missing_ok=True)
    except Exception as exc:
        errors.append(f"urllib: {exc}")
        dest_zip.unlink(missing_ok=True)

    raise RuntimeError(
        "Google Drive download failed. Share the zip as **Anyone with the link → Viewer**, "
        "then set secrets to either full DATA_ZIP_URL or DATA_DRIVE_FILE_ID only. "
        f"Details: {'; '.join(errors)}"
    )


def _extract_data_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


@st.cache_resource(show_spinner=False)
def _prepare_cloud_gpcr_data(zip_url: str, data_dir_name: str, subdir_hint: str) -> str:
    """
    Download/extract GPCR training data once per Streamlit container.
    Returns resolved GPCR_DATA_ROOT as a string, or "" on failure.
    """
    base = (PROJECT_ROOT / data_dir_name).resolve()
    marker = base / ".gpcr_data_ready"
    root = _find_gpcr_data_root(base, subdir_hint=subdir_hint)
    if root and marker.exists() and _is_manuscript_ready_gpcr_root(root):
        return str(root)
    if marker.exists():
        marker.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="gpcr_zip_") as tmp:
        zip_path = Path(tmp) / "gpcr_data.zip"
        _download_google_drive(zip_url, zip_path)
        if base.exists():
            shutil.rmtree(base)
        _extract_data_zip(zip_path, base)

    root = _find_gpcr_data_root(base, subdir_hint=subdir_hint)
    if not root:
        raise FileNotFoundError(
            f"Extracted data under {base} but no Josh_Receptor_Features folder found. "
            "Set DATA_EXTRACTED_SUBDIR in secrets to the folder inside the zip."
        )
    if not _is_manuscript_ready_gpcr_root(root):
        raise FileNotFoundError(
            f"Extracted data at {root} is missing ML_code. "
            "Use the full training zip (GPCRtryagain - Delete - Copy), not pocket CSVs only."
        )
    marker.write_text(str(root), encoding="utf-8")
    return str(root)


def _log_manuscript_debug(prefix: str = "post-bootstrap") -> None:
    try:
        from src.gpcr.manuscript_features import manuscript_debug_status

        dbg = manuscript_debug_status(PROJECT_ROOT)
        msg = " ".join(f"{k}={v}" for k, v in dbg.items())
        print(f"[manuscript-debug:{prefix}] {msg}")
    except Exception as exc:
        print(f"[manuscript-debug:{prefix}] status unavailable: {exc}")


def _bootstrap_cloud_gpcr_data() -> None:
    """
    On Streamlit Cloud: download DATA_ZIP_URL, extract, set GPCR_DATA_ROOT / MANUSCRIPT_ML_ROOT.
    Skipped only when the current root already has Josh_Receptor_Features and ML_code.
    """
    zip_url = _read_deploy_cfg("DATA_ZIP_URL")
    if not zip_url:
        zip_url = _read_deploy_cfg("DATA_DRIVE_FILE_ID")
    current = os.environ.get("GPCR_DATA_ROOT", "").strip()
    if current and _is_manuscript_ready_gpcr_root(Path(current)):
        _apply_gpcr_data_root(Path(current))
        _log_manuscript_debug("ready")
        return

    if not zip_url:
        if current and _is_valid_gpcr_data_root(Path(current)):
            _apply_gpcr_data_root(Path(current))
        _log_manuscript_debug("no-zip-url")
        return

    data_dir_name = _read_deploy_cfg("DATA_DIR", "runtime_data")
    subdir_hint = _read_deploy_cfg("DATA_EXTRACTED_SUBDIR", "")

    try:
        with st.status("Downloading GPCR data (first run may take several minutes)...", expanded=True):
            resolved = _prepare_cloud_gpcr_data(zip_url, data_dir_name, subdir_hint)
            _apply_gpcr_data_root(Path(resolved))
            st.write(f"Using **GPCR_DATA_ROOT**: `{resolved}`")
            ml = os.environ.get("MANUSCRIPT_ML_ROOT", "")
            if ml:
                st.write(f"Using **MANUSCRIPT_ML_ROOT**: `{ml}`")
            _log_manuscript_debug("after-download")
            if not os.environ.get("MANUSCRIPT_ML_ROOT", "").strip():
                st.warning(
                    "Downloaded data has no **ML_code** folder — manuscript predictions will be inaccurate."
                )
    except (urllib.error.URLError, zipfile.BadZipFile, RuntimeError, FileNotFoundError) as exc:
        st.error(f"Could not prepare GPCR data from DATA_ZIP_URL: {exc}")
        st.info(
            "Set Streamlit secrets: `DATA_ZIP_URL`, optional `DATA_DIR` (default runtime_data), "
            "optional `DATA_EXTRACTED_SUBDIR` if the zip has one top-level folder."
        )


_bootstrap_cloud_gpcr_data()
import pandas as pd
from rdkit import Chem

from src.gpcr.predict import (
    predict_single,
    predict_batch,
    load_predictor,
    get_available_receptors,
    get_gpcr_data_root,
)
from src.gpcr.receptor_names import receptor_display_options, resolve_receptor_folder
from src.gpcr.manuscript_bundle import manuscript_bundle_available, scan_manuscript_artifacts
from src.gpcr.manuscript_features import manuscript_debug_status
from src.gpcr.structure_view import py3dmol_available
from src.gpcr.docking import compute_receptor_grid_params, run_single_receptor_docking

try:
    import streamlit.components.v1 as st_components
except ImportError:
    st_components = None

# Data paths for demo tool
DATA_DIR = PROJECT_ROOT / "data"
RECEPTORS_FILE = DATA_DIR / "gpcr_class_a_receptors.txt"
DEMO_REFERENCE_FILE = DATA_DIR / "demo_reference.csv"


def extract_smiles_from_file(file_content: bytes, file_extension: str) -> Optional[str]:
    """
    Extract SMILES string from various molecular file formats.
    Supported formats: SDF, PDB, PDBQT, MOL, MOL2, CSV (first row only).
    """
    try:
        ext = file_extension.lower()
        if ext == ".sdf":
            from io import StringIO
            sdf_data = StringIO(file_content.decode("utf-8"))
            supplier = Chem.SDMolSupplier(sdf_data)
            for m in supplier:
                if m is not None:
                    return Chem.MolToSmiles(m, canonical=True)
        elif ext == ".mol":
            mol = Chem.MolFromMolBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdbqt":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".mol2":
            try:
                mol = Chem.MolFromMol2Block(file_content.decode("utf-8"))
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                pass
        elif ext == ".csv":
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_content))
            col = next((c for c in df.columns if c.lower() in ("smiles", "smi") or c == "SMILES"), None)
            if col and len(df) > 0:
                return str(df[col].iloc[0]).strip()
    except Exception:
        pass
    return None


# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="GPCR Class A Functional Activity Prediction",
    page_icon=None,
    layout="wide",
    menu_items={
        "About": "GPCR Class A Functional Activity Prediction GUI - Predicts Agonist/Antagonist/Inactive for receptor-ligand pairs.",
    },
)

# Inject custom CSS — solid fills (no gradients on chrome); 3D viewer unchanged
st.markdown("""
<style>
    :root {
        --bg: #eef6ff;
        --card: #ffffff;
        --ink: #0f172a;
        --brand: #38bdf8;
        --brand2: #60a5fa;
        --sidebar-bg: #cfe4f7;
        --hero-bg: #dbeafe;
    }
    .stApp { background: var(--bg); color: var(--ink); }
    .main .block-container,
    section.main .block-container {
        background: transparent !important;
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 1300px;
    }
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        color: #0f3554;
        width: 300px !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] h3 { color: #0f3554 !important; }
    .stButton > button {
        background: var(--brand2);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: none;
    }
    .stButton > button:hover {
        background: #3b82f6;
        color: #fff;
    }
    .hero {
        background: var(--hero-bg);
        color: #0f3554;
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        border: 1px solid #bfdbfe;
        box-shadow: none;
    }
    .hero h2 { color: #0b2a44; margin-bottom: 0.4rem; }
    .hero p { margin: 0; color: #1e3a5f; }
    .card {
        background: var(--card);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        box-shadow: none;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PREDICTOR (cached)
# ============================================================================

@st.cache_resource
def get_predictor(
    model_type: Optional[str] = None,
    evaluation_regime: Optional[str] = None,
    seed: int = 42,
):
    """Load predictor (manuscript regimes or legacy demo bundle)."""
    return load_predictor(
        HANDOFF_DIR,
        model_type=model_type,
        evaluation_regime=evaluation_regime,
        seed=seed,
    )

# ============================================================================
# PAGES
# ============================================================================

def render_home_page():
    """Render the home/dashboard page."""
    st.markdown(
        """
        <div class="hero">
          <h2>GPCR Class A Functional Activity</h2>
          <p>Manuscript-aligned ligand + receptor + interaction feature inference with a streamlined screening workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("GPCR Class A Functional Activity Prediction")
    st.caption(
        "Machine learning-based prediction of Agonist/Antagonist/Inactive activity for GPCR Class A receptor-ligand pairs."
    )

    st.sidebar.markdown("### Project Snapshot")
    st.sidebar.markdown(
        """
        - **Model focus:** GPCR Class A functional activity
        - **Classes:** Agonist, Antagonist, Inactive
        - **Features:** Ligand (PhysChem + ECFP) + Receptor (31) + Interaction (14)
        - **Models:** LightGBM, Random Forest, XGBoost (ensemble)
        - **Artifacts:** Auto-loads sibling **artifact sahith** or **./artifacts** (2103-dim descriptors)
        """
    )
    st.sidebar.info(
        "Receptor pocket CSVs default to **GUI_Folder/Josh_Receptor_Features** (set **GPCR_DATA_ROOT** to override). "
        "Trained models load from **artifact sahith** next to this project or from **./artifacts**."
    )

    st.markdown(
        """
        ## Why this app exists
        Drug discovery teams need to predict the functional activity of ligands binding to GPCR Class A receptors.
        This GUI provides a user-friendly interface for predicting whether a ligand acts as an **Agonist**, **Antagonist**, 
        or is **Inactive** for a given GPCR Class A receptor. The model uses machine learning approaches including
        LightGBM, Random Forest, and XGBoost to make predictions with uncertainty quantification.
        """
    )

    st.markdown(
        """
        ### Model highlights
        - **Multi-class classification:** Predicts Agonist (class 0), Antagonist (class 1), or Inactive (class 2)
        - **Feature engineering:** Combines ligand physicochemical properties, ECFP fingerprints, receptor features, and interaction terms
        - **Ensemble support:** Works with multiple model seeds for robust predictions
        - **Uncertainty quantification:** Provides error probabilities and confidence intervals
        - **Evaluation regimes:** Supports baseline, random stratified, scaffold split, and LORO (Leave-One-Receptor-Out) evaluation
        """
    )

    st.divider()

    st.markdown("## Quick start")

    st.info(
        "**Ready to predict!** Use **GPCR Ligand Functional Activity Prediction** for single or batch predictions."
    )

    st.markdown(
        """
        ---
        ### Navigation
        - **Home:** This overview
        - **Documentation:** Setup, model details, and usage
        - **GPCR Ligand Functional Activity Prediction:** Run predictions (receptor + ligand)
        """
    )


def render_documentation_page():
    """Render the documentation page."""
    st.title("Documentation & Runbook")
    st.caption("Reference material for the GPCR Class A Functional Activity Prediction GUI.")

    st.markdown(
        """
        ## Purpose
        This application provides a Streamlit interface for predicting GPCR Class A receptor-ligand functional activity.
        It supports single predictions (receptor name + ligand SMILES/structure file) and batch CSV processing.
        """
    )

    st.markdown(
        """
        ## Repository structure
        ```
        .
        ├── streamlit_app.py       # Main application
        ├── requirements.txt      # Dependencies
        ├── src/gpcr/             # Prediction module
        │   ├── predict.py        # predict_single, predict_batch, load_predictor
        │   └── cli.py           # Command-line interface
        └── artifacts/            # Or use sibling artifact sahith/ with demo_rf/, …
            ├── model_seed0.pkl (or .joblib)
            ├── model_seed1.pkl
            ├── ...
            ├── feature_config.json
            └── threshold.json (optional)
        ```
        """
    )

    st.markdown(
        """
        ## Local setup
        1. Create and activate a virtual environment (conda, venv, or poetry).
        2. Install dependencies: `pip install -r requirements.txt`.
        3. Receptor data: keep **GUI_Folder** beside **GPCR-FAP-main** (auto-detects **Josh_Receptor_Features**), or set **`GPCR_DATA_ROOT`**.
        4. **Trained models:** place them under **`./artifacts`** *or* a sibling folder **`artifact sahith`** (same layout: `demo_rf/`, `demo_lightgbm/`, …); the app picks those up automatically (see below).
        5. Launch the app: `streamlit run streamlit_app.py`.
        6. Streamlit will open at `http://localhost:8501`. Use the sidebar to switch between pages.
        """
    )

    st.markdown(
        """
        ## Model overview
        - **Classes:** Agonist (0), Antagonist (1), Inactive (2)
        - **Manuscript mode** (`artifacts/manuscript/`):
          - Ligand: RDKit + Mordred descriptors from enriched training CSVs (~1,636 columns)
          - Receptor: 31 pocket features from `Josh_Receptor_Features`
          - Interaction: 14 ligand × receptor terms
          - **Total: ~1,681 features** (see `manifest.json`)
          - Trained on ~40,611 pairs; independent-ligand models fit on dev80 (80% canonical SMILES split)
        - **Demo / legacy mode** (`demo_*` bundles):
          - Ligand: 10 RDKit + 2048-bit Morgan ECFP4 = 2,058
          - Receptor + interaction: 31 + 14 → **2,103 features** (small demo training set)
        - **Models:** Ensemble of LightGBM, Random Forest, or XGBoost models
        - **Evaluation:** Baseline, Random Stratified, Scaffold Split, LORO
        """
    )

    st.markdown(
        """
        ## Adding your ML artifacts
        
        Place your trained model files under **`./artifacts`** (inside this repo) **or** under a sibling folder
        **`artifact sahith`** (flat layout: `demo_rf/model_seed0.pkl`, …). The GUI checks `./artifacts` first, then **`artifact sahith`**, then **`../artifacts`**.
        
        Example layout under `artifacts/`:
        - `model_seed0.pkl` (or `.joblib`)
        - `model_seed1.pkl`
        - `model_seed2.pkl`
        - ... (as many seeds as you have)
        
        Optionally create:
        - `feature_config.json`: Feature configuration (class names, etc.)
        - `threshold.json`: Classification thresholds (if applicable)
        """
    )

    st.markdown(
        """
        ## Uncertainty Quantification
        The model provides uncertainty estimates for each prediction:
        - **Standard Error:** Calculated from the variance across the ensemble models (std_dev / √n).
        - **95% Confidence Interval:** Probability ± 2×SE, providing a range within which the true probability likely falls.
        - **Display:** Single predictions show probability distributions and confidence intervals. Batch CSV outputs include columns for standard error and CI bounds.
        """
    )

    st.markdown(
        """
        ## CLI usage
        From the project folder:
        ```bash
        python -m src.gpcr.cli --receptor "ADRB2" --ligand "CCO" --output out.csv
        python -m src.gpcr.cli --input example_inputs.csv --output out.csv
        ```
        Output columns: receptor, ligand_smiles, canonical_smiles, predicted_class, class_id, prob_agonist, prob_antagonist, prob_inactive, prob_std_error, error.
        """
    )

    st.success("Questions? Refer to the ML GPCR Class A Functional Activity Manuscript for model details.")


def _load_receptor_select_options():
    """
    (folder_name, display_label) for the receptor dropdown.
    Uses Josh_Receptor_Features folder names; labels include gene aliases (e.g. beta2 (ADRB2)).
    """
    options = receptor_display_options(get_gpcr_data_root())
    if options:
        return options
    if not RECEPTORS_FILE.exists():
        return []
    try:
        lines = RECEPTORS_FILE.read_text(encoding="utf-8").strip().splitlines()
        return [(s.strip(), s.strip()) for s in lines if s.strip()]
    except Exception:
        return []


def _load_demo_reference():
    """Load demo reference data (receptor, ligand, experimental_class) for comparison table."""
    if not DEMO_REFERENCE_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(DEMO_REFERENCE_FILE, encoding="utf-8")
    except pd.errors.ParserError:
        df = pd.read_csv(DEMO_REFERENCE_FILE, encoding="utf-8", engine="python", on_bad_lines="skip")
    return df


def render_demo_prediction_page():
    """Render the Demo Prediction Tool page: predicted vs experimental comparison table."""
    st.title("Demo Prediction Tool")
    st.caption(
        "Compare model predictions to experimental values (Agonist / Antagonist / Inactive) "
        "using Random Forest, LightGBM, XGBoost, or Ensemble."
    )

    ref_df = _load_demo_reference()
    if ref_df.empty or "smiles" not in ref_df.columns or "experimental_class" not in ref_df.columns:
        st.warning(
            "Demo reference data not found or missing columns. Add data/demo_reference.csv with columns: "
            "receptor, name, smiles, experimental_class (Agonist/Antagonist/Inactive)."
        )
        return

    st.sidebar.markdown("### Demo settings")
    model_type_label = st.sidebar.selectbox(
        "Model",
        options=["Random Forest", "LightGBM", "XGBoost", "Ensemble"],
        index=0,
        key="demo_model",
    )
    model_type_map = {"Random Forest": "rf", "LightGBM": "lightgbm", "XGBoost": "xgboost", "Ensemble": "ensemble"}
    model_type = model_type_map[model_type_label]

    try:
        predictor = get_predictor(model_type)
    except Exception as e:
        st.error(f"Could not load {model_type_label} model: {e}")
        st.info(
            "Ensure **./artifacts** or sibling **artifact sahith** contains **demo_rf**, **demo_lightgbm**, "
            "**demo_xgboost**, and/or **demo_ensemble** with **model_seed*.pkl** and **feature_config.json**."
        )
        return

    # Run predictions for all reference rows
    pairs = [(str(row["receptor"]), str(row["smiles"])) for _, row in ref_df.iterrows()]
    with st.spinner(f"Running {model_type_label} on {len(ref_df)} reference compounds..."):
        results = predict_batch(pairs, predictor=predictor)

    # Build comparison table: experimental vs predicted
    out = ref_df[["receptor", "name", "smiles", "experimental_class"]].copy()
    out["predicted_class"] = [r.predicted_class for r in results]
    out["P(Agonist)"] = [round(r.prob_agonist, 4) for r in results]
    out["P(Antagonist)"] = [round(r.prob_antagonist, 4) for r in results]
    out["P(Inactive)"] = [round(r.prob_inactive, 4) for r in results]
    out["match"] = [
        "✓" if str(row["experimental_class"]).strip().lower() == str(row["predicted_class"]).strip().lower() else "✗"
        for _, row in out.iterrows()
    ]
    out = out.rename(columns={"match": "Match"})

    st.markdown(f"**Model:** {model_type_label} · **Reference compounds:** {len(ref_df)}")

    # Summary metrics
    n_match = out["Match"].eq("✓").sum()
    accuracy = n_match / len(out) * 100 if len(out) else 0
    st.metric("Agreement with experiment", f"{n_match} / {len(out)} ({accuracy:.1f}%)")

    st.subheader("Predicted vs experimental")
    st.dataframe(
        out[
            [
                "receptor",
                "name",
                "experimental_class",
                "predicted_class",
                "P(Agonist)",
                "P(Antagonist)",
                "P(Inactive)",
                "Match",
            ]
        ],
        use_container_width=True,
        height=400,
    )
    st.download_button(
        "Download comparison (CSV)",
        out.to_csv(index=False),
        f"demo_predicted_vs_experimental_{model_type}.csv",
        "text/csv",
        key="demo_download",
    )


def render_gpcr_prediction_page():
    """Render the GPCR Ligand Functional Activity Prediction page."""
    st.markdown(
        """
        <div class="hero">
          <h2>Screen Ligands Against Class A GPCRs</h2>
          <p>Select a receptor, provide SMILES or upload structural files, and get Agonist/Antagonist/Inactive probabilities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("GPCR Ligand Functional Activity Prediction")
    st.markdown(
        """
        Predict GPCR Class A receptor-ligand functional activity. Choose a model, select a receptor and provide a ligand (SMILES or structure file),
        or upload a CSV file for batch processing. The model outputs probabilities for Agonist, Antagonist, and Inactive classes.
        
        **Input modes:** Single receptor-ligand pair | Batch (CSV with receptor and ligand columns)
        """
    )

    _artifact_scan = scan_manuscript_artifacts(HANDOFF_DIR)
    _has_manuscript = manuscript_bundle_available(HANDOFF_DIR)
    _ms_regimes = _artifact_scan.get("regimes") or {}
    _ms_seeds = _artifact_scan.get("seeds") or [42]

    st.markdown("#### Model source (manuscript vs demo)")
    _regime_labels = {
        "independent_ligand": "Independent ligand test (paper: dev80-trained)",
        "scaffold": "Scaffold split",
        "loro": "Leave-one-receptor-out (LORO)",
    }
    regime_map: dict = {"Demo bundle (2103 features, legacy)": None}
    regime_options: list = []

    if _has_manuscript:
        for key, label in _regime_labels.items():
            if _ms_regimes.get(key):
                regime_options.append(label)
                regime_map[label] = key
        regime_options.append("Demo bundle (2103 features, legacy)")
        regime_map["Demo bundle (2103 features, legacy)"] = None
        if not regime_options:
            regime_options = ["Demo bundle (2103 features, legacy)"]
        regime_index = 0
        if "ensemble" in (_ms_regimes.get("independent_ligand") or {}):
            pass  # default stays independent ligand
    else:
        regime_options = ["Demo bundle (2103 features, legacy)"]
        regime_index = 0
        st.info(
            "Manuscript models not deployed yet. Using legacy demo bundle (~2103 features). "
            "Run `scripts/export_manuscript_models.py` and add `artifacts/manuscript/`."
        )

    regime_label = st.selectbox(
        "Evaluation regime",
        regime_options,
        index=regime_index,
        key="gpcr_eval_regime",
        help="Only regimes with exported artifacts are listed.",
    )
    evaluation_regime = regime_map[regime_label]

    _model_labels = {
        "rf": "Random Forest",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "ensemble": "Ensemble (stacking)",
    }
    model_type_map = {v: k for k, v in _model_labels.items()}

    if evaluation_regime and _ms_regimes.get(evaluation_regime):
        available_models = list(_ms_regimes[evaluation_regime].keys())
        model_options = [_model_labels[m] for m in ("ensemble", "rf", "lightgbm", "xgboost") if m in available_models]
        default_model_ix = 0 if "Ensemble (stacking)" in model_options else 0
    else:
        model_options = list(_model_labels.values())
        default_model_ix = 0

    st.markdown("#### Select model")
    model_type_label = st.selectbox(
        "Model type",
        model_options,
        index=default_model_ix,
        key="gpcr_pred_model",
        help="Only models present under artifacts/manuscript/ are shown.",
    )
    model_type = model_type_map[model_type_label]

    if evaluation_regime == "loro" and model_type == "ensemble":
        st.warning("Manuscript stacking ensemble was evaluated on the **independent ligand** split, not LORO.")

    seed = 42
    if evaluation_regime and evaluation_regime in _ms_regimes:
        regime_seeds = sorted(
            set(_ms_regimes[evaluation_regime].get(model_type, [])) & set(_ms_seeds)
        ) or [42]
        seed = st.selectbox(
            "Random seed",
            regime_seeds,
            index=0,
            key="gpcr_model_seed",
            help="Seeds exported to artifacts/manuscript/ (default: 42).",
        )

    if evaluation_regime and not _has_manuscript:
        st.error(
            "Manuscript models not found. Run `scripts/export_manuscript_models.py` on your training PC, "
            "then copy `artifacts/manuscript/` into this project. See `docs/MANUSCRIPT_STREAMLIT_SETUP.md`."
        )
        return

    try:
        predictor = get_predictor(model_type, evaluation_regime=evaluation_regime, seed=seed)
    except Exception as e:
        st.error(f"Could not load {model_type_label} model: {e}")
        if evaluation_regime:
            st.info(
                "Export models first: `docs/MANUSCRIPT_STREAMLIT_SETUP.md` and `scripts/export_manuscript_models.py`.\n"
                f"Expected: `artifacts/manuscript/{evaluation_regime}/{model_type}/model_seed{seed}.pkl`"
            )
        else:
            st.info(
                "Ensure **./artifacts** contains demo subfolders (**demo_rf/**, etc.) with valid **model_seed0.pkl** "
                "(placeholder tiny .pkl files are ignored)."
            )
        return

    st.sidebar.markdown("### Model Info")
    _mode = getattr(predictor, "feature_mode", "demo_2103")
    st.sidebar.info(
        f"**Regime:** {regime_label}\n\n"
        f"**Model:** {model_type_label}\n\n"
        f"**Seed:** {seed}\n\n"
        f"**Feature mode:** {_mode}\n\n"
        f"**Loaded:** {len(predictor.models)} estimator(s)\n\n"
        f"**Classes:** {', '.join(predictor.class_names)}"
    )
    _gdata = os.environ.get("GPCR_DATA_ROOT", "").strip()
    _efd = getattr(predictor, "expected_feature_dim", None)
    st.caption(
        f"**ML bundle:** `{HANDOFF_DIR}` · **Pocket data:** `{_gdata or 'default'}` · "
        f"**Features:** {_efd if _efd is not None else '—'} dims"
    )
    if _mode == "manuscript":
        st.caption("Install **mordred** for best ligand descriptor parity with enriched training CSVs.")
        dbg = manuscript_debug_status(HANDOFF_DIR)
        # Keep a plain stdout line so Streamlit Cloud logs capture this state.
        print(
            "[manuscript-debug] "
            f"gpcr_data_root={dbg['gpcr_data_root']} "
            f"ml_root={dbg['ml_root']} "
            f"ml_root_exists={dbg['ml_root_exists']} "
            f"shared_utilities_imported={dbg['shared_utilities_imported']} "
            f"manifest_exists={dbg['manifest_exists']} "
            f"manifest_feature_count={dbg['manifest_feature_count']} "
            f"ligand_lookup_exists={dbg['ligand_lookup_exists']} "
            f"ligand_lookup_entries={dbg['ligand_lookup_entries']} "
            f"ligand_lookup_source={dbg['ligand_lookup_source']}"
        )
        with st.sidebar.expander("Manuscript feature diagnostics", expanded=False):
            st.write(f"GPCR data root: `{dbg['gpcr_data_root']}`")
            st.write(f"MANUSCRIPT_ML_ROOT: `{dbg['ml_root'] or '(not set)'}`")
            st.write(f"ML root exists: `{dbg['ml_root_exists']}`")
            st.write(f"shared_utilities import: `{dbg['shared_utilities_imported']}`")
            st.write(f"manifest.json exists: `{dbg['manifest_exists']}`")
            st.write(f"manifest feature count: `{dbg['manifest_feature_count']}`")
            st.write(f"ligand lookup exists: `{dbg['ligand_lookup_exists']}`")
            st.write(f"ligand lookup entries: `{dbg['ligand_lookup_entries']}`")
            st.write(f"ligand lookup source: `{dbg['ligand_lookup_source']}`")

    st.divider()

    input_mode = st.radio(
        "Input mode",
        ["Single receptor-ligand pair", "Batch (CSV)"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "Single receptor-ligand pair":
        receptor_options = _load_receptor_select_options()
        if not receptor_options:
            st.warning(
                "No receptors found under **Josh_Receptor_Features**. "
                "Set **GPCR_DATA_ROOT** to the folder that contains **Josh_Receptor_Features**, "
                "or place **GUI_Folder** next to this project (see README)."
            )
        folder_names = [f for f, _ in receptor_options]
        display_labels = [lbl for _, lbl in receptor_options]
        receptor_pick = st.selectbox(
            "GPCR Class A Receptor",
            options=["Select receptor..."] + display_labels,
            key="receptor_input",
            help="Pocket folder name with optional gene symbol (e.g. beta2 = ADRB2).",
        )
        if receptor_pick and receptor_pick != "Select receptor...":
            idx = display_labels.index(receptor_pick)
            receptor_selected = folder_names[idx]
        else:
            receptor_selected = ""

        ligand_input = st.text_input(
            "Ligand SMILES (or upload a structure file below)",
            placeholder="e.g. CCO, c1ccccc1",
            key="ligand_input",
        )
        
        st.markdown("**Or upload a ligand structure file:**")
        structure_file = st.file_uploader(
            "Upload ligand structure file",
            type=["sdf", "mol", "pdb", "pdbqt", "mol2", "csv"],
            key="structure_upload",
            help="Supported: SDF, MOL, PDB, PDBQT, MOL2, CSV (first smiles row).",
        )
        
        ligand_to_use = None
        if structure_file:
            content = structure_file.read()
            ext = os.path.splitext(structure_file.name)[1]
            extracted = extract_smiles_from_file(content, ext)
            if extracted:
                ligand_to_use = extracted
                st.success(f"Extracted SMILES from {structure_file.name}")
            else:
                st.error(f"Could not extract SMILES from {ext.upper()} file. Try SMILES input instead.")
        elif ligand_input and ligand_input.strip():
            ligand_to_use = ligand_input.strip()

        st.caption(
            "Workflow: run FAP prediction first, then click the docking button below to generate and visualize "
            "a top docking pose."
        )

        def _render_single_prediction_from_session(pred: dict) -> None:
            """Render persisted single-prediction outputs so docking reruns do not reset the panel."""
            st.success("Valid input")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Class", str(pred["predicted_class"]))
            with col2:
                st.metric("Receptor", str(pred["receptor"]))
            with col3:
                st.metric("Class ID", int(pred["class_id"]))

            st.subheader("Probability Distributions")
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            with prob_col1:
                st.metric("P(Agonist)", f"{float(pred['prob_agonist']):.4f}")
            with prob_col2:
                st.metric("P(Antagonist)", f"{float(pred['prob_antagonist']):.4f}")
            with prob_col3:
                st.metric("P(Inactive)", f"{float(pred['prob_inactive']):.4f}")

            import plotly.graph_objects as go
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Agonist", "Antagonist", "Inactive"],
                        y=[float(pred["prob_agonist"]), float(pred["prob_antagonist"]), float(pred["prob_inactive"])],
                        marker_color=["#7E57C2", "#673AB7", "#512DA8"],
                        text=[
                            f"{float(pred['prob_agonist']):.3f}",
                            f"{float(pred['prob_antagonist']):.3f}",
                            f"{float(pred['prob_inactive']):.3f}",
                        ],
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                title="Class Probability Distribution",
                xaxis_title="Class",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            std_err = pred.get("prob_std_error")
            if std_err is not None:
                std_err = float(std_err)
                st.markdown("#### Uncertainty Analysis")
                err_col1, err_col2, err_col3 = st.columns(3)
                with err_col1:
                    st.metric("Standard Error", f"± {std_err * 100:.2f}%")
                prob_max = max(float(pred["prob_agonist"]), float(pred["prob_antagonist"]), float(pred["prob_inactive"]))
                ci_lower = max(0.0, prob_max - 2 * std_err)
                ci_upper = min(1.0, prob_max + 2 * std_err)
                with err_col2:
                    st.metric("95% CI Lower", f"{ci_lower:.4f}")
                with err_col3:
                    st.metric("95% CI Upper", f"{ci_upper:.4f}")
                st.info(
                    f"**Prediction Range:** Highest probability = {prob_max:.4f} ± {std_err:.4f} "
                    f"(95% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )

        if st.button("Predict", type="primary", key="btn_single"):
            if receptor_selected and ligand_to_use:
                result = predict_single(
                    receptor_selected,
                    ligand_to_use,
                    predictor=predictor,
                )
                if result.is_valid:
                    st.success("Valid input")
                    st.session_state["last_single_prediction"] = {
                        "receptor": result.receptor,
                        "canonical_smiles": result.canonical_smiles,
                        "predicted_class": result.predicted_class,
                        "class_id": int(result.class_id),
                        "prob_agonist": float(result.prob_agonist),
                        "prob_antagonist": float(result.prob_antagonist),
                        "prob_inactive": float(result.prob_inactive),
                        "prob_std_error": float(result.prob_std_error) if result.prob_std_error is not None else None,
                    }
                    st.session_state.pop("last_docking_result", None)
                else:
                    st.session_state.pop("last_single_prediction", None)
                    st.session_state.pop("last_docking_result", None)
                    st.error(result.error)
            else:
                st.warning("Please select a GPCR Class A receptor and provide ligand SMILES or upload a structure file.")

        last_pred = st.session_state.get("last_single_prediction")
        if last_pred:
            _render_single_prediction_from_session(last_pred)
            st.divider()
            st.subheader("Docking + receptor-ligand visualization")
            st.caption(
                "**Recommended** grid center and size come from this receptor's `<id>_ligand_only.pdb` "
                "(centroid and padded extent, each axis clipped to 15–20 Å). You can override these in the panel below. "
                "Pose generation uses SMINA with defaults: exhaustiveness=64, num_modes=10, seed=42."
            )

            dock_folder = resolve_receptor_folder(
                str(last_pred["receptor"]),
                get_gpcr_data_root(),
            ) or str(last_pred["receptor"])
            rec_center, rec_size, grid_help = compute_receptor_grid_params(dock_folder)

            with st.expander("Docking search box (recommended vs. custom)", expanded=False):
                if rec_center is None or rec_size is None:
                    st.info(grid_help)
                else:
                    st.caption(
                        "These defaults follow the co-crystal ligand geometry. Edited values are passed to SMINA as "
                        "`--center_*` and `--size_*`."
                    )
                    cx, cy, cz = st.columns(3)
                    with cx:
                        st.number_input(
                            "Center X (Å)",
                            format="%.3f",
                            step=0.1,
                            value=float(rec_center[0]),
                            key=f"dock_cx_{dock_folder}",
                        )
                    with cy:
                        st.number_input(
                            "Center Y (Å)",
                            format="%.3f",
                            step=0.1,
                            value=float(rec_center[1]),
                            key=f"dock_cy_{dock_folder}",
                        )
                    with cz:
                        st.number_input(
                            "Center Z (Å)",
                            format="%.3f",
                            step=0.1,
                            value=float(rec_center[2]),
                            key=f"dock_cz_{dock_folder}",
                        )
                    sx, sy, sz = st.columns(3)
                    with sx:
                        st.number_input(
                            "Size X (Å)",
                            format="%.3f",
                            step=0.5,
                            min_value=1.0,
                            max_value=80.0,
                            value=float(rec_size[0]),
                            key=f"dock_sx_{dock_folder}",
                        )
                    with sy:
                        st.number_input(
                            "Size Y (Å)",
                            format="%.3f",
                            step=0.5,
                            min_value=1.0,
                            max_value=80.0,
                            value=float(rec_size[1]),
                            key=f"dock_sy_{dock_folder}",
                        )
                    with sz:
                        st.number_input(
                            "Size Z (Å)",
                            format="%.3f",
                            step=0.5,
                            min_value=1.0,
                            max_value=80.0,
                            value=float(rec_size[2]),
                            key=f"dock_sz_{dock_folder}",
                        )
                    if st.button("Reset box to recommended", key=f"dock_reset_grid_{dock_folder}"):
                        st.session_state[f"dock_cx_{dock_folder}"] = float(rec_center[0])
                        st.session_state[f"dock_cy_{dock_folder}"] = float(rec_center[1])
                        st.session_state[f"dock_cz_{dock_folder}"] = float(rec_center[2])
                        st.session_state[f"dock_sx_{dock_folder}"] = float(rec_size[0])
                        st.session_state[f"dock_sy_{dock_folder}"] = float(rec_size[1])
                        st.session_state[f"dock_sz_{dock_folder}"] = float(rec_size[2])
                        st.rerun()

            if st.button("Run docking and show top pose", key="btn_single_docking", type="secondary"):
                with st.spinner("Running docking..."):
                    grid_kw = {}
                    if rec_center is not None and rec_size is not None:
                        grid_kw["grid_center"] = (
                            float(st.session_state[f"dock_cx_{dock_folder}"]),
                            float(st.session_state[f"dock_cy_{dock_folder}"]),
                            float(st.session_state[f"dock_cz_{dock_folder}"]),
                        )
                        grid_kw["grid_size"] = (
                            float(st.session_state[f"dock_sx_{dock_folder}"]),
                            float(st.session_state[f"dock_sy_{dock_folder}"]),
                            float(st.session_state[f"dock_sz_{dock_folder}"]),
                        )
                    dock_res = run_single_receptor_docking(
                        receptor_folder=dock_folder,
                        canonical_smiles=str(last_pred["canonical_smiles"]),
                        **grid_kw,
                    )
                st.session_state["last_docking_result"] = dock_res.__dict__

            dock_result = st.session_state.get("last_docking_result")
            if dock_result and dock_result.get("receptor_name") == str(last_pred["receptor"]):
                if dock_result.get("ok"):
                    if not py3dmol_available():
                        st.info("Install **py3Dmol** to render the docked complex: `pip install py3Dmol`")
                    elif st_components is None:
                        st.warning("streamlit.components is unavailable; cannot embed the docked 3D viewer.")
                    elif dock_result.get("html"):
                        st.markdown(
                            "**3D viewer:** white receptor cartoon, green ligand sticks; the **three closest residues** "
                            "to the docked ligand are emphasized with **dashed cylinders** from the best contact atom on "
                            "each residue to the ligand (MBind-style; teal ≈ polar, green ≈ aromatic C–C, slate ≈ other)."
                        )
                        st_components.html(str(dock_result["html"]), height=560, scrolling=False)
                        st.caption(
                            "**3D viewer:** drag to rotate the scene • **Ctrl+drag** or **middle mouse** drag to pan "
                            "(move left/right and up/down) • scroll to zoom."
                        )
                        score = dock_result.get("score_kcal_mol")
                        st.markdown(
                            f"**Top Pose Docking Score (kcal/mol):** "
                            f"{float(score):.3f}" if score is not None else "**Top Pose Docking Score (kcal/mol):** N/A"
                        )
                        gc = dock_result.get("center")
                        gs = dock_result.get("size")
                        if gc and gs and len(gc) == 3 and len(gs) == 3:
                            st.caption(
                                f"**Search box used:** center ({float(gc[0]):.3f}, {float(gc[1]):.3f}, {float(gc[2]):.3f}) Å · "
                                f"size ({float(gs[0]):.3f}, {float(gs[1]):.3f}, {float(gs[2]):.3f}) Å"
                            )
                        contacts = dock_result.get("contact_summary")
                        if contacts:
                            st.markdown("**Closest residue contacts (≤5 Å, best heavy-atom pair per residue):**")
                            for line in contacts:
                                st.markdown(f"- {line}")
                    else:
                        st.warning("Docking succeeded, but the 3D viewer payload was empty.")
                else:
                    st.error(str(dock_result.get("message", "Docking failed.")))

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="csv_upload",
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            receptor_col = next(
                (c for c in df.columns if c.lower() in ("receptor", "receptor_name", "gpcr")),
                None
            )
            ligand_col = next(
                (c for c in df.columns if c.lower() in ("ligand", "smiles", "canonical_smiles", "smi")),
                None
            )
            if receptor_col is None:
                st.error("CSV must have a 'receptor' column.")
                st.info(f"Available columns: {', '.join(df.columns)}")
            elif ligand_col is None:
                st.error("CSV must have a 'ligand' or 'smiles' column.")
                st.info(f"Available columns: {', '.join(df.columns)}")
            else:
                if st.button("Predict batch", type="primary", key="btn_batch"):
                    pairs = list(zip(df[receptor_col].astype(str), df[ligand_col].astype(str)))
                    results = predict_batch(pairs, predictor=predictor)
                    
                    df_out = df.copy()
                    df_out["predicted_class"] = [r.predicted_class for r in results]
                    df_out["class_id"] = [r.class_id for r in results]
                    df_out["prob_agonist"] = [r.prob_agonist for r in results]
                    df_out["prob_antagonist"] = [r.prob_antagonist for r in results]
                    df_out["prob_inactive"] = [r.prob_inactive for r in results]
                    df_out["prob_std_error"] = [
                        f"{r.prob_std_error:.6f}" if r.prob_std_error is not None else ""
                        for r in results
                    ]
                    df_out["prob_std_error_pct"] = [
                        f"{r.prob_std_error * 100:.2f}%" if r.prob_std_error is not None else ""
                        for r in results]
                    df_out["canonical_smiles"] = [r.canonical_smiles for r in results]
                    df_out["error"] = [r.error for r in results]

                    st.subheader("Results")
                    st.dataframe(df_out, use_container_width=True)

                    st.subheader("Download results")
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False),
                        "gpcr_predictions.csv",
                        "text/csv",
                        key="download_csv",
                    )
        else:
            st.info("Upload a CSV file with 'receptor' and 'ligand' (or 'smiles') columns to run batch predictions.")

    st.divider()
    st.caption(
        "GPCR Class A Functional Activity Prediction. Multi-class classification: Agonist/Antagonist/Inactive."
    )


# ============================================================================
# MAIN - NAVIGATION
# ============================================================================

def main():
    """Main app entry point with navigation."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("")

    if st.sidebar.button("Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = "Home"

    if st.sidebar.button("Documentation", use_container_width=True, key="nav_docs"):
        st.session_state.current_page = "Documentation"

    if st.sidebar.button("GPCR Ligand Functional Activity Prediction", use_container_width=True, key="nav_prediction"):
        st.session_state.current_page = "GPCR Ligand Functional Activity Prediction"

    st.sidebar.markdown("---")

    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Documentation":
        render_documentation_page()
    elif st.session_state.current_page == "GPCR Ligand Functional Activity Prediction":
        render_gpcr_prediction_page()


if __name__ == "__main__":
    main()

