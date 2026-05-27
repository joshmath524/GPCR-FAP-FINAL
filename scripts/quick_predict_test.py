import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(r"C:\Users\joshmatchem\Music\GPCR-FAP-main (2)\GPCRtryagain - Delete - Copy")
os.environ["GPCR_DATA_ROOT"] = str(DATA)
os.environ["MANUSCRIPT_ML_ROOT"] = str(DATA / "ML_code")

from src.gpcr.predict import load_predictor, predict_single

pred = load_predictor(
    str(ROOT), model_type="rf", evaluation_regime="independent_ligand", seed=42
)
cases = [
    ("beta2", "CC(C)NCC(O)c1cc(O)c(O)cc1O", "salbutamol"),
    ("ADRB2", "CC(C)NCC(O)c1cc(O)c(O)cc1O", "salbutamol ADRB2"),
]
for rec, smi, label in cases:
    r = predict_single(rec, smi, predictor=pred)
    print(
        label,
        rec,
        r.predicted_class,
        f"agonist={r.prob_agonist:.3f}",
        f"antag={r.prob_antagonist:.3f}",
        f"inactive={r.prob_inactive:.3f}",
        r.error or "",
    )
