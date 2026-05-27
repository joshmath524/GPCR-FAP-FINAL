# Re-export stacking ensemble with Code S21-style OOF meta training (seed 42).
$ErrorActionPreference = "Stop"
$Root = "C:\Users\joshmatchem\Music\GPCR-FAP-main (2)"
$Data = Join-Path $Root "GPCRtryagain - Delete - Copy"
$Proj = Join-Path $Root "GPCR-FAP-main"

$env:MANUSCRIPT_DATA_ROOT = $Data
$env:MANUSCRIPT_ML_ROOT = Join-Path $Data "ML_code"
$env:MANUSCRIPT_SPLIT_DIR = Join-Path $Data "MLcodes\pan_gpcr_results\splits"
$env:GPCR_EXPORT_ROOT = $Proj

Set-Location $Proj
# Resumes from artifacts/manuscript/.../ensemble/checkpoint_seed42.joblib after each OOF fold.
# Use --fresh to discard checkpoint and start over.
py -3 scripts\export_manuscript_models.py --regime ensemble --model ensemble --seeds 42
