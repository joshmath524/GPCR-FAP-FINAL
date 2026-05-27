# Export manuscript models for Streamlit (seed 42 only).
# RF is skipped if model_seed42.pkl already exists. Always builds ensemble.
# Usage: .\scripts\run_export_manuscript_seed42.ps1

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Data = "C:\Users\joshmatchem\Music\GPCR-FAP-main (2)\GPCRtryagain - Delete - Copy"

$env:MANUSCRIPT_DATA_ROOT = $Data
$env:MANUSCRIPT_ML_ROOT    = Join-Path $Data "ML_code"
$env:MANUSCRIPT_SPLIT_DIR  = Join-Path $Data "MLcodes\pan_gpcr_results\splits"
$env:GPCR_EXPORT_ROOT       = $Root
$env:GPCR_DATA_ROOT         = $Data
$env:PYTHONUNBUFFERED       = "1"

Set-Location $Root

Write-Host "=== RF independent ligand (skip if exists) ==="
py -3 scripts\export_manuscript_models.py --regime independent_ligand --model rf --seeds 42 --skip-existing

Write-Host "=== Stacking ensemble (seed 42) ==="
py -3 scripts\export_manuscript_models.py --regime ensemble --model ensemble --seeds 42
