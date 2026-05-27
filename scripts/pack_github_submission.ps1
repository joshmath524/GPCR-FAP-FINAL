# Recreates FINAL_GITHUB_SUBMISSION/ — a clean mirror of this repo for GitHub upload (no .git, no caches).
# Run:  powershell -ExecutionPolicy Bypass -File scripts\pack_github_submission.ps1

$ErrorActionPreference = "Stop"
$src = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $src "streamlit_app.py"))) {
    Write-Error "Run from project root (expected streamlit_app.py next to scripts/)."
}
$dst = Join-Path $src "FINAL_GITHUB_SUBMISSION"
Write-Host "Source: $src"
Write-Host "Dest:   $dst"
if (Test-Path $dst) {
    Remove-Item $dst -Recurse -Force
}
New-Item -ItemType Directory -Path $dst -Force | Out-Null
$excludeDirs = @(
    ".git", "__pycache__", ".venv", "venv", "FINAL_GITHUB_SUBMISSION",
    "docking_results", "tmp_dock_test", ".streamlit", ".ruff_cache",
    ".pytest_cache", ".mypy_cache"
)
$args = @($src, $dst, "/E")
foreach ($d in $excludeDirs) { $args += "/XD"; $args += $d }
$args += "/XF"; $args += ".DS_Store"
$args += "/NFL", "/NDL", "/NJH", "/NJS", "/nc", "/ns", "/np"
& robocopy @args
if ($LASTEXITCODE -ge 8) { exit $LASTEXITCODE }
Write-Host "Done. See FINAL_GITHUB_SUBMISSION\HOW_TO_UPLOAD_TO_GITHUB.txt"
