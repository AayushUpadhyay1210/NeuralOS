# NeuralOS setup for Windows PowerShell. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Add -NoRL to skip PyTorch / Stable-Baselines3 (baselines and the UI still work).
param([switch]$NoRL)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
& $py -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'"
if ($LASTEXITCODE -ne 0) { throw "Python 3.10 or newer is required" }

if (-not (Test-Path .venv)) { & $py -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($NoRL) {
    & .\.venv\Scripts\python.exe -m pip install -e "backend[dev]"
} else {
    # On Windows the default PyPI torch wheel is CPU-only.
    & .\.venv\Scripts\python.exe -m pip install -e "backend[rl,dev]"
}
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Push-Location frontend
npm ci
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm ci failed" }
Pop-Location
Write-Host ""
Write-Host "Setup complete. Activate the environment with: .\.venv\Scripts\Activate.ps1"
