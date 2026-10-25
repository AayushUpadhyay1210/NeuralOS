# Runs every automated check on Windows PowerShell.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest;              if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
& $py -m ruff check backend tests; if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
& $py -m mypy;                if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
Push-Location frontend
foreach ($s in "typecheck", "lint", "test", "build") {
    npm run $s
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm run $s failed" }
}
Pop-Location
Write-Host "All checks passed."
