# Windows setup (PowerShell) - equivalent of `make setup` + `make test`.
# From the repo root:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"

function Run-Step($label, $exe, $stepArgs) {
    Write-Host ""
    Write-Host ">>> $label"
    & $exe @stepArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "FAILED at step: $label (exit $LASTEXITCODE). Fix or report this output; setup did NOT complete."
        exit 1
    }
}

$ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "python $ver detected"
if ([version]$ver -lt [version]"3.11") {
    Write-Host "Need Python 3.11+. With conda:  conda create -n bball python=3.11 -y ; conda activate bball ; then re-run this script."
    exit 1
}

Run-Step "create venv" "python" @("-m", "venv", ".venv")
$py = ".venv\Scripts\python.exe"
Run-Step "upgrade pip" $py @("-m", "pip", "install", "--upgrade", "pip")
Run-Step "install torch (cpu, ~200 MB - be patient, do not Ctrl+C)" $py @(
    "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu",
    "torch==2.8.0", "torchvision==0.23.0")
Run-Step "install project + app deps" $py @("-m", "pip", "install", "-e", ".[dev,app]")
Run-Step "run test suite" $py @("-m", "pytest", "-q")

Write-Host ""
Write-Host "Setup complete. Start the workbench with:  powershell -ExecutionPolicy Bypass -File .\app.ps1"
Write-Host "Then open http://localhost:8000"
