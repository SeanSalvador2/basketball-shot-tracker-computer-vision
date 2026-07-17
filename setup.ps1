# Windows setup (PowerShell) - equivalent of `make setup` + `make test`.
# From the repo root:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"

$ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "python $ver detected"
if ([version]$ver -lt [version]"3.11") {
    Write-Host "Need Python 3.11+. With conda:  conda create -n bball python=3.11 -y ; conda activate bball ; then re-run this script."
    exit 1
}

python -m venv .venv
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
& .venv\Scripts\python.exe -m pip install -e ".[dev,app]"
& .venv\Scripts\python.exe -m pytest -q

Write-Host ""
Write-Host "Setup complete. Start the workbench with:  powershell -ExecutionPolicy Bypass -File .\app.ps1"
Write-Host "Then open http://localhost:8000"
