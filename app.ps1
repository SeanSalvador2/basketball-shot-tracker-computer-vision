# Start the web workbench on Windows —  powershell -ExecutionPolicy Bypass -File .\app.ps1
$ErrorActionPreference = "Stop"
& .venv\Scripts\python.exe -m uvicorn bball.app.server:app --host 0.0.0.0 --port 8000
