param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
$env:TEMP = Join-Path $projectPath 'tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectPath '.cache\pip'
& (Join-Path $projectPath '.venv\Scripts\python.exe') -m uvicorn app.main:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
