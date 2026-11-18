$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
$env:TEMP = Join-Path $projectPath 'tmp'
$env:TMP = $env:TEMP
& (Join-Path $projectPath '.venv\Scripts\python.exe') -m pytest -q
exit $LASTEXITCODE
