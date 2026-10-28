# Serve the risk desk on localhost. Paths are relative to this repository.
$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
$env:TEMP = Join-Path $Root "tmp"
$env:TMP = Join-Path $Root "tmp"
$env:PYTHONPYCACHEPREFIX = Join-Path $Root ".cache\pyc"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "tmp"), (Join-Path $Root ".cache\pyc") | Out-Null
Set-Location $Root

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 (Join-Path $Root "dashboard\server.py")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python (Join-Path $Root "dashboard\server.py")
} else {
    throw "Python 3 is required."
}
