# Serve the risk desk on localhost. Nothing is written to C:.
$Root = "D:\HPC_L2"
$env:TEMP = Join-Path $Root "tmp"
$env:TMP = Join-Path $Root "tmp"
$env:PYTHONPYCACHEPREFIX = Join-Path $Root ".cache\pyc"
Set-Location $Root
py -3.11 (Join-Path $Root "dashboard\server.py")
