# Build the CUDA risk engine. Downloads and object files stay on D:\HPC_L2.
$ErrorActionPreference = "Stop"
$Root = "D:\HPC_L2"
$env:TEMP = Join-Path $Root "tmp"
$env:TMP = Join-Path $Root "tmp"
$env:PIP_CACHE_DIR = Join-Path $Root ".cache\pip"
$env:PYTHONPYCACHEPREFIX = Join-Path $Root ".cache\pyc"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
New-Item -ItemType Directory -Force -Path @(
    (Join-Path $Root "tmp"),
    (Join-Path $Root "build"),
    (Join-Path $Root ".cache\pip"),
    (Join-Path $Root ".cache\pyc"),
    (Join-Path $Root "lambda_pkg")
) | Out-Null

$Nvrtc = Join-Path $Root ".cuda_pkgs\nvidia\cuda_nvrtc\bin\nvrtc64_120_0.dll"
if (-not (Test-Path $Nvrtc)) {
    py -3.11 -m pip install --target (Join-Path $Root ".cuda_pkgs") --cache-dir $env:PIP_CACHE_DIR "nvidia-cuda-nvrtc-cu12"
}

$Gcc = "D:\mys64\mingw64\bin\gcc.exe"
& $Gcc -O3 -std=c11 -Wall -Wextra -o (Join-Path $Root "build\mc_engine.exe") (Join-Path $Root "src\engine.c") -lm
if ($LASTEXITCODE -ne 0) { throw "gcc failed" }

Copy-Item (Join-Path $Root "aws\handler.py") (Join-Path $Root "lambda_pkg\handler.py") -Force
Copy-Item (Join-Path $Root "common\jobtext.py") (Join-Path $Root "lambda_pkg\jobtext.py") -Force

& (Join-Path $Root "build\mc_engine.exe") selftest
if ($LASTEXITCODE -ne 0) { throw "selftest failed" }
Write-Host "Built D:\HPC_L2\build\mc_engine.exe"
