# Build the CUDA risk engine from the repository that contains this script.
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

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

function Test-Gcc64([string]$Path) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return $false }
    $machine = & $Path -dumpmachine 2>$null
    return [string]$machine -match "x86_64"
}

function Find-Gcc64 {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:CC) { $candidates.Add($env:CC) }
    $onPath = Get-Command gcc -ErrorAction SilentlyContinue
    if ($onPath) { $candidates.Add($onPath.Source) }
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem)) {
        $root = $drive.Root
        foreach ($rel in @(
            "mingw64\bin\gcc.exe",
            "msys64\mingw64\bin\gcc.exe",
            "TDM-GCC-64\bin\gcc.exe"
        )) {
            $candidates.Add((Join-Path $root $rel))
        }
        Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $candidates.Add((Join-Path $_.FullName "mingw64\bin\gcc.exe"))
            $candidates.Add((Join-Path $_.FullName "bin\gcc.exe"))
        }
    }
    foreach ($candidate in $candidates) {
        if (Test-Gcc64 $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    }
    throw "No 64-bit gcc was found. Install MinGW-w64 or set CC to its gcc.exe."
}

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)"
        if ($LASTEXITCODE -eq 0) { return @{ Exe = "py"; Prefix = @("-3") } }
    }
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        & $cmd.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)"
        if ($LASTEXITCODE -eq 0) { return @{ Exe = $cmd.Source; Prefix = @() } }
    }
    throw "Python 3.9 or newer is required."
}

$Python = Find-Python
$NvrtcDir = Join-Path $Root ".cuda_pkgs\nvidia\cuda_nvrtc\bin"
$HasNvrtc = $false
if (Test-Path -LiteralPath $NvrtcDir) {
    $HasNvrtc = [bool](Get-ChildItem -LiteralPath $NvrtcDir -Filter "nvrtc64_*.dll" -ErrorAction SilentlyContinue)
}
if (-not $HasNvrtc) {
    & $Python.Exe @($Python.Prefix + @("-m", "pip", "install", "--target", (Join-Path $Root ".cuda_pkgs"), "--cache-dir", $env:PIP_CACHE_DIR, "nvidia-cuda-nvrtc-cu12"))
    if ($LASTEXITCODE -ne 0) { throw "pip could not install nvidia-cuda-nvrtc-cu12" }
}

$Gcc = Find-Gcc64
Write-Host "Using $Gcc"
& $Gcc -O3 -std=c11 -Wall -Wextra -o (Join-Path $Root "build\mc_engine.exe") (Join-Path $Root "src\engine.c") -lm
if ($LASTEXITCODE -ne 0) { throw "gcc failed" }

Copy-Item (Join-Path $Root "aws\handler.py") (Join-Path $Root "lambda_pkg\handler.py") -Force
Copy-Item (Join-Path $Root "common\jobtext.py") (Join-Path $Root "lambda_pkg\jobtext.py") -Force

& (Join-Path $Root "build\mc_engine.exe") selftest
if ($LASTEXITCODE -ne 0) { throw "selftest failed" }
Write-Host "Built $(Join-Path $Root 'build\mc_engine.exe')"
