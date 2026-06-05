param(
    [string]$EnvName = "mda",
    [string]$PythonVersion = "3.10",
    [switch]$SkipSystemInstalls,
    [switch]$SkipCheckpoint
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Find-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniforge3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\Programs\Miniforge3\Scripts\conda.exe"
    )

    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }

    return $null
}

function Ensure-Miniforge {
    $conda = Find-Conda
    if ($conda) {
        return $conda
    }

    if ($SkipSystemInstalls) {
        throw "conda was not found. Install Miniforge3 or rerun without -SkipSystemInstalls."
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget was not found. Install Miniforge3 manually, then rerun this script."
    }

    winget install --id CondaForge.Miniforge3 --exact --scope user --accept-package-agreements --accept-source-agreements
    $conda = Find-Conda
    if (-not $conda) {
        throw "Miniforge3 installed, but conda.exe was not found in the expected user paths. Restart PowerShell and rerun."
    }
    return $conda
}

function Ensure-Ffmpeg {
    $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpeg) {
        return
    }

    $wingetFfmpeg = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
    if (Test-Path (Join-Path $wingetFfmpeg "ffmpeg.exe")) {
        $env:PATH = "$wingetFfmpeg;$env:PATH"
        return
    }

    if ($SkipSystemInstalls) {
        Write-Warning "ffmpeg was not found. Video input will not work until ffmpeg is on PATH."
        return
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Warning "winget was not found. Install ffmpeg manually for video input support."
        return
    }

    winget install --id Gyan.FFmpeg --exact --scope user --accept-package-agreements --accept-source-agreements
    if (Test-Path (Join-Path $wingetFfmpeg "ffmpeg.exe")) {
        $env:PATH = "$wingetFfmpeg;$env:PATH"
    }
}

function Test-CondaEnv {
    param([string]$CondaPath, [string]$Name)
    $json = & $CondaPath env list --json | ConvertFrom-Json
    foreach ($envPath in $json.envs) {
        if ((Split-Path $envPath -Leaf) -eq $Name) {
            return $true
        }
    }
    return $false
}

$conda = Ensure-Miniforge
Ensure-Ffmpeg

Push-Location $RepoRoot
try {
    $env:PYTHONNOUSERSITE = "1"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

    if (-not (Test-CondaEnv -CondaPath $conda -Name $EnvName)) {
        & $conda create -n $EnvName "python=$PythonVersion" -y
    }

    & $conda run -n $EnvName python -m pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url "https://download.pytorch.org/whl/cu128"
    & $conda run -n $EnvName python -m pip install "numpy<2" "xformers==0.0.31.post1" "setuptools<81" addict
    & $conda run -n $EnvName python -m pip install -e .
    & $conda run -n $EnvName python -m pip install hydra-core lightning lightning-bolts torchmetrics rootutils accelerate peft

    if (-not $SkipCheckpoint) {
        New-Item -ItemType Directory -Force -Path "checkpoints\MDA" | Out-Null
        if (-not (Test-Path "checkpoints\MDA\DA3_MOG_Sky_LogL2.ckpt")) {
            & $conda run -n $EnvName hf download sy000/MDA DA3_MOG_Sky_LogL2.ckpt --local-dir checkpoints/MDA
        }
    }

    & $conda run -n $EnvName python -c "import torch, numpy, xformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('numpy', numpy.__version__); print('xformers', xformers.__version__)"
}
finally {
    Pop-Location
}
