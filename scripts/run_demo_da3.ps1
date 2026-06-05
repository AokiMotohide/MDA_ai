param(
    [string]$InputPath = "assets/examples/mono/painting/painting.jpeg",
    [string]$OutputDir = "eval_results/demo_windows_da3",
    [string]$ModelName = "mda_mog_sky_l2",
    [int]$Size = 512,
    [int]$MaxChunk = 1,
    [int]$Fps = 15,
    [string]$EnvName = "mda",
    [switch]$Viewer
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

    throw "conda was not found. Run scripts/setup_windows_mda_da3.ps1 first."
}

function Add-FfmpegToPath {
    $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($cmd) {
        return
    }

    $wingetFfmpeg = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
    if (Test-Path (Join-Path $wingetFfmpeg "ffmpeg.exe")) {
        $env:PATH = "$wingetFfmpeg;$env:PATH"
    }
}

$conda = Find-Conda
Add-FfmpegToPath

Push-Location $RepoRoot
try {
    $env:PYTHONNOUSERSITE = "1"

    if (-not (Test-Path "checkpoints\MDA\DA3_MOG_Sky_LogL2.ckpt")) {
        throw "DA3 MDA checkpoint was not found at checkpoints/MDA/DA3_MOG_Sky_LogL2.ckpt. Run scripts/setup_windows_mda_da3.ps1 first."
    }

    $viewerFlag = if ($Viewer) { "--viewer" } else { "--no-viewer" }
    $args = @(
        "demo.py",
        $InputPath,
        "--model_name", $ModelName,
        "--size", "$Size",
        "--max_chunk", "$MaxChunk",
        "--fps", "$Fps",
        "--output_dir", $OutputDir,
        $viewerFlag
    )

    & $conda run -n $EnvName python @args
}
finally {
    Pop-Location
}
