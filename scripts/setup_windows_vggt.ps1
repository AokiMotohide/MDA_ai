param(
    [string]$EnvName = "mda",
    [switch]$InstallCommercialModel
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VggtCommit = "9e4fa662a8893ed348d048e8b57816c12593448b"

function Find-Conda {
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $candidates = @(
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniforge3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\Programs\Miniforge3\Scripts\conda.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "conda was not found. Run scripts/setup_windows_mda_da3.ps1 first."
}

$conda = Find-Conda
Push-Location $RepoRoot
try {
    $env:PYTHONNOUSERSITE = "1"
    & $conda run -n $EnvName python -m pip install `
        "git+https://github.com/facebookresearch/vggt.git@$VggtCommit" `
        onnxruntime scipy

    & $conda run -n $EnvName python -c "from vggt.models.vggt import VGGT; import onnxruntime; print('VGGT package: OK'); print('onnxruntime: OK')"

    if ($InstallCommercialModel) {
        & $conda run -n $EnvName python python_mda_customScript\run_vggt_commercial.py --download-only
    }
}
finally {
    Pop-Location
}
