param(
    [string]$EnvName = "mda"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

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
    throw "Miniforge3 was installed but conda.exe was not found. Open a new PowerShell window and rerun this script."
}

Push-Location $RepoRoot
try {
    $env:PYTHONNOUSERSITE = "1"

    # 商用 VGGT は DA3 MDA 重みを使わない。環境だけ作り、不要な重み取得を避ける。
    & .\scripts\setup_windows_mda_da3.ps1 -EnvName $EnvName -SkipCheckpoint
    $conda = Find-Conda

    Write-Host ""
    Write-Host "Hugging Face の Read 権限トークンを入力してください。"
    Write-Host "事前に https://huggingface.co/facebook/VGGT-1B-Commercial で利用規約へ同意する必要があります。"
    & $conda run -n $EnvName huggingface-cli login

    & .\scripts\setup_windows_vggt.ps1 -EnvName $EnvName -InstallCommercialModel
}
finally {
    Pop-Location
}
