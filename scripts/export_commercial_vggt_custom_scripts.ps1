param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\dist")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$PackageName = "MDA_CommercialVGGT_CustomScripts.zip"
$StageDirectory = Join-Path $env:TEMP ("MDA_CommercialVGGT_CustomScripts_" + [guid]::NewGuid().ToString("N"))

$Files = @(
    "python_mda_customScript\run_vggt.py",
    "python_mda_customScript\run_vggt_commercial.py",
    "python_mda_customScript\run_vggt_common.py",
    "python_mda_customScript\run_vggt_manual.md",
    "scripts\setup_windows_mda_da3.ps1",
    "scripts\setup_windows_vggt.ps1",
    "scripts\setup_windows_commercial_vggt.ps1",
    "WINDOWS_BEGINNER_SETUP.md",
    "WINDOWS_RECONSTRUCTION_GUIDE.md",
    "CUSTOM_SCRIPT_PACKAGE_MANIFEST.md"
)

New-Item -ItemType Directory -Force -Path $OutputDirectory, $StageDirectory | Out-Null
try {
    foreach ($relativePath in $Files) {
        $source = Join-Path $RepoRoot $relativePath
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Required custom-script file was not found: $source"
        }
        $destination = Join-Path $StageDirectory $relativePath
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }

    $archive = Join-Path $OutputDirectory $PackageName
    Compress-Archive -Path (Join-Path $StageDirectory "*") -DestinationPath $archive -Force
    $hash = Get-FileHash -LiteralPath $archive -Algorithm SHA256
    Write-Host "作成完了: $archive"
    Write-Host "SHA256    : $($hash.Hash)"
}
finally {
    Remove-Item -LiteralPath $StageDirectory -Recurse -Force -ErrorAction SilentlyContinue
}
