#requires -Version 5.1
<#
.SYNOPSIS
  Installs the Windows toolchain XSEmu needs.

.DESCRIPTION
  Installs, through winget: Rust (rustup), the Visual Studio 2022 C++ build
  tools, CMake and Ninja; then Qt 6.8.3 (MSVC 2022 x64) through aqtinstall.
  Run it from an elevated PowerShell once, then use .\scripts\build-windows.ps1.

  Nothing is installed silently without winget's own confirmation. If winget
  is missing, install "App Installer" from the Microsoft Store first.

.PARAMETER Qtdir
  Where Qt is installed. Defaults to C:\Qt.

.PARAMETER SkipQt
  Do not install Qt (only the emulator will be built).
#>
[CmdletBinding()]
param(
    [string]$QtDir = 'C:\Qt',
    [switch]$SkipQt
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Step([string]$text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Have([string]$cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

if (-not (Have 'winget')) {
    throw 'winget is not available. Install "App Installer" from the Microsoft Store, or install the tools by hand (see README, Requirements).'
}

$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elevated) {
    Write-Warning 'Not elevated: the Visual Studio build tools installer may ask for administrator rights.'
}

Write-Step 'Rust (rustup)'
if (Have 'cargo') { Write-Host '  already installed' }
else { winget install --id Rustlang.Rustup --exact --accept-source-agreements --accept-package-agreements }

Write-Step 'Visual Studio 2022 C++ build tools'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$haveMsvc = (Test-Path $vswhere) -and (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if ($haveMsvc) {
    Write-Host '  already installed'
} else {
    winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --accept-source-agreements --accept-package-agreements `
        --override '--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
}

Write-Step 'CMake and Ninja'
foreach ($pkg in 'Kitware.CMake', 'Ninja-build.Ninja') {
    winget install --id $pkg --exact --accept-source-agreements --accept-package-agreements
}

if (-not $SkipQt) {
    Write-Step "Qt 6.8.3 into $QtDir"
    if (Test-Path (Join-Path $QtDir '6.8.3\msvc2022_64\lib\cmake\Qt6')) {
        Write-Host '  already installed'
    } else {
        if (-not (Have 'python')) {
            winget install --id Python.Python.3.12 --exact --accept-source-agreements --accept-package-agreements
        }
        python -m pip install --upgrade aqtinstall
        python -m aqt install-qt windows desktop 6.8.3 win64_msvc2022_64 -O $QtDir
    }
}

Write-Step 'Done'
Write-Host 'Open a new PowerShell window (so PATH is refreshed) and run:' -ForegroundColor Green
Write-Host "  .\scripts\build-windows.ps1 -QtDir `"$QtDir\6.8.3\msvc2022_64`""
