#requires -Version 5.1
<#
.SYNOPSIS
  Builds xcl.exe into a portable Windows folder.

.DESCRIPTION
  From a normal PowerShell window in the repository root:

      .\scripts\build-windows.ps1

  It builds xcl.exe with Cargo, copies DXC (dxcompiler.dll, dxil.dll) from the
  Windows SDK when it finds them, puts the D3D12 Agility SDK in dist\bin\D3D12\
  (needed for Shader Model 6.6 and mesh shaders on Windows 10), writes a short
  quick start and a run.cmd, and zips the folder as dist\xcl-windows-x64.zip.

  Run .\scripts\setup-windows.ps1 first on a machine without the toolchain.

.PARAMETER Clean
  Remove dist\ and the release binary before building.
#>
[CmdletBinding()]
param(
    [switch]$Clean
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = Split-Path $PSScriptRoot -Parent
$target = Join-Path $root 'target\release'
$dist = Join-Path $root 'dist'
$bin = Join-Path $dist 'bin'

function Write-Step([string]$text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

function Need([string]$cmd, [string]$hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "'$cmd' is not on PATH. $hint"
    }
}

# A normal PowerShell window does not have link.exe; borrow the environment from
# the Visual Studio installation when it is there.
function Import-MsvcEnvironment {
    if (Get-Command link.exe -ErrorAction SilentlyContinue) { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { return }
    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { return }
    Write-Step 'Importing the MSVC environment'
    $envText = & cmd.exe /c "`"$vcvars`" >nul && set"
    foreach ($line in $envText) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}

Push-Location $root
try {
    Write-Step 'Checking the toolchain'
    Need 'cargo' 'Install Rust from https://rustup.rs or run .\scripts\setup-windows.ps1.'
    if (Get-Command rustup -ErrorAction SilentlyContinue) {
        rustup target add x86_64-pc-windows-msvc | Out-Host
    }
    Import-MsvcEnvironment
    if (-not (Get-Command link.exe -ErrorAction SilentlyContinue)) {
        throw 'link.exe was not found. Open "x64 Native Tools Command Prompt for VS", or run .\scripts\setup-windows.ps1.'
    }

    if ($Clean) {
        Write-Step 'Cleaning'
        Remove-Item -Recurse -Force $dist -ErrorAction SilentlyContinue
        cargo clean -p xcl
    }

    Write-Step 'Building xcl.exe'
    cargo build --release --locked -p xcl
    $exe = Join-Path $target 'xcl.exe'
    if (-not (Test-Path $exe)) { throw "cargo finished but $exe does not exist" }
    New-Item -ItemType Directory -Force $bin | Out-Null
    Copy-Item $exe $bin -Force

    Write-Step 'Bundling DXC'
    $kits = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
    $dxc = $kits | ForEach-Object { Join-Path $_.FullName 'x64\dxcompiler.dll' } |
           Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($dxc) {
        $dir = Split-Path $dxc
        Copy-Item (Join-Path $dir 'dxcompiler.dll') $bin -Force
        if (Test-Path (Join-Path $dir 'dxil.dll')) { Copy-Item (Join-Path $dir 'dxil.dll') $bin -Force }
        Write-Host "  copied from $dir"
    } else {
        Write-Warning 'dxcompiler.dll was not found in the Windows SDK; xcl.exe will look for it on the target PC.'
    }

    $agility = Join-Path $bin 'D3D12'
    if (Test-Path $agility) {
        Write-Host "  Agility SDK already present in $agility"
    } else {
        try {
            Write-Step 'Downloading the D3D12 Agility SDK'
            $zip = Join-Path $env:TEMP 'd3d12-agility.zip'
            Invoke-WebRequest -UseBasicParsing -Uri 'https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.615.1' -OutFile $zip
            $tmp = Join-Path $env:TEMP ('d3d12-agility-' + [guid]::NewGuid())
            Expand-Archive $zip -DestinationPath $tmp -Force
            New-Item -ItemType Directory -Force $agility | Out-Null
            Copy-Item (Join-Path $tmp 'build\native\bin\x64\*.dll') $agility -Force
            Remove-Item -Recurse -Force $tmp
            Write-Host "  copied to $agility"
        } catch {
            Write-Warning "the Agility SDK was not bundled: $_"
        }
    }

    Write-Step 'Writing the quick start'
    @'
xcl - portable Windows build
============================

Run run.cmd, or from a terminal:

  xcl.exe run "D:\Games\MyTitle"

xcl.exe opened from Explorer asks for the game's exe or MicrosoftGame.config.

Requirements: Windows 10/11 x64, a D3D12 GPU with Shader Model 6.6 and mesh
shaders. DXC is bundled; on Windows 10 the D3D12 Agility SDK is in D3D12\.
Gaming Services (Microsoft Store) adds dialogs and invites; without it xcl uses
its own task queues, async and networking.

Per-title data, logs and report.json: %LOCALAPPDATA%\XCL\<title>\.
Experimental research software: use only titles you own.
'@ | Set-Content (Join-Path $bin 'QUICKSTART.txt') -Encoding UTF8

    @'
@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
    start "" "xcl.exe"
) else (
    "xcl.exe" %*
)
'@ | Set-Content (Join-Path $bin 'run.cmd') -Encoding ASCII

    Write-Step 'Zipping'
    $zipPath = Join-Path $dist 'xcl-windows-x64.zip'
    Compress-Archive -Path (Join-Path $bin '*') -DestinationPath $zipPath -Force

    Write-Step 'Done'
    Get-ChildItem $bin | Select-Object Name, Length | Format-Table -AutoSize
    Write-Host "Built into $bin" -ForegroundColor Green
    Write-Host "Portable zip: $zipPath" -ForegroundColor Green
} finally {
    Pop-Location
}
