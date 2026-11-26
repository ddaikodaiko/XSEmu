#requires -Version 5.1
<#
.SYNOPSIS
  Builds XSEmu and, when Qt is available, the launcher, into a portable folder.

.DESCRIPTION
  One command from a normal PowerShell window:

      .\scripts\build-windows.ps1

  It builds XSEmu.exe with Cargo, builds XSEmuLauncher with CMake when Qt 6 and
  Ninja are present, copies DXC (and the D3D12 Agility SDK when found) next to
  them, and writes dist\bin\QUICKSTART.txt. Run .\scripts\setup-windows.ps1
  first on a machine without the toolchain.

.PARAMETER QtDir
  Qt installation prefix (the folder that holds lib\cmake\Qt6). Defaults to
  QT_ROOT_DIR or C:\Qt\6.8.3\msvc2022_64.

.PARAMETER SkipLauncher
  Build only XSEmu.exe (the command-line emulator).

.PARAMETER Clean
  Remove the CMake build folder and the release binaries before building.
#>
[CmdletBinding()]
param(
    [string]$QtDir = $env:QT_ROOT_DIR,
    [switch]$SkipLauncher,
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
    if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
        Write-Warning 'rustup not found; assuming the MSVC toolchain is already installed.'
    } else {
        rustup target add x86_64-pc-windows-msvc | Out-Host
    }
    Import-MsvcEnvironment
    if (-not (Get-Command link.exe -ErrorAction SilentlyContinue)) {
        throw 'link.exe was not found. Open "x64 Native Tools Command Prompt for VS", or run .\scripts\setup-windows.ps1.'
    }

    if ($Clean) {
        Write-Step 'Cleaning'
        Remove-Item -Recurse -Force (Join-Path $root 'build') -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force (Join-Path $root 'dist') -ErrorAction SilentlyContinue
    }

    Write-Step 'Building XSEmu.exe'
    cargo build --release --locked -p xcl
    $exe = Join-Path $target 'XSEmu.exe'
    if (-not (Test-Path $exe)) { throw "cargo finished but $exe does not exist" }

    New-Item -ItemType Directory -Force $bin | Out-Null
    Copy-Item $exe $bin -Force

    if (-not $SkipLauncher) {
        if (-not $QtDir) {
            $candidate = 'C:\Qt\6.8.3\msvc2022_64'
            if (Test-Path $candidate) { $QtDir = $candidate }
        }
        $haveQt = $QtDir -and (Test-Path (Join-Path $QtDir 'lib\cmake\Qt6'))
        if (-not $haveQt) {
            Write-Warning 'Qt 6 was not found (pass -QtDir, set QT_ROOT_DIR, or run .\scripts\setup-windows.ps1). Building the command-line emulator only.'
        } elseif (-not (Get-Command cmake -ErrorAction SilentlyContinue) -or -not (Get-Command ninja -ErrorAction SilentlyContinue)) {
            Write-Warning 'CMake or Ninja is missing; building the command-line emulator only.'
        } else {
            Write-Step "Building the launcher with Qt at $QtDir"
            $build = Join-Path $root 'build'
            cmake -S launcher -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release `
                -DCMAKE_PREFIX_PATH="$QtDir" -DXSEMU_LAUNCHER_TESTS=OFF -DXSEMU_FAKE_EMULATOR=OFF
            if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
            cmake --build $build
            if ($LASTEXITCODE -ne 0) { throw 'the launcher build failed' }
            cmake --install $build --prefix $dist
            if ($LASTEXITCODE -ne 0) { throw 'the launcher install failed' }
            # windeployqt runs at install time; the emulator goes next to the launcher.
            Copy-Item $exe $bin -Force
        }
    }

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
        Write-Warning 'dxcompiler.dll was not found in the Windows SDK; XSEmu will look for it on the target PC.'
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
    @"
XSEmu - portable Windows build
==============================

Run XSEmuLauncher.exe (or run-xsemu.cmd). Choose a title's folder or its exe.

Requirements: Windows 10/11 x64, a D3D12 GPU with Shader Model 6.6 and mesh
shaders. DXC is bundled; the D3D12 Agility SDK is in D3D12\.

Logs and per-title data live in %LOCALAPPDATA%\XCL\<title>\.
Experimental research software: use only titles you own.
"@ | Set-Content (Join-Path $bin 'QUICKSTART.txt') -Encoding UTF8
    @'
@echo off
start "" "%~dp0XSEmuLauncher.exe"
'@ | Set-Content (Join-Path $bin 'run.cmd') -Encoding ASCII

    Write-Step 'Done'
    Get-ChildItem $bin | Select-Object Name, Length | Format-Table -AutoSize
    Write-Host "Built into $bin" -ForegroundColor Green
    if (-not (Test-Path (Join-Path $bin 'XSEmuLauncher.exe'))) {
        Write-Host 'Launcher not built. Run XSEmu.exe from a terminal, e.g.:' -ForegroundColor Yellow
        Write-Host "  $bin\XSEmu.exe run `"D:\Games\MyTitle`""
    }
} finally {
    Pop-Location
}
