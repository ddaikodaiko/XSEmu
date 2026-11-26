@echo off
rem Builds XSEmu on first use, then opens the launcher.
setlocal
cd /d "%~dp0"

if not exist "dist\bin\XSEmuLauncher.exe" (
    echo XSEmu is not built yet. Building it now...
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build-windows.ps1"
    if errorlevel 1 (
        echo.
        echo The build failed. See the messages above.
        pause
        exit /b 1
    )
)

if exist "dist\bin\XSEmuLauncher.exe" (
    start "" "dist\bin\XSEmuLauncher.exe"
) else (
    echo The launcher was not built.
    echo Run a title with: dist\bin\XSEmu.exe run "D:\Games\MyTitle"
    pause
)
