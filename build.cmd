@echo off
rem One-command Windows build. See scripts\build-windows.ps1 for options.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build-windows.ps1" %*
if errorlevel 1 (
    echo.
    echo The build failed. See the messages above.
    pause
    exit /b 1
)
pause
