@echo off
rem Builds xcl.exe on first use, then starts it. Arguments go to xcl.exe.
setlocal
cd /d "%~dp0"

if not exist "dist\bin\xcl.exe" (
    echo xcl is not built yet. Building it now...
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build-windows.ps1"
    if errorlevel 1 (
        echo.
        echo The build failed. See the messages above.
        pause
        exit /b 1
    )
)

if "%~1"=="" (
    start "" "dist\bin\xcl.exe"
) else (
    "dist\bin\xcl.exe" %*
)
