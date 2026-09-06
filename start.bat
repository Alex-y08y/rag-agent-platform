@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
    echo.
    echo Failed to start. Please ensure PowerShell is available.
    pause
)
