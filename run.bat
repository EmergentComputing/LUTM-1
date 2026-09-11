@echo off
setlocal
cd /d "%~dp0"

call conda activate slackenv
if errorlevel 1 (
    echo Could not activate the conda environment "slackenv".
    pause
    exit /b 1
)

python control_pad.py
if errorlevel 1 pause
