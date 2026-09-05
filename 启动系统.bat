@echo off
title CT Diagnostic System - Launcher
cd /d "%~dp0"

echo ============================================
echo   Computational Thinking Diagnostic System
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ first.
    echo Download: https://www.python.org/downloads/
    echo Check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

python -c "import streamlit, xgboost, shap, statsmodels, openpyxl, docx" >nul 2>nul
if errorlevel 1 (
    echo First run: installing dependencies, please wait 3-5 minutes...
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check network and retry.
        pause
        exit /b 1
    )
)

echo Starting system... browser will open shortly.
start "" http://localhost:8501
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause
