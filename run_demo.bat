@echo off
rem One-click LucidPPN demo for Windows. Double-click this file, or run it from a terminal.
rem First run: creates .venv and installs CPU PyTorch + Streamlit (about 5-10 minutes, needs internet).
rem Later runs start the app straight away. Needs Python 3.10-3.13 (python.org, tick "Add python.exe to PATH").
cd /d "%~dp0"

if exist .venv\setup_done goto run

echo Setting up the demo environment (first run only, about 5-10 minutes)...
if not exist .venv\Scripts\python.exe (
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo.
    echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation, then run this file again.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip || goto failed
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto failed
.venv\Scripts\python.exe -m pip install streamlit matplotlib pillow numpy fastapi uvicorn python-multipart || goto failed
echo ok> .venv\setup_done

:run
echo Starting the LucidPPN demo at http://localhost:8501 (close this window to stop it)
.venv\Scripts\python.exe -m streamlit run deploy\app.py
exit /b 0

:failed
echo.
echo Installation failed (see the error above). Check your internet connection and run this file again.
pause
exit /b 1
