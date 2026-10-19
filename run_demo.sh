#!/usr/bin/env bash
# One-click LucidPPN demo for macOS / Linux:  bash run_demo.sh
# First run: creates .venv and installs PyTorch + Streamlit (about 5-10 minutes, needs internet).
# Later runs start the app straight away. Needs Python 3.10-3.13.
set -e
cd "$(dirname "$0")"

if [ ! -f .venv/setup_done ]; then
    echo "Setting up the demo environment (first run only, about 5-10 minutes)..."
    PY=$(command -v python3 || command -v python) || { echo "Python 3 not found: install Python 3.12 from https://www.python.org/downloads/"; exit 1; }
    [ -x .venv/bin/python ] || "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    if [ "$(uname)" = "Darwin" ]; then
        .venv/bin/python -m pip install torch torchvision
    else
        .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    .venv/bin/python -m pip install streamlit matplotlib pillow numpy fastapi uvicorn python-multipart
    touch .venv/setup_done
fi

echo "Starting the LucidPPN demo at http://localhost:8501 (Ctrl+C to stop)"
.venv/bin/python -m streamlit run deploy/app.py
