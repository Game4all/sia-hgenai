#!/bin/bash

APP_FILE="main.py"
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo Creating virtual env for app...
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install -r requirements.txt


echo Launching Streamlit app...
streamlit run "$APP_FILE" --server.port 8501 --server.address 0.0.0.0