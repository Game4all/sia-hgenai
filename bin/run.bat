@echo off

set APP_FILE="main.py"
set VENV_DIR=".venv"

if not exist %VENV_DIR% (
    echo Creating virtual env for app...
    python -m venv %VENV_DIR%
)

call %VENV_DIR%\Scripts\activate
pip install -r requirements.txt


echo Launching Streamlit app...
streamlit run %APP_FILE%