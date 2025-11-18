@echo off
setlocal
REM Start the Ion Trap Resonance Explorer
cd /d %~dp0\..
REM Install/update required packages to user site (no admin, no venv)
python -m pip install --user -r iontrap_applet\requirements-app.txt
REM Run the app
python -m streamlit run iontrap_applet\app.py
