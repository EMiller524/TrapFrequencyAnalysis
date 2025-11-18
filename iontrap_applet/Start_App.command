#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python3 -m pip install --user --upgrade -r iontrap_applet/requirements-app.txt
python3 -m streamlit run iontrap_applet/app.py
chmod +x iontrap_applet/Start_App.command
