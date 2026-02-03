#!/usr/bin/env bash
set -euo pipefail

# Always run from where this .command lives
cd "$(dirname "$0")"

# Detect app folder automatically (supports repo root OR iontrap_applet/)
if [[ -f "app.py" && -f "requirements-app.txt" ]]; then
  APP_DIR="."
elif [[ -f "iontrap_applet/app.py" && -f "iontrap_applet/requirements-app.txt" ]]; then
  APP_DIR="iontrap_applet"
else
  echo "Couldn't find app.py and requirements-app.txt."
  echo "Place this .command in the repo root or inside iontrap_applet/."
  exit 1
fi

# Pick a Python (prefer python3)
PY=$(command -v python3 || true)
if [[ -z "${PY}" ]]; then
  PY=$(command -v python || true)
fi
if [[ -z "${PY}" ]]; then
  echo "Python 3 not found. Install Python 3 first (e.g., from python.org or Homebrew)."
  exit 1
fi

# Install/upgrade deps into user site (no admin needed)
"$PY" -m pip install --user --upgrade -r "$APP_DIR/requirements-app.txt"

# Run the Streamlit app with the SAME interpreter that installed deps
exec "$PY" -m streamlit run "$APP_DIR/app.py"
