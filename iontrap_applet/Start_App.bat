@echo off
setlocal EnableExtensions

REM Start the Ion Trap Resonance Explorer
cd /d "%~dp0\.."

set "VENV_DIR=.venv"
set "PY_CREATE="

REM --- need py -3.11 if available ---
where py >nul 2>nul
if errorlevel 1 goto NO_PY

REM py exists: require 3.11
py -3.11 -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [error] 'py' exists but Python 3.11 is not installed.
    echo         Install Python 3.11.x and try again.
    pause
    exit /b 1
)
set "PY_CREATE=py -3.11"
goto HAVE_CREATE

:NO_PY
REM No py: require python on PATH to be 3.11.x
python -c "import sys; raise SystemExit(0 if (sys.version_info[0]==3 and sys.version_info[1]==11) else 1)" >nul 2>nul
if errorlevel 1 (
    set "SYSVER=unknown"
    for /f "delims=" %%V in ('python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2^>nul') do set "SYSVER=%%V"
    echo [error] Python 3.11.x is required, but 'python' on PATH is %SYSVER%.
    echo         Install Python 3.11.x, or install the Windows Python Launcher ('py') so the script can select it.
    pause
    exit /b 1
)
set "PY_CREATE=python"

:HAVE_CREATE
REM --- Create venv if needed ---
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [setup] Creating virtual environment in %VENV_DIR% ...
    %PY_CREATE% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [error] Failed to create venv.
        pause
        exit /b 1
    )
)

REM  Activate venv 
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [error] Failed to activate venv.
    pause
    exit /b 1
)

REM enforce 3.11.x
python -c "import sys; raise SystemExit(0 if (sys.version_info[0]==3 and sys.version_info[1]==11) else 1)" >nul 2>nul
if errorlevel 1 (
    for /f "delims=" %%V in ('python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"') do set "VENVVER=%%V"
    echo [error] This app requires Python 3.11.x, but the venv is %VENVVER%.
    echo         Delete the "%VENV_DIR%" folder and run this launcher again.
    pause
    exit /b 1
)

REM -install packages 
python -m pip install --upgrade pip >nul
python -m pip install -r "iontrap_applet\requirements-app.txt"
if errorlevel 1 (
    echo [error] Installing requirements failed.
    pause
    exit /b 1
)

REM -run the appp
python -m streamlit run "iontrap_applet\app.py"
