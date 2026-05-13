@echo off
REM Solar AI Diagnostic — startup script (Windows)
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set VENV=%SCRIPT_DIR%.venv
set BACKEND=%SCRIPT_DIR%backend
set MODELS=%BACKEND%\models\saved

echo === Solar AI Diagnostic ===

REM ── Python check ──────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install it from https://python.org
    pause
    exit /b 1
)

REM ── Virtual environment ───────────────────────────────────────────────
if not exist "%VENV%" (
    echo Creating virtual environment...
    python -m venv "%VENV%"
)
call "%VENV%\Scripts\activate.bat"

REM ── Dependencies ──────────────────────────────────────────────────────
echo Installing dependencies...
pip install -r "%BACKEND%\requirements.txt" -q --disable-pip-version-check

REM ── Train models if missing ───────────────────────────────────────────
if not exist "%MODELS%\best_model.pkl" (
    echo Training AI models (first run - takes ~30 seconds)...
    python "%BACKEND%\models\train_all.py"
)

REM ── Find a free port and start Flask ─────────────────────────────────
echo Finding available port...
python -c "
import socket, subprocess, sys, os
port = 5001
while True:
    try:
        s = socket.socket()
        s.bind(('', port))
        s.close()
        break
    except OSError:
        port += 1
print(f'Starting server on http://localhost:{port}')
env = os.environ.copy()
env['PORT'] = str(port)
os.chdir(r'%BACKEND%')
subprocess.run([sys.executable, 'app.py'], env=env)
"

pause
