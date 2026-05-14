@echo off
REM ============================================================
REM  comfyui-mesh server — one-shot installer
REM ============================================================
REM
REM  What this does:
REM    1. Finds Python on your system (py launcher preferred, python fallback)
REM    2. Creates a local .venv in this folder if one doesn't exist
REM    3. Upgrades pip + wheel inside the venv
REM    4. Clones ComfyUI as ..\ComfyUI if not already there
REM    5. Installs ComfyUI's requirements (gets torch with CUDA)
REM    6. Installs the back-half server's extras (cuda-bindings, etc.)
REM    7. Runs install_check.py to confirm everything is wired up
REM
REM  Re-run safely — every step is idempotent. Skips work already done.
REM
REM  After this finishes, drop your FLUX safetensors into this folder
REM  and double-click run_server_gui.bat (or run_server.bat).
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  comfyui-mesh server installer
echo ============================================================
echo.

REM ---- 1. Find a Python interpreter ----
where py >nul 2>&1
if not errorlevel 1 (
    set "PY=py -3"
    goto :have_python
)
where python >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :have_python
)
echo [install] ERROR: no Python found on PATH.
echo            Install Python 3.10 or newer from https://www.python.org/downloads/
echo            then re-run this script.
exit /b 1

:have_python
echo [install] python interpreter: %PY%
%PY% --version

REM ---- 2. Create .venv if missing ----
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_PY%" (
    echo [1/6] venv already exists at %VENV_DIR%
) else (
    echo [1/6] creating venv at %VENV_DIR% ...
    %PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [install] ERROR: venv creation failed.
        exit /b 1
    )
)

REM ---- 3. Upgrade pip + wheel ----
echo [2/6] upgrading pip and wheel ...
"%VENV_PY%" -m pip install --upgrade pip wheel >nul
if errorlevel 1 (
    echo [install] ERROR: pip/wheel upgrade failed.
    exit /b 1
)

REM ---- 4. Clone ComfyUI if missing ----
set "COMFY_DIR=%~dp0..\ComfyUI"
if exist "%COMFY_DIR%\comfy" (
    echo [3/6] ComfyUI source already at %COMFY_DIR%
    set "SKIP_COMFY_DEPS=1"
) else (
    echo [3/6] cloning ComfyUI to %COMFY_DIR% ...
    where git >nul 2>&1
    if errorlevel 1 (
        echo [install] ERROR: git not found on PATH. Install Git for Windows
        echo            ^(https://git-scm.com/^) or manually clone:
        echo                git clone https://github.com/comfyanonymous/ComfyUI "%COMFY_DIR%"
        exit /b 1
    )
    pushd "%~dp0.."
    git clone https://github.com/comfyanonymous/ComfyUI ComfyUI
    if errorlevel 1 (
        popd
        echo [install] ERROR: ComfyUI clone failed.
        exit /b 1
    )
    popd
    set "SKIP_COMFY_DEPS=0"
)

REM ---- 5. Install ComfyUI's requirements (gets torch with CUDA) ----
if "%SKIP_COMFY_DEPS%"=="1" (
    echo [4/6] skipping ComfyUI requirements ^(already installed^)
) else (
    echo [4/6] installing ComfyUI's requirements ^(this pulls torch — multi-GB^) ...
    "%VENV_PY%" -m pip install -r "%COMFY_DIR%\requirements.txt"
    if errorlevel 1 (
        echo [install] ERROR: ComfyUI requirements install failed.
        exit /b 1
    )
)

REM ---- 6. Install our extras (cuda-bindings, etc.) ----
echo [5/6] installing comfyui-mesh server requirements ...
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [install] ERROR: server requirements install failed.
    exit /b 1
)

REM ---- 7. Run the pre-flight check ----
echo [6/6] running install_check.py ...
echo.
"%VENV_PY%" "%~dp0install_check.py"

echo.
echo ============================================================
echo  Install complete.
echo ============================================================
echo.
echo  Next:
echo    1. Drop your FLUX safetensors checkpoint into this folder
echo       ^(e.g. flux-2-klein-9b-fp8.safetensors^)
echo.
echo    2. Launch the server:
echo         run_server_gui.bat   ^(Tkinter UI — recommended for first run^)
echo         run_server.bat       ^(headless; edit N_BLOCKS at the top first^)
echo.
echo  The server listens on 0.0.0.0:7777 by default. Tell the
echo  ComfyUI client this host's LAN/Tailscale IP and that port.
echo.

endlocal
