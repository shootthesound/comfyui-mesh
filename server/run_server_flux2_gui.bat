@echo off
REM Launcher: comfyui-mesh server with the Tkinter GUI.
REM
REM No N_BLOCKS env var here -- the GUI's spinbox handles that. Same goes
REM for device pinning: the GUI's dropdown sets CUDA_VISIBLE_DEVICES on
REM the subprocess. This bat just opens the GUI.

setlocal

REM ---- Where to find ComfyUI's Python sources ----
if "%COMFYUI_PATH%"=="" (
    set COMFYUI_PATH=%~dp0ComfyUI
)
echo [run_server_gui] COMFYUI_PATH=%COMFYUI_PATH%

REM ---- Pick python: prefer local .venv, else fall back to PATH ----
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [run_server_gui] using venv python: %VENV_PY%
) else (
    set "PY=python"
    echo [run_server_gui] WARNING: %VENV_PY% not found, falling back to 'python' on PATH
)

REM ---- Drop a marker file just before launching python so the GUI
REM ---- can compute "bat -> python ready" wall-clock delta in its
REM ---- startup log. The marker is touched (created/updated) here;
REM ---- mesh_server_gui.py reads its mtime in _log_session_header.
echo. > "%~dp0mesh_server_gui_bat_t0.tmp"

REM ---- Clear stale ready-sentinel so the splash actually waits ----
del /f /q "%~dp0mesh_server_gui_ready.tmp" >nul 2>nul

REM ---- Launch the cmd-console splash in parallel. It tells the user
REM ---- something is happening during the 10-30s python.exe + venv +
REM ---- tkinter cold start, and self-closes once mesh_server_gui.py
REM ---- writes the ready sentinel.
start "ComfyUI Mesh : Daedalus FLUX 2 starting" "%~dp0_splash_flux2.cmd"

REM ---- Launch the GUI (uses pythonw if available so no console window) ----
set "VENV_PYW=%~dp0.venv\Scripts\pythonw.exe"
if exist "%VENV_PYW%" (
    REM Launch GUI without a parent console. The Tk window has its own
    REM output area for the server's stdout.
    start "" "%VENV_PYW%" "%~dp0mesh_server_gui.py"
) else (
    "%PY%" "%~dp0mesh_server_gui.py"
)
