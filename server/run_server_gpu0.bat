@echo off
REM Launcher: comfyui-mesh back-half server pinned to GPU 0 (the first card).
REM
REM Use this when GPU 0 is your second card (the one NOT running ComfyUI).
REM For a two-machine setup where this machine only has one GPU, use this.

setlocal

REM Pin this process to the first GPU. After this line the server only
REM sees one card and addresses it as cuda:0.
set CUDA_VISIBLE_DEVICES=0
echo [run_server] pinned to physical GPU 0 (CUDA_VISIBLE_DEVICES=0)

REM ---- 1. Where to find ComfyUI's Python sources ----
if "%COMFYUI_PATH%"=="" (
    set COMFYUI_PATH=%~dp0..\ComfyUI
)
echo [run_server] COMFYUI_PATH=%COMFYUI_PATH%

REM ---- 2. Where the model weights live ----
set WEIGHTS=%~dp0flux-2-klein-9b-fp8.safetensors
echo [run_server] weights=%WEIGHTS%

REM ---- 3. Network bind ----
if "%PORT%"=="" set PORT=7777
if "%BIND%"=="" set BIND=0.0.0.0
echo [run_server] listening on %BIND%:%PORT%

REM ---- 4. Pick python: prefer local .venv, else fall back to PATH ----
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [run_server] using venv python: %VENV_PY%
) else (
    set "PY=python"
    echo [run_server] WARNING: %VENV_PY% not found, falling back to 'python' on PATH
)

REM ---- 5. Slim-load via N_BLOCKS env var (optional; defaults to full load) ----
if not "%N_BLOCKS%"=="" (
    set "N_BLOCKS_ARG=--n-blocks %N_BLOCKS%"
    echo [run_server] slim load: --n-blocks %N_BLOCKS%
) else (
    set "N_BLOCKS_ARG="
    echo [run_server] full load (N_BLOCKS env var not set)
)

REM ---- 6. Launch ----
"%PY%" -u "%~dp0mesh_server.py" --weights "%WEIGHTS%" --port %PORT% --bind %BIND% --device cuda:0 --dtype bfloat16 %N_BLOCKS_ARG%
