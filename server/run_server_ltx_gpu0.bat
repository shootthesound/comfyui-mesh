@echo off
REM Launcher: comfyui-mesh LTX back-half server pinned to GPU 0.
REM
REM Use this when GPU 0 is your second card (the one NOT running ComfyUI).
REM For a two-machine setup where this machine only has one GPU, use this.

setlocal

REM ============================================================
REM  EDIT THIS: how many of the LAST transformer_blocks to load.
REM  Must match the Icarus LTX node's `n_blocks_remote` setting.
REM  Leave at 0 to load the full back-half model.
REM ============================================================
set N_BLOCKS=8

REM ============================================================
REM  Optional primary LoRA -- leave LORA empty to skip.
REM ============================================================
set LORA=
set LORA_STRENGTH=1.0

REM ============================================================
REM  Optional second LoRA (LTX 2.3 Distilled LoRA, typical 0.5).
REM ============================================================
set LORA2=
set LORA2_STRENGTH=0.5

REM Pin this process to the first GPU. After this line the server only
REM sees one card and addresses it as cuda:0.
set CUDA_VISIBLE_DEVICES=0
echo [run_server_ltx] pinned to physical GPU 0 (CUDA_VISIBLE_DEVICES=0)

REM ---- 1. Where to find ComfyUI's Python sources ----
if "%COMFYUI_PATH%"=="" (
    set COMFYUI_PATH=%~dp0ComfyUI
)
echo [run_server_ltx] COMFYUI_PATH=%COMFYUI_PATH%

REM ---- 2. Where the model weights live ----
set WEIGHTS=%~dp0ltx-2.3-22b-dev-fp8.safetensors
echo [run_server_ltx] weights=%WEIGHTS%

REM ---- 3. Network bind ----
if "%PORT%"=="" set PORT=7777
if "%BIND%"=="" set BIND=0.0.0.0
echo [run_server_ltx] listening on %BIND%:%PORT%

REM ---- 4. Pick python: prefer local .venv, else fall back to PATH ----
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [run_server_ltx] using venv python: %VENV_PY%
) else (
    set "PY=python"
    echo [run_server_ltx] WARNING: %VENV_PY% not found, falling back to 'python' on PATH
)

REM ---- 5. Translate N_BLOCKS into CLI arg ----
if "%N_BLOCKS%"=="0" (
    set "N_BLOCKS_ARG="
    echo [run_server_ltx] full load (N_BLOCKS=0)
) else (
    set "N_BLOCKS_ARG=--n-blocks %N_BLOCKS%"
    echo [run_server_ltx] slim load: --n-blocks %N_BLOCKS%
)

REM ---- 6. Translate LORA / LORA2 into CLI args ----
set "LORA_ARGS="
if not "%LORA%"=="" (
    set "LORA_ARGS=--lora "%LORA%" --lora-strength %LORA_STRENGTH%"
    echo [run_server_ltx] primary LoRA: %LORA% strength=%LORA_STRENGTH%
)
if not "%LORA2%"=="" (
    set "LORA_ARGS=%LORA_ARGS% --lora2 "%LORA2%" --lora2-strength %LORA2_STRENGTH%"
    echo [run_server_ltx] distill LoRA: %LORA2% strength=%LORA2_STRENGTH%
)

REM ---- 7. Launch ----
"%PY%" -u "%~dp0mesh_server_ltx.py" --weights "%WEIGHTS%" --port %PORT% --bind %BIND% --device cuda:0 --dtype bfloat16 %N_BLOCKS_ARG% %LORA_ARGS%
