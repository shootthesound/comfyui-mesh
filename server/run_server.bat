@echo off
REM Launcher for the comfyui-mesh back-half server on the 4090 machine.
REM
REM This is the "default" launcher -- uses whichever GPU CUDA picks (usually
REM GPU 0). For an explicit GPU choice on a multi-GPU host, use:
REM     run_server_gpu0.bat   (pins to physical GPU 0)
REM     run_server_gpu1.bat   (pins to physical GPU 1)

setlocal

REM ============================================================
REM  EDIT THIS: how many of the LAST double_blocks to load.
REM  Must match the node's `n_blocks_remote` setting.
REM  Leave at 0 to load the full back-half model.
REM ============================================================
set N_BLOCKS=4

REM ============================================================
REM  Optional LoRA -- leave LORA empty to skip. If set, the LoRA
REM  is applied to the slim back-half model server-side.
REM  Examples:
REM     set LORA=
REM     set LORA=loras\my_style.safetensors
REM     set LORA=C:\path\to\my_lora.safetensors
REM ============================================================
set LORA=
set LORA_STRENGTH=1.0

REM ---- 1. Where to find ComfyUI's Python sources ----
if "%COMFYUI_PATH%"=="" (
    REM Default: ComfyUI lives inside this server folder (install.bat clones it here).
    set COMFYUI_PATH=%~dp0ComfyUI
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

REM ---- 5. Translate N_BLOCKS (from the EDIT-ME line at the top) into the CLI arg ----
if "%N_BLOCKS%"=="0" (
    set "N_BLOCKS_ARG="
    echo [run_server] full load (N_BLOCKS=0)
) else (
    set "N_BLOCKS_ARG=--n-blocks %N_BLOCKS%"
    echo [run_server] slim load: --n-blocks %N_BLOCKS%
)

REM ---- 6. Translate LORA into CLI args (if set) ----
if "%LORA%"=="" (
    set "LORA_ARGS="
) else (
    set "LORA_ARGS=--lora "%LORA%" --lora-strength %LORA_STRENGTH%"
    echo [run_server] applying LoRA: %LORA% strength=%LORA_STRENGTH%
)

REM ---- 7. Launch ----
"%PY%" -u "%~dp0mesh_server.py" --weights "%WEIGHTS%" --port %PORT% --bind %BIND% --device cuda:0 --dtype bfloat16 %N_BLOCKS_ARG% %LORA_ARGS%
