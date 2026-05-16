@echo off
REM Launcher for the comfyui-mesh LTX back-half server.
REM
REM This is the "default" headless launcher for LTX 2.3 -- uses whichever
REM GPU CUDA picks (usually GPU 0). For an explicit GPU choice on a
REM multi-GPU host, use:
REM     run_server_ltx_gpu0.bat   (pins to physical GPU 0)
REM     run_server_ltx_gpu1.bat   (pins to physical GPU 1)
REM
REM For the GUI version (recommended for first run), use:
REM     run_server_ltx_gui.bat

setlocal

REM ============================================================
REM  EDIT THIS: how many of the LAST transformer_blocks to load.
REM  Must match the Icarus LTX node's `n_blocks_remote` setting.
REM  Leave at 0 to load the full back-half model.
REM ============================================================
set N_BLOCKS=8

REM ============================================================
REM  Optional primary LoRA -- leave LORA empty to skip. If set,
REM  the LoRA is applied to the slim back-half model server-side.
REM  Use this slot for character / style LoRAs.
REM  Examples:
REM     set LORA=
REM     set LORA=loras\my_style.safetensors
REM     set LORA=C:\path\to\my_lora.safetensors
REM ============================================================
set LORA=
set LORA_STRENGTH=1.0

REM ============================================================
REM  Optional second LoRA -- intended for the LTX 2.3 Distilled LoRA
REM  (which most users want stacked on the base model). Default
REM  strength 0.5 matches the typical workflow value.
REM ============================================================
set LORA2=
set LORA2_STRENGTH=0.5

REM ---- 1. Where to find ComfyUI's Python sources ----
if "%COMFYUI_PATH%"=="" (
    REM Default: ComfyUI lives inside this server folder (install.bat clones it here).
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
