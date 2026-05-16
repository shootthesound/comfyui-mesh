@echo off
REM Launcher: comfyui-mesh LTX back-half server on CPU (system RAM only).
REM
REM CPU equivalent of run_server_ltx.bat. With CUDA hidden from this
REM process, PyTorch falls back to CPU -- the model loads into system
REM RAM and the back-half forward pass runs on CPU cores.
REM
REM Honest expectations:
REM   - SLOW. LTX-AV blocks are heavier than FLUX's; expect minutes
REM     per timestep on a desktop CPU.
REM   - 64+ GB system RAM recommended (the 22B model is big).
REM   - fp8 weights might not load on CPU -- ComfyUI's fp8 ops are
REM     CUDA-tuned. If you hit a kernel-not-found error, you need a
REM     bf16 variant of the checkpoint; point WEIGHTS at it.
REM   - codec_mode on the Icarus LTX node MUST be 'raw'. NVENC is
REM     GPU silicon -- there's nothing to drive it on CPU.

setlocal

REM ============================================================
REM  EDIT THIS: how many of the LAST transformer_blocks to load.
REM  Must match the Icarus LTX node's `n_blocks_remote` setting.
REM  Leave at 0 to load the full back-half model.
REM ============================================================
set N_BLOCKS=11

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

REM Hide all CUDA devices so the server can't accidentally pick a GPU.
set CUDA_VISIBLE_DEVICES=
echo [run_server_ltx] CUDA hidden -- running on CPU / system RAM

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

REM ---- 7. Launch with --device cpu ----
"%PY%" -u "%~dp0mesh_server_ltx.py" --weights "%WEIGHTS%" --port %PORT% --bind %BIND% --device cpu --dtype bfloat16 %N_BLOCKS_ARG% %LORA_ARGS%
