@echo off
REM Launcher: comfyui-mesh back-half server on CPU (system RAM only).
REM
REM This is the "use my system RAM as extra working memory" mode. With
REM CUDA hidden from this process, PyTorch falls back to CPU — the
REM model loads into system RAM and the back-half forward pass runs
REM on CPU cores.
REM
REM Why this is interesting: it proves the architectural decoupling.
REM With codec_mode=raw on the client side, the server's hardware can
REM be ANY PyTorch backend — Nvidia GPU, AMD GPU, Apple Silicon, or
REM just DDR + CPU cores. The wire doesn't know.
REM
REM Honest expectations:
REM   - SLOW. Expect ~30-90 seconds per timestep on a desktop CPU.
REM     For 4-step distilled FLUX.2 Klein, that's 2-6 minutes per
REM     generation just for the back-half work. Not a production
REM     speed; it's an architectural-proof setup.
REM   - 64+ GB system RAM recommended (model is ~18 GB bf16 working
REM     set + scratch).
REM   - fp8 weights might not load on CPU — ComfyUI's fp8 ops are
REM     CUDA-tuned. If you hit a kernel-not-found error, you need a
REM     bf16 variant of the checkpoint; point WEIGHTS at it.
REM   - codec_mode on the client side MUST be 'raw'. NVENC is GPU
REM     silicon — there's nothing to drive it on CPU.

setlocal

REM Hide all CUDA devices so the server can't accidentally pick a GPU.
set CUDA_VISIBLE_DEVICES=
echo [run_server] CUDA hidden — running on CPU / system RAM

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

REM ---- 6. Launch with --device cpu ----
"%PY%" -u "%~dp0mesh_server.py" --weights "%WEIGHTS%" --port %PORT% --bind %BIND% --device cpu --dtype bfloat16 %N_BLOCKS_ARG%
