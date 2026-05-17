@echo off
REM Launcher: LTX-AV server-side smoke test, with logging to a timestamped
REM text file in this folder so you can inspect / share the result later.
REM
REM Loads the LTX-AV slim back-half via load_ltx_av(), builds a synthetic
REM block_wrap payload, runs forward_back_half_ltx() once, asserts the
REM output (vx, ax) shapes / dtypes are right and contain no NaN/Inf.
REM
REM Catches: slim-load fp8 metadata remap regressions, LTXAV variant
REM detection drift, forward-pass kwarg mismatches against newer ComfyUI
REM versions of comfy.ldm.lightricks.av_model.
REM
REM Expected wall-clock: ~30-60s (model load dominates).

setlocal enabledelayedexpansion

REM ============================================================
REM  EDIT THIS: number of back-half transformer_blocks to slim-load
REM  for the test. Smaller = faster load, faster forward.
REM  Doesn't need to match your live server's --n-blocks.
REM ============================================================
set N_BLOCKS=8

REM ============================================================
REM  EDIT THIS: which LTX safetensors to load.
REM  Default assumes it lives in this folder.
REM ============================================================
set WEIGHTS=%~dp0ltx-2.3-22b-dev-fp8.safetensors

REM ---- 1. Where to find ComfyUI's Python sources ----
if "%COMFYUI_PATH%"=="" (
    set COMFYUI_PATH=%~dp0ComfyUI
)
echo [run_smoke] COMFYUI_PATH=%COMFYUI_PATH%

REM ---- 2. Pick python: prefer local .venv, else PATH ----
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [run_smoke] using venv python: %VENV_PY%
) else (
    set "PY=python"
    echo [run_smoke] WARNING: %VENV_PY% not found, falling back to 'python' on PATH
)

REM ---- 3. Build a timestamped log filename ----
REM Format: smoke_test_ltx_YYYYMMDD_HHMMSS.log (no spaces/colons that
REM Windows filesystems hate). %DATE% / %TIME% format is locale-
REM dependent so we extract via wmic which is always YYYYMMDDHHMMSS.NN.
for /f %%i in ('wmic os get localdatetime ^| findstr /r "^[0-9]"') do set DT=%%i
set "STAMP=%DT:~0,8%_%DT:~8,6%"
set "LOG=%~dp0smoke_test_ltx_%STAMP%.log"
echo [run_smoke] logging to: %LOG%
echo.

REM ---- 4. Sanity: weights present ----
if not exist "%WEIGHTS%" (
    echo.
    echo [run_smoke] ERROR: weights file not found:
    echo   %WEIGHTS%
    echo.
    echo Either drop the .safetensors file into this folder, or edit the
    echo WEIGHTS= line near the top of this .bat to point at it.
    pause
    exit /b 2
)

REM ---- 5. Launch the smoke test, capturing output to the log file
REM       AND showing it live on the console via PowerShell's Tee-Object.
REM       PowerShell is always present on Windows so this needs no
REM       extra install. The %errorlevel% from the python process gets
REM       lost through the pipe, so we re-read it from the captured
REM       log's last line by grepping for the SMOKE TEST footer.
REM ============================================================
"%PY%" -u "%~dp0smoke_test_server_ltx.py" --weights "%WEIGHTS%" --n-blocks %N_BLOCKS% --device cuda:0 --dtype bfloat16 2>&1 | "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -Command "$input | Tee-Object -FilePath '%LOG%'"

echo.
echo [run_smoke] done. Full log saved at:
echo   %LOG%
echo.
findstr /c:"SMOKE TEST PASSED" "%LOG%" >nul
if %errorlevel%==0 (
    echo [run_smoke] RESULT: PASSED
    set "RC=0"
) else (
    echo [run_smoke] RESULT: FAILED ^(see log for FAIL lines^)
    set "RC=1"
)
echo.
pause
exit /b %RC%
