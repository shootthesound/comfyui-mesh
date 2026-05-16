@echo off
REM ============================================================
REM  comfyui-mesh server -- update the bundled ComfyUI clone
REM ============================================================
REM
REM  What this does:
REM    1. git pull on ..\ComfyUI (the clone install.bat created)
REM    2. Re-installs ComfyUI's requirements (picks up any new deps)
REM    3. Re-installs the server's own extras (cheap, idempotent)
REM
REM  CUDA torch is NOT touched here -- install.bat installed cu128
REM  wheels and they don't change unless you intentionally upgrade
REM  the card or a new ComfyUI version requires a newer torch.
REM  Re-run install.bat for that.
REM
REM  Run this whenever:
REM    - You want bug fixes / new features from upstream ComfyUI
REM    - The client side complains about ComfyUI version mismatch
REM      (the fp8 detection + FLUX implementation evolve over time;
REM      mismatched ComfyUI versions between client and server is
REM      the most common silent-correctness gotcha)
REM ============================================================

setlocal

set "COMFY_DIR=%~dp0..\ComfyUI"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%COMFY_DIR%\.git" (
    echo [update] no ComfyUI git repo at %COMFY_DIR%
    echo [update] run install.bat first to set things up.
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo [update] no .venv at %~dp0.venv
    echo [update] run install.bat first to set things up.
    exit /b 1
)

where git >nul 2>&1
if errorlevel 1 (
    echo [update] ERROR: git not found on PATH.
    echo            Install Git for Windows ^(https://git-scm.com/^) or
    echo            update %COMFY_DIR% manually with your git client.
    exit /b 1
)

echo.
echo ============================================================
echo  Updating ComfyUI in %COMFY_DIR%
echo ============================================================
echo.

echo [1/3] git pull ...
pushd "%COMFY_DIR%"
git pull
if errorlevel 1 (
    echo [update] git pull failed.
    popd
    exit /b 1
)
popd

echo.
echo [2/3] re-installing ComfyUI requirements ^(picks up new deps^) ...
"%VENV_PY%" -m pip install -r "%COMFY_DIR%\requirements.txt"
if errorlevel 1 (
    echo [update] ComfyUI requirements install failed.
    exit /b 1
)

echo.
echo [3/3] re-installing server requirements ...
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt" >nul
if errorlevel 1 (
    echo [update] server requirements install failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Update complete. Restart the server to pick up the new code.
echo ============================================================
echo.

endlocal
