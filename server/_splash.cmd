@echo off
REM Tiny console splash shown during the python.exe + venv + tkinter
REM cold-start gap. The launcher bat starts this in parallel with
REM pythonw; mesh_server_gui.py drops mesh_server_gui_ready.tmp once
REM its window can paint, and this script polls for that file and
REM exits when it appears.
REM
REM Why this exists: on a cold launch, Python + venv site.py + tkinter
REM DLL loads + Defender scan can eat 10-30s before our GUI script
REM runs at all. Without this, the user clicks the launcher and stares
REM at nothing.

title ComfyUI Mesh : Daedalus starting...
mode con cols=64 lines=12 >nul

echo.
echo   Starting ComfyUI Mesh : Daedalus...
echo.
echo   First launch can take 10-30 seconds on Windows
echo   (Python + venv site init + tkinter DLLs + Defender scan).
echo.
echo   This window closes when the main GUI is ready.
echo.

:wait
if exist "%~dp0mesh_server_gui_ready.tmp" goto done
REM Poll every second. Safety: bail after 120 ticks (~2 minutes) so
REM the splash can't dangle forever if the GUI crashed before signaling.
set /a "_ticks=_ticks+1"
if "%_ticks%"=="120" goto done
timeout /t 1 /nobreak >nul 2>nul
goto wait

:done
exit
