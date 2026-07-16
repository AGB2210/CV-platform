@echo off
REM ===========================================================================
REM  CV Platform - one-click start
REM
REM  Double-click this file, or run `start.bat` from a terminal.
REM
REM  This is deliberately a thin shim. All the real work (dependency install,
REM  process supervision, health checks, clean shutdown) lives in
REM  scripts\dev.py, because batch is poorly suited to any of it. The only jobs
REM  here are: find Python, run the script, and keep the window open long
REM  enough to read an error if it fails.
REM ===========================================================================

setlocal

REM %~dp0 is this file's own directory, with a trailing backslash. Using it
REM means double-clicking works regardless of what the shell thinks the current
REM directory is (double-click can start you in C:\Windows\System32).
cd /d "%~dp0"

REM Locate Python. `py` (the Windows launcher) is preferred when present -- it
REM resolves the newest install even if PATH points at an old one.
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PY=py -3"
    goto :run
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PY=python"
    goto :run
)

echo.
echo   [X] Python was not found on your PATH.
echo.
echo   Install Python 3.10 or newer from https://www.python.org/downloads/
echo   During install, tick "Add python.exe to PATH", then open a NEW terminal.
echo.
pause
exit /b 1

:run
%PY% scripts\dev.py %*
set "EXITCODE=%ERRORLEVEL%"

REM Only pause on failure. On a clean Ctrl+C exit the window just closes, which
REM is what you want; on a crash it stays open so the error is readable instead
REM of flashing past.
if not "%EXITCODE%"=="0" (
    echo.
    echo   Exited with code %EXITCODE%.
    pause
)

endlocal
exit /b %EXITCODE%
