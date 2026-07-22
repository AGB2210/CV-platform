@echo off
REM ===========================================================================
REM  CV Platform - one-click start
REM
REM  Double-click this file, or run `start.bat` from a terminal.
REM
REM  This is deliberately a thin shim. All the real work (create the Python
REM  environment, install dependencies, build the interface if needed, serve
REM  everything, clean shutdown) lives in scripts\app.py, because batch is
REM  poorly suited to any of it. The only jobs here are: find Python, run the
REM  script, and keep the window open long enough to read an error if it fails.
REM
REM  ONE launcher for everyone. The same file a developer runs from source and a
REM  user runs from a downloaded release: it builds the interface when there is
REM  source to build and none is built yet, and otherwise just serves the
REM  interface a release already ships. So "working on it" and "using it" are
REM  the same experience.
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
%PY% scripts\app.py %*
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
