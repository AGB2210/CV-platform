@echo off
REM ===========================================================================
REM  CV Platform - start the app
REM
REM  Double-click this file.
REM
REM  This is the RELEASE entry point: the interface is already built, so this
REM  starts one process (the backend) which also serves the app. Contrast
REM  start.bat, which is for working on the source and runs a second dev server
REM  for the frontend.
REM
REM  First run creates a Python environment and installs dependencies, which
REM  takes a minute or two. After that it starts in seconds.
REM ===========================================================================

setlocal

REM %~dp0 is this file's own directory. Using it means double-clicking works
REM regardless of what the shell thinks the current directory is.
cd /d "%~dp0"

REM Prefer the py launcher: it resolves the newest install even when PATH
REM points at an older one.
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
%PY% scripts\serve.py %*
set "EXITCODE=%ERRORLEVEL%"

REM Only pause on failure. A clean exit just closes the window; a crash stays
REM open so the error is readable instead of flashing past.
if not "%EXITCODE%"=="0" (
    echo.
    echo   Exited with code %EXITCODE%.
    pause
)

endlocal
exit /b %EXITCODE%
