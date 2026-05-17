@echo off
REM Create the Python virtual environment and install the package in
REM editable mode along with dev dependencies.
setlocal
cd /d "%~dp0"
make env
exit /b %errorlevel%
