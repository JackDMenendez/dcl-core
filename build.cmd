@echo off
REM Build / test entry-point from the repository root.
REM
REM Usage:
REM   build.cmd            : full build: env -> lint -> tests (default)
REM   build.cmd tests      : run pytest
REM   build.cmd tests-fast : pytest, skip slow tests
REM   build.cmd lint       : ruff
REM   build.cmd typecheck  : mypy
REM   build.cmd coverage   : pytest with coverage
REM   build.cmd build      : produce sdist + wheel in dist/
REM   build.cmd docs       : build docs
REM   build.cmd clean      : remove build artefacts
REM
REM Requires GNU Make >= 4.3 and Python 3.11+.
REM On Windows, run from an MSYS2 UCRT64 shell or a Command Prompt
REM with GNU Make on PATH (e.g. via `pacman -S make` in MSYS2).

setlocal

cd /d "%~dp0"

set "target=%~1"
if "%target%"=="" set "target=all"

where make >nul 2>nul
if errorlevel 1 (
    echo build.cmd: GNU Make is required but not found in PATH. 1>&2
    exit /b 1
)

make %target%
exit /b %errorlevel%
