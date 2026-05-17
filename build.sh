#!/usr/bin/env bash
# Build / test entry-point from the repository root.
#
# Usage:
#   ./build.sh            # full build: env -> lint -> tests (default)
#   ./build.sh tests      # run pytest
#   ./build.sh tests-fast # pytest, skip slow tests
#   ./build.sh lint       # ruff
#   ./build.sh typecheck  # mypy
#   ./build.sh coverage   # pytest with coverage
#   ./build.sh build      # produce sdist + wheel in dist/
#   ./build.sh docs       # build docs
#   ./build.sh clean      # remove build artefacts
#
# Requires GNU Make >= 4.3 and Python 3.11+.
# On Windows, run from an MSYS2 UCRT64 shell.

set -euo pipefail

cd "$(dirname "$0")"

target="${1:-all}"

if ! command -v make >/dev/null 2>&1; then
    echo "build.sh: GNU Make is required but not found in PATH." >&2
    exit 1
fi

exec make "$target"
