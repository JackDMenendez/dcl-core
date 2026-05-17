# Shared make variables. Included from the root makefile and from
# experiments/makefile (via RELATIVE_PATH=../).

# Get the absolute path of the current Makefile
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(dir $(MKFILE_PATH))

# Virtual environment.
VENV := $(RELATIVE_PATH).venv
ARGS ?= -u
PYTHON = "$(VENV)/Scripts/python" $(ARGS)
PIP    = "$(VENV)/Scripts/pip.exe"

# Build directories.
build_dir := $(RELATIVE_PATH)build
data_dir  := $(RELATIVE_PATH)data

# For substituting spaces in file names, if needed.
nullstring :=
space := $(nullstring) $(nullstring)
