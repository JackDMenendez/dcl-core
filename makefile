# Core-software template -- root build
# ----------------------------------------------------------------------------
# Requirements
#   GNU Make >= 4.3   (uses grouped-target `&:` syntax in experiments/)
#   Python 3.11+ + venv
#
#   Run from an MSYS2 UCRT64 shell on Windows. The stock Windows GNU Make
#   port (3.81) is too old. Confirm with: `make --version` -> >= 4.3.
# ----------------------------------------------------------------------------
# Targets
#   all          env -> lint -> tests        (default)
#   env          create .venv, install package in editable mode + dev deps
#   tests        run pytest against tests/
#   tests-fast   pytest with -m 'not slow'
#   tests-gpu    pytest with -m gpu (requires CUDA-capable host)
#   lint         ruff check (style + lint)
#   typecheck    mypy on src/dcl_core/
#   coverage     pytest with coverage; report to terminal + htmlcov/
#   build        python -m build -- produces sdist + wheel in dist/
#   experiments  delegate to experiments/makefile (see WARNING below)
#   docs         build docs (mkdocs) -- requires `pip install -e .[docs]`
#   clean        remove build/, dist/, *.egg-info/, caches
#   clean-env    also remove .venv
#   help         print this list
#
# Long experiments -- prefer running them one at a time
#   `make experiments` runs every target in experiments/makefile.
#   Some experiments take hours. Prefer named targets:
#       make -C experiments exp_00_hop_drift
# ----------------------------------------------------------------------------

RELATIVE_PATH :=
include common.mak

# --- Phony list -------------------------------------------------------------
.PHONY: all help env tests tests-fast tests-gpu lint typecheck coverage \
        build experiments docs clean clean-env

all: env lint tests
	@echo "================ Build Complete ================"

help:
	@sed -n '1,40p' makefile

# --- Environment ------------------------------------------------------------
$(VENV):
	python -m venv $(VENV)

# Install the package in editable mode with dev extras, plus the small
# direct-make-target deps from virtual-env-requirements.txt.
$(VENV)/touchfile: $(VENV) pyproject.toml virtual-env-requirements.txt
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r virtual-env-requirements.txt
	$(PIP) install -e .[dev]
	touch $(VENV)/touchfile

env: $(VENV)/touchfile

# --- Tests ------------------------------------------------------------------
tests: $(VENV)/touchfile
	$(PYTHON) -m pytest tests -v

tests-fast: $(VENV)/touchfile
	$(PYTHON) -m pytest tests -v -m "not slow"

tests-gpu: $(VENV)/touchfile
	$(PYTHON) -m pytest tests -v -m gpu

coverage: $(VENV)/touchfile
	$(PYTHON) -m pytest tests --cov=dcl_core --cov-report=term --cov-report=html

# --- Lint / typecheck -------------------------------------------------------
lint: $(VENV)/touchfile
	$(PYTHON) -m ruff check src tests experiments

typecheck: $(VENV)/touchfile
	$(PYTHON) -m mypy src/dcl_core

# --- Build artefacts --------------------------------------------------------
build: $(VENV)/touchfile
	$(PYTHON) -m build

# --- Experiments ------------------------------------------------------------
experiments:
	@echo "WARNING: 'make experiments' runs every target in experiments/makefile."
	@echo "         Some take hours."
	@echo "         Prefer 'make -C experiments <target>' for individual runs."
	$(MAKE) -C experiments all

# --- Docs -------------------------------------------------------------------
docs: $(VENV)/touchfile
	$(PIP) install -e .[docs]
	$(PYTHON) -m mkdocs build

# --- Clean ------------------------------------------------------------------
clean:
	-rm -rf $(build_dir) dist
	-rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	-find src -name "*.egg-info" -type d -exec rm -rf {} +
	-find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "================ Clean Complete ================"

clean-env:
	-rm -rf $(VENV)
	@echo "================ clean-env Complete ================"
