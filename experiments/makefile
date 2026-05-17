# experiments/makefile -- per-experiment targets.
#
# Invoked from the repo root either via:
#   make experiments              (runs every target in EXPERIMENTS, slow)
#   make -C experiments exp_00_hop_drift
#
# Add a new experiment by:
#   1. dropping exp_NN_<name>.py and exp_NN_<name>.md alongside this file
#   2. appending the target name to EXPERIMENTS
#   3. adding a target rule below
#   4. adding a row to EXPERIMENTS.md

RELATIVE_PATH := ../
include $(RELATIVE_PATH)common.mak

EXPERIMENTS := exp_00_hop_drift

.PHONY: all $(EXPERIMENTS)

all: $(EXPERIMENTS)

# --- Individual experiments -------------------------------------------------
exp_00_hop_drift:
	$(PYTHON) exp_00_hop_drift.py
