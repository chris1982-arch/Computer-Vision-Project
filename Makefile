# Module 2 - real vs decoy filter (setup: see README, Module 2 "Reproduce")
#   make         fit on dev (test), filter eval, draw figures
#   make score   final eval TPR/FPR, run once when all eval frames are labelled

PY ?= .venv/bin/python

.PHONY: module2 score

module2:
	$(PY) src/decoy_filter.py --split test --fit labels/chris_test --labels labels/chris_test
	$(PY) src/decoy_filter.py --split eval
	$(PY) src/plot_filter.py

score:
	$(PY) src/decoy_filter.py --split eval --labels labels/eval_first64 labels/eval_last63
