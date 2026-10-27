# Module 2 - real vs decoy filter
#   make         filter test (dev, with TPR/FPR) and eval
#   make score   final eval TPR/FPR, once all eval frames are labelled

PY ?= .venv/bin/python

.PHONY: module2 score

module2:
	$(PY) src/decoy_filter.py --split test --labels labels/chris_test
	$(PY) src/decoy_filter.py --split eval

score:
	$(PY) src/decoy_filter.py --split eval --labels labels/eval_first64 labels/eval_last63
