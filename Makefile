# Module 2 - real vs decoy filter
#   make           filter test (dev, with TPR/FPR) and eval
#   make score     final eval TPR/FPR, once all eval frames are labelled
# Module 3 - 3D localization and evaluation (run after Module 2)
#   make localize  3D positions -> outputs/localized_{test,eval}.json
#   make evaluate  baseline vs filter TPR/FPR -> outputs/evaluation_{test,eval}.txt
#   make figures   plots and example frames -> outputs/figures/

PY ?= .venv/bin/python

.PHONY: module2 score localize evaluate figures

module2:
	$(PY) src/decoy_filter.py --split test --labels labels/chris_test
	$(PY) src/decoy_filter.py --split eval

score:
	$(PY) src/decoy_filter.py --split eval --labels labels/eval_first64 labels/eval_last63

localize:
	$(PY) src/localize_3d.py --split test
	$(PY) src/localize_3d.py --split eval

evaluate:
	$(PY) src/evaluate.py --split test
	$(PY) src/evaluate.py --split eval

figures: localize
	$(PY) src/visualize_results.py
	$(PY) src/plot_3d_example.py
