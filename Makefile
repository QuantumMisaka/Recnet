# Core-only engineering entrypoints (no adapter/builder assumptions).
# Override with absolute interpreter paths, e.g.:
#   make check PYTHON=/opt/envs/dpa4/bin/python PIPELINE_PYTHON=/opt/envs/dpa4/bin/python
PYTHON ?= python
PIPELINE_PYTHON ?= $(PYTHON)
LINT_PYTHON ?= $(PYTHON)

.PHONY: check lint install-smoke

check: lint install-smoke

lint:
	"$(LINT_PYTHON)" -m ruff check handlers ccqn utils run_dp_ts.py slabsite.py

install-smoke:
	"$(PYTHON)" scripts/check_core_install.py --pipeline-python "$(PIPELINE_PYTHON)"
