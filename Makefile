# Makefile — MIGUEL_DAVID_LLM + OpenEvolve Integration
#
# Unified command wrappers for common operations.
# All targets auto-read .env environment variables.

.DEFAULT_GOAL := help

# OpenEvolve evolution parameters (override via CLI, e.g.: make evolve ITERS=20 TASKS=3)
ITERS ?= 10
TASKS ?= 5
OUTPUT ?= config/openevolve_output

# ============================================================================
# Environment
# ============================================================================
.PHONY: install
install:  ## Sync dependencies (uv sync)
	uv sync

# ============================================================================
# Testing
# ============================================================================
.PHONY: test-mock
test-mock:  ## Mock mode integration test (zero cost)
	uv run --env-file .env python run_smoke.py --mock

.PHONY: test
test:  ## Real LLM integration test (all tasks)
	uv run --env-file .env python run_smoke.py

.PHONY: smoke
smoke:  ## Smoke test (1 task only)
	uv run --env-file .env python run_smoke.py --tasks 1

.PHONY: pipeline
pipeline:  ## Run full pipeline (all tasks, real LLM)
	uv run --env-file .env python run_pipeline.py

# ============================================================================
# OpenEvolve Evolution
# ============================================================================
.PHONY: evolve
evolve:  ## Launch OpenEvolve evolution (adjustable: ITERS=N TASKS=N)
	OPENEVOLVE_NUM_TASKS=$(TASKS) \
	uv run --env-file .env python -m openevolve.cli \
	    config/agents_evolving.yaml \
	    openevolve_evaluator.py \
	    --config config/openevolve_config.yaml \
	    --output $(OUTPUT) \
	    --iterations $(ITERS)

.PHONY: evolve-resume
evolve-resume:  ## Resume from checkpoint (specify CHECKPOINT=path)
	@if [ -z "$(CHECKPOINT)" ]; then \
	    echo "ERROR: Must specify CHECKPOINT, e.g.: make evolve-resume CHECKPOINT=config/openevolve_output/checkpoints/checkpoint_10"; \
	    exit 1; \
	fi
	OPENEVOLVE_NUM_TASKS=$(TASKS) \
	uv run --env-file .env python -m openevolve.cli \
	    config/agents_evolving.yaml \
	    openevolve_evaluator.py \
	    --config config/openevolve_config.yaml \
	    --output $(OUTPUT) \
	    --checkpoint $(CHECKPOINT) \
	    --iterations $(ITERS)

.PHONY: evolve-test
evolve-test:  ## Local integration test evaluator (no evolution)
	uv run --env-file .env python openevolve_evaluator.py

# ============================================================================
# Maintenance
# ============================================================================
.PHONY: clean
clean:  ## Clean __pycache__ and .pyc files
	@if exist __pycache__ (for /d /r %%d in (__pycache__) do @if not "%%d"==".venv" rd /s /q "%%d") 2>nul || echo cleaned

.PHONY: clean-output
clean-output:  ## Delete OpenEvolve output (WARNING: deletes checkpoints!)
	@echo WARNING: This will delete $(OUTPUT) and all checkpoints.
	@if exist $(OUTPUT) rd /s /q $(OUTPUT)

# ============================================================================
# Help
# ============================================================================
.PHONY: help
help:  ## List all targets
	@echo Available commands:
	@findstr /R "^[a-zA-Z_-]*:.*##" Makefile
