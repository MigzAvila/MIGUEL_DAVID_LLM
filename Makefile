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

.PHONY: test-best
test-best:  ## Run integration test using the best evolved agents and tasks
	OPENEVOLVE_AGENTS_YAML=config/openevolve_output/checkpoints/checkpoint_50/best_program.yaml \
	OPENEVOLVE_TASKS_YAML=config/openevolve_tasks_output/checkpoints/checkpoint_50/best_program.yaml \
	uv run --env-file .env python run_smoke.py $(if $(TASKS),--tasks $(TASKS))

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

.PHONY: evolve-v2
evolve-v2:  ## Launch Phase 2 OpenEvolve (builds on best agents & best tasks)
	OPENEVOLVE_NUM_TASKS=$(TASKS) \
	OPENEVOLVE_TASKS_YAML=config/openevolve_tasks_output/checkpoints/checkpoint_50/best_program.yaml \
	uv run --env-file .env python -m openevolve.cli \
	    config/openevolve_output/checkpoints/checkpoint_50/best_program.yaml \
	    openevolve_evaluator.py \
	    --config config/openevolve_config.yaml \
	    --output config/openevolve_final_phase_output \
	    --iterations $(ITERS)

.PHONY: evolve-tasks
evolve-tasks:  ## Launch OpenEvolve evolution on tasks.yaml
	OPENEVOLVE_NUM_TASKS=$(TASKS) \
	uv run --env-file .env python -m openevolve.cli \
	    config/tasks_evolving.yaml \
	    openevolve_evaluator_tasks.py \
	    --config config/openevolve_task_config.yaml \
	    --output config/openevolve_tasks_output \
	    --iterations $(ITERS)

.PHONY: evolve-analyst
evolve-analyst:  ## Launch OpenEvolve targeting only star accuracy
	OPENEVOLVE_NUM_TASKS=$(TASKS) OPENEVOLVE_TARGET_METRIC=preference_estimation \
	uv run --env-file .env python -m openevolve.cli \
	    config/agents_evolving.yaml \
	    openevolve_evaluator.py \
	    --config config/openevolve_config.yaml \
	    --output config/openevolve_analyst_output \
	    --iterations $(ITERS)

.PHONY: evolve-simulator
evolve-simulator:  ## Launch OpenEvolve targeting only text review generation quality
	OPENEVOLVE_NUM_TASKS=$(TASKS) OPENEVOLVE_TARGET_METRIC=review_generation \
	uv run --env-file .env python -m openevolve.cli \
	    config/agents_evolving.yaml \
	    openevolve_evaluator.py \
	    --config config/openevolve_config.yaml \
	    --output config/openevolve_simulator_output \
	    --iterations $(ITERS)

.PHONY: evolve-balanced
evolve-balanced:  ## Launch OpenEvolve targeting overall quality (both star rating and review generation)
	OPENEVOLVE_NUM_TASKS=$(TASKS) OPENEVOLVE_TARGET_METRIC=overall_quality \
	uv run --env-file .env python -m openevolve.cli \
	    config/agents_evolving.yaml \
	    openevolve_evaluator.py \
	    --config config/openevolve_config.yaml \
	    --output config/openevolve_balanced_output \
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

.PHONY: visualizer
visualizer:  ## Launch OpenEvolve visualizer for baseline evolution
	uv run python scripts/visualizer.py --path $(OUTPUT)

.PHONY: visualizer-v2
visualizer-v2:  ## Launch OpenEvolve visualizer for the v2 evolution
	uv run python scripts/visualizer.py --path config/openevolve_final_phase_output

.PHONY: visualizer-tasks
visualizer-tasks:  ## Launch OpenEvolve visualizer for task evolution
	uv run python scripts/visualizer.py --path config/openevolve_tasks_output

.PHONY: visualizer-analyst
visualizer-analyst:  ## Launch OpenEvolve visualizer for analyst evolution
	uv run python scripts/visualizer.py --path config/openevolve_analyst_output

.PHONY: visualizer-simulator
visualizer-simulator:  ## Launch OpenEvolve visualizer for simulator evolution
	uv run python scripts/visualizer.py --path config/openevolve_simulator_output

.PHONY: visualizer-balanced
visualizer-balanced:  ## Launch OpenEvolve visualizer for balanced evolution
	uv run python scripts/visualizer.py --path config/openevolve_balanced_output

# ============================================================================
# Maintenance
# ============================================================================
.PHONY: clean
clean:  ## Clean __pycache__ and .pyc files
	find . -type d -name "__pycache__" -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || echo cleaned

.PHONY: clean-output
clean-output:  ## Delete OpenEvolve output (WARNING: deletes checkpoints!)
	@echo "WARNING: This will delete $(OUTPUT) and all checkpoints."
	rm -rf $(OUTPUT)

# ============================================================================
# Help
# ============================================================================
.PHONY: help
help:  ## List all targets
	@echo Available commands:
	@findstr /R "^[a-zA-Z_-]*:.*##" Makefile
