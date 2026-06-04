.DEFAULT_GOAL := help
SHELL         := bash
PYTHON        := python3

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR  := src/hmats
TEST_DIR := tests

# ── Colours ───────────────────────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m

.PHONY: help install install-dev sync lint format typecheck test test-fast \
        clean clean-all notebook run

# ── Help ──────────────────────────────────────────────────────────────────────
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BOLD)$(CYAN)Hybrid Multi-Agent Trading System$(RESET)\n\n"} \
	     /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────
install: ## Install production dependencies with uv
	uv sync --no-dev

install-dev: ## Install all dependencies including dev tools
	uv sync --extra dev

sync: ## Re-sync environment after pyproject.toml changes
	uv sync

# ── Code quality ──────────────────────────────────────────────────────────────
lint: ## Run ruff linter (check only)
	uv run ruff check $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Run ruff linter and auto-fix issues
	uv run ruff check --fix $(SRC_DIR) $(TEST_DIR)

format: ## Format code with ruff
	uv run ruff format $(SRC_DIR) $(TEST_DIR)

format-check: ## Check formatting without applying changes
	uv run ruff format --check $(SRC_DIR) $(TEST_DIR)

typecheck: ## Run mypy static type checking
	uv run mypy $(SRC_DIR)

check: lint format-check typecheck ## Run all code quality checks (no fixes)

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run full test suite with coverage
	uv run pytest $(TEST_DIR)

test-fast: ## Run tests without coverage (faster feedback)
	uv run pytest $(TEST_DIR) --no-cov -x

test-watch: ## Re-run tests on file change (requires pytest-watch)
	uv run ptw $(TEST_DIR) -- --no-cov -q

# ── Notebook ──────────────────────────────────────────────────────────────────
notebook: ## Launch Jupyter Lab
	uv run jupyter lab

# ── Application ───────────────────────────────────────────────────────────────
run: ## Run the CLI entry point (pass ARGS="..." to supply arguments)
	uv run trading $(ARGS)

# ── Clean ─────────────────────────────────────────────────────────────────────
clean: ## Remove build artefacts and cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"         -delete
	rm -rf dist/ build/ htmlcov/ .coverage coverage.xml

clean-all: clean ## Remove everything including the virtual environment
	rm -rf .venv


# make nbconvert NB=src/hmats/notebooks/06_lgbm_agent.ipynb
nbconvert: ## Convert notebook to Python script (pass NB="path/to/notebook.ipynb")
	uv run jupyter nbconvert --to script $(NB) --output-dir $(dir $(NB))