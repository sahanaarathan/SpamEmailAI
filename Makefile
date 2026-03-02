.PHONY: help install install-dev train evaluate test lint format run-api run-ui docker-up docker-down clean

PYTHON := python
PYTEST := pytest
BLACK  := black
ISORT  := isort
FLAKE8 := flake8

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────────────

install:  ## Install production dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install  ## Install dev dependencies and pre-commit hooks
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "Dev environment ready!"

# ─── ML Pipeline ──────────────────────────────────────────────────────────────

train:  ## Train the model (saves to models/)
	$(PYTHON) scripts/train.py

evaluate:  ## Evaluate the saved model on test data
	$(PYTHON) scripts/evaluate.py

# ─── Testing ──────────────────────────────────────────────────────────────────

test:  ## Run all tests
	$(PYTEST) tests/ -v

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# ─── Code Quality ─────────────────────────────────────────────────────────────

lint:  ## Run all linters
	$(BLACK) --check src/ tests/ scripts/
	$(ISORT) --check-only src/ tests/ scripts/
	$(FLAKE8) src/ tests/ scripts/ --max-line-length=100 --ignore=E203,W503

format:  ## Auto-format code
	$(BLACK) src/ tests/ scripts/
	$(ISORT) src/ tests/ scripts/

# ─── Run Locally ──────────────────────────────────────────────────────────────

run-api:  ## Start the FastAPI backend (requires trained model)
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:  ## Start the Streamlit UI in direct mode
	streamlit run src/ui/app.py

run-ui-api:  ## Start the Streamlit UI in API mode (requires API running)
	USE_API=true API_URL=http://localhost:8000 streamlit run src/ui/app.py

# ─── Docker ───────────────────────────────────────────────────────────────────

docker-train:  ## Run model training in Docker
	docker compose --profile train up train

docker-up:  ## Build and start all services (API + UI)
	docker compose up --build -d

docker-down:  ## Stop all services
	docker compose down

docker-logs:  ## Tail logs from all services
	docker compose logs -f

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:  ## Remove generated files (models, logs, cache)
	rm -rf models/*.joblib logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
