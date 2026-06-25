.PHONY: setup pipeline train serve test lint format clean drift mlflow

setup:
	pip install -r requirements-dev.txt
	dvc init --no-scm || true
	pre-commit install

pipeline:
	dvc repro

train:
	dvc repro train evaluate

mlflow:
	docker compose up -d mlflow

serve:
	docker compose up --build

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ api/
	mypy src/ api/

format:
	ruff format src/ tests/ api/
	ruff check --fix src/ tests/ api/

drift:
	python src/monitoring/detect_drift.py --simulate-drift

clean:
	dvc gc -w --force || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
