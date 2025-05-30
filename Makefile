.PHONY: help install test lint format docs clean build deploy

help:
	@echo "MICAP - Market Intelligence & Competitor Analysis Platform"
	@echo "Available commands:"
	@echo "  install     - Install dependencies and setup environment"
	@echo "  test        - Run all tests"
	@echo "  lint        - Run code linting and formatting checks"
	@echo "  format      - Format code with black and isort"
	@echo "  docs        - Generate documentation"
	@echo "  clean       - Clean temporary files and caches"
	@echo "  build       - Build Docker image"
	@echo "  deploy      - Deploy using docker-compose"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	python -m spacy download en_core_web_sm
	pre-commit install

test:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

lint:
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	pylint src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

docs:
	python scripts/check_docstrings.py --fail-under=80

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf htmlcov/ dist/ build/ .mypy_cache/

build:
	docker build -t micap:latest .

deploy:
	docker-compose up -d 