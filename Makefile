# AgentBeats Makefile
# Common commands for development and deployment

.PHONY: help install install-dev test lint format clean run

help:  ## Show this help message
	@echo "AgentBeats - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest test_refactored_architecture.py -v

test-all:  ## Run all tests with coverage
	pytest -v --cov=src --cov-report=term-missing

lint:  ## Run linters (ruff, mypy)
	ruff check src/
	mypy src/ --ignore-missing-imports

format:  ## Format code with black and isort
	black src/
	isort src/

format-check:  ## Check code formatting without changes
	black --check src/
	isort --check src/

clean:  ## Clean up temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/

run:  ## Run the complete system
	python main.py launch-refactored

run-mcp:  ## Run MCP server only
	python main.py mcp

run-purple:  ## Run purple agent only
	python main.py purple

run-green:  ## Run green agent only
	python main.py green --mcp http://localhost:9006

test-quick:  ## Run quick test of purple agent
	python main.py test-purple

check: lint test  ## Run linters and tests

setup: install-dev  ## Initial setup for development
	@echo "✓ Dependencies installed"
	@echo "Next steps:"
	@echo "  1. Set OPENAI_API_KEY in .env file"
	@echo "  2. Run 'make test' to verify installation"
	@echo "  3. Run 'make run' to start the system"

all: clean format lint test  ## Clean, format, lint, and test

.DEFAULT_GOAL := help

