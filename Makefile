.PHONY: install lint format typecheck test example build check clean

install:   ## sync the venv with all extras and dev tools
	uv sync --all-extras --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -q

example:
	uv run python examples/workbook_case.py

build:
	uv build

check: lint typecheck test example

clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .hypothesis carbitrage-report.xlsx
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
