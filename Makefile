.PHONY: install lint format typecheck test check demo build-vocab fetch-library build-index setup clean

install:
	uv sync --extra agent --extra dev

# No Athena account, API key, or model download needed — see demo/README.md.
demo:
	uv run python scripts/run_demo.py

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

typecheck:
	uv run mypy src

test:
	uv run pytest

check: lint typecheck test

# Requires an unzipped Athena download in data/athena/ — see README Setup.
build-vocab:
	uv run python scripts/load_vocab.py data/athena --output data/vocab.duckdb

fetch-library:
	uv run python scripts/fetch_phenotype_library.py

build-index:
	uv run python scripts/build_index.py

# Chains every optional data-build step after install. Run once per machine,
# after placing an unzipped Athena download in data/athena/ (see README).
setup: install build-vocab fetch-library build-index

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
