sources := "src tests"

default:
    @just --list

format:
    uv run ruff format {{sources}}

fix: format
    uv run ruff check --fix {{sources}}

lint:
    uv run ruff format --check {{sources}}
    uv run ruff check {{sources}}

typecheck:
    uv run ty check --python-version 3.10 {{sources}}

test:
    uv run python -m unittest discover -s tests -v
    bun test tests/omp_sdk.test.ts

# Provision the pinned optional OMP SDK; not required by `test`.
omp-runtime:
    uv run agent-execution-worker install-omp

check: lint typecheck test
