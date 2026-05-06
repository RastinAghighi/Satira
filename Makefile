.PHONY: install test test-unit test-integration test-coverage lint format typecheck run-api docker-up docker-down

install:
	poetry install

test:
	bash scripts/run_tests.sh

test-unit:
	poetry run pytest -v tests/unit

test-integration:
	poetry run pytest -v --timeout=60 tests/integration

test-coverage:
	poetry run pytest --cov=satira --cov-report=term-missing --cov-report=html

lint:
	poetry run ruff check src/ tests/

format:
	poetry run black src/ tests/

typecheck:
	poetry run mypy src/satira

run-api:
	poetry run uvicorn satira.api:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up -d

docker-down:
	docker compose down
