.PHONY: install test lint format run-api docker-up docker-down

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check src/ tests/

format:
	poetry run ruff format src/ tests/

run-api:
	poetry run uvicorn satira.api:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up -d

docker-down:
	docker compose down
