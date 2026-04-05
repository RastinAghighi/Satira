# Satira — Multimodal Satire & Misinformation Detection Engine

A multimodal detection engine that combines text, image, and graph-based analysis to identify satire and misinformation across media sources.

## Setup

```bash
# Install dependencies
make install

# Copy environment variables
cp .env.example .env

# Run the API server
make run-api

# Run tests
make test
```

## Development

```bash
# Lint and format
make lint
make format

# Docker services
make docker-up
make docker-down
```
