#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTEST="poetry run pytest"

unit_status=0
integration_status=0
coverage_status=0

echo "=========================================="
echo "[1/3] Unit tests"
echo "=========================================="
$PYTEST -v tests/unit || unit_status=$?

echo
echo "=========================================="
echo "[2/3] Integration tests"
echo "=========================================="
$PYTEST -v --timeout=60 tests/integration || integration_status=$?

echo
echo "=========================================="
echo "[3/3] Coverage report"
echo "=========================================="
$PYTEST --cov=satira --cov-report=term-missing --cov-report=html || coverage_status=$?

echo
echo "=========================================="
echo "Summary"
echo "=========================================="
printf "  unit tests:        %s\n"        "$([ $unit_status -eq 0 ] && echo PASS || echo "FAIL ($unit_status)")"
printf "  integration tests: %s\n" "$([ $integration_status -eq 0 ] && echo PASS || echo "FAIL ($integration_status)")"
printf "  coverage run:      %s\n"      "$([ $coverage_status -eq 0 ] && echo PASS || echo "FAIL ($coverage_status)")"

if [ $unit_status -ne 0 ] || [ $integration_status -ne 0 ] || [ $coverage_status -ne 0 ]; then
  exit 1
fi
echo "All checks passed."
