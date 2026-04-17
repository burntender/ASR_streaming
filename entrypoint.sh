#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "serve" ]]; then
  shift
  exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8080}" "$@"
fi

exec python /app/transcribe.py "$@"
