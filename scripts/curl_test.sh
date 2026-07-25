#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:-sample_images/test_note.jpg}"
API_URL="${API_URL:-http://127.0.0.1:8000/predict}"

curl --fail-with-body --silent --show-error \
  -X POST "$API_URL" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@${IMAGE_PATH}" | python -m json.tool
