#!/usr/bin/env bash
# Download the four paper models into a FlexFlow half-precision cache.
# Weights are written under ./downloaded (or $FF_CACHE_PATH), not committed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! python3 -c 'import torch, transformers, numpy, accelerate' >/dev/null 2>&1; then
  echo "Installing Python dependencies from requirements.txt"
  python3 -m pip install -r "$ROOT/requirements.txt"
fi

exec python3 "$ROOT/download_models.py" "$@"
