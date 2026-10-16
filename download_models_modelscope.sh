#!/usr/bin/env bash
# Download experiment models from ModelScope (魔搭社区).
# Prefer this when HuggingFace downloads stall.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY LD_PRELOAD || true

OUT=/root/autodl-tmp/hf_home/modelscope
mkdir -p "$OUT"
PY="${PYTHON:-/root/autodl-tmp/envs/spork/bin/python}"
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"

echo "ModelScope download: $MODEL → $OUT"
"$PY" -u -c "
from modelscope import snapshot_download
p = snapshot_download('$MODEL', cache_dir='$OUT')
print('DONE', p)
"
