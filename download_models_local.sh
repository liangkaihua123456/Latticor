#!/usr/bin/env bash
set -euo pipefail
source /etc/network_turbo || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate flexflow

export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME=/root/autodl-tmp/hf_home
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_home
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$HF_HOME" "$TMPDIR" /root/autodl-tmp/flexflow_cache /root/.cache
ln -sfn /root/autodl-tmp/flexflow_cache /root/.cache/flexflow

CACHE=/root/autodl-tmp/flexflow_cache
cd /root/autodl-tmp/IEEE_Conference_smart/specinfer-ae-main

download_one() {
  local name="$1"
  echo "===== Downloading $name ====="
  python ./FlexFlow/inference/utils/download_hf_model.py \
    --half-precision-only \
    --cache-folder "$CACHE" \
    "$name"
  echo "===== Done $name ====="
}

# Models for single-node basic_test (and offloading)
download_one "JackFram/llama-68m"
download_one "huggyllama/llama-7b"
download_one "facebook/opt-125m"
download_one "facebook/opt-13b"

echo "=== ALL MODELS DONE ==="
du -sh "$CACHE"/*/* 2>/dev/null | head -40
