#!/usr/bin/env bash
# Run Diffusion Lattice → Tree (A+B) experiments.
# Prefers conda flexflow when it has CUDA PyTorch; otherwise uses spork.
set -euo pipefail
SPEC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset LD_PRELOAD || true
export PYTHONPATH="$SPEC_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_DIR="$SPEC_ROOT/FlexFlow/inference/output/diffusion_tree_spec"
mkdir -p "$OUT_DIR"

PY=""
if env -u LD_PRELOAD /root/miniconda3/envs/flexflow/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  PY=/root/miniconda3/envs/flexflow/bin/python
  echo "Using flexflow (CUDA torch OK)"
elif env -u LD_PRELOAD /root/autodl-tmp/envs/spork/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  PY=/root/autodl-tmp/envs/spork/bin/python
  echo "NOTE: flexflow has CPU-only torch; using spork CUDA python for A+B"
else
  echo "ERROR: no Python with CUDA torch found."
  echo "Install into flexflow: pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124"
  exit 1
fi

env -u LD_PRELOAD "$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0),
      "mem_gb", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
PY

MODE="${1:-quick}"
TARGET="${DTS_TARGET:-JackFram/llama-68m}"
DRAFT="${DTS_DRAFT:-JackFram/llama-68m}"
# Prefer converted FlexFlow LLaMA-7B, then HF/ModelScope caches.
if ls /root/autodl-tmp/hf_home/models/llama-7b-from-flexflow/*.safetensors >/dev/null 2>&1; then
  TARGET=/root/autodl-tmp/hf_home/models/llama-7b-from-flexflow
  echo "Detected FlexFlow-converted LLaMA-7B → $TARGET"
elif ls /root/autodl-tmp/hf_home/models/llama-7b/*.safetensors >/dev/null 2>&1; then
  TARGET=/root/autodl-tmp/hf_home/models/llama-7b
  echo "Detected local LLaMA-7B → $TARGET"
elif ls /root/autodl-tmp/hf_home/modelscope/Qwen/Qwen2.5-7B-Instruct/*.safetensors >/dev/null 2>&1 \
  || ls /root/autodl-tmp/hf_home/modelscope/models/Qwen/Qwen2.5-7B-Instruct/*.safetensors >/dev/null 2>&1; then
  if ls /root/autodl-tmp/hf_home/modelscope/Qwen/Qwen2.5-7B-Instruct/*.safetensors >/dev/null 2>&1; then
    TARGET=/root/autodl-tmp/hf_home/modelscope/Qwen/Qwen2.5-7B-Instruct
  else
    TARGET=/root/autodl-tmp/hf_home/modelscope/models/Qwen/Qwen2.5-7B-Instruct
  fi
  echo "Detected ModelScope Qwen2.5-7B → $TARGET"
elif ls /root/autodl-tmp/hf_home/modelscope/Qwen/Qwen2.5-3B-Instruct/*.safetensors >/dev/null 2>&1 \
  || ls /root/autodl-tmp/hf_home/modelscope/models/Qwen/Qwen2.5-3B-Instruct/*.safetensors >/dev/null 2>&1; then
  if ls /root/autodl-tmp/hf_home/modelscope/Qwen/Qwen2.5-3B-Instruct/*.safetensors >/dev/null 2>&1; then
    TARGET=/root/autodl-tmp/hf_home/modelscope/Qwen/Qwen2.5-3B-Instruct
  else
    TARGET=/root/autodl-tmp/hf_home/modelscope/models/Qwen/Qwen2.5-3B-Instruct
  fi
  echo "Detected ModelScope Qwen2.5-3B → $TARGET"
fi

if [[ "$MODE" == "smoke" ]]; then
  env -u LD_PRELOAD "$PY" "$SPEC_ROOT/diffusion_tree_spec/smoke_test.py" 2>&1 | tee "$OUT_DIR/smoke.log"
elif [[ "$MODE" == "full" ]]; then
  env -u LD_PRELOAD "$PY" -m diffusion_tree_spec.run_experiments --full --out-dir "$OUT_DIR" \
    --target "$TARGET" --draft "$DRAFT" 2>&1 | tee "$OUT_DIR/run_full.log"
else
  env -u LD_PRELOAD "$PY" -m diffusion_tree_spec.run_experiments --quick --out-dir "$OUT_DIR" \
    --target "$TARGET" --draft "$DRAFT" 2>&1 | tee "$OUT_DIR/run_quick.log"
fi

echo "Done. Results: $OUT_DIR"
