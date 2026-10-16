#!/usr/bin/env bash
# Usage: source activate_specinfer.sh
SPEC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate flexflow

export SPEC_ROOT
export FF_HOME="$SPEC_ROOT/FlexFlow"
export FF_CACHE_PATH=/root/autodl-tmp/flexflow_cache
export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${FF_HOME}/build:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${CONDA_PREFIX}/bin:${PATH}"
export TMPDIR=/root/autodl-tmp/tmp

mkdir -p /root/autodl-tmp/home_cache "$TMPDIR"
[[ -L /root/.cache ]] || { rm -rf /root/.cache 2>/dev/null; ln -sfn /root/autodl-tmp/home_cache /root/.cache; }
ln -sfn /root/autodl-tmp/flexflow_cache /root/autodl-tmp/home_cache/flexflow 2>/dev/null || true

alias si-incr="$FF_HOME/build/inference/incr_decoding/incr_decoding"
alias si-spec="$FF_HOME/build/inference/spec_infer/spec_infer"
alias si-basic="$SPEC_ROOT/basic_test_single_gpu.sh"

echo "SpecInfer ready (conda: flexflow)"
echo "  SPEC_ROOT=$SPEC_ROOT"
echo "  Cache: $FF_CACHE_PATH"
echo "  Run: bash $SPEC_ROOT/basic_test_single_gpu.sh"
echo "  A+B (Diffusion Lattice→Tree): bash $SPEC_ROOT/run_diffusion_tree_experiments.sh quick"
echo "  Note: A+B needs CUDA PyTorch; if flexflow torch is CPU-only, the launcher falls back to spork."
