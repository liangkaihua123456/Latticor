#!/usr/bin/env bash
# Single-GPU SpecInfer smoke test (skips multi-node / 4-GPU sections).
set -euo pipefail

cd "$(dirname "$0")"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flexflow

export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export FF_CACHE_PATH=/root/autodl-tmp/flexflow_cache
export PATH="/usr/local/cuda/bin:${PATH}"

# Prefer autodl-tmp for any incidental caches (root FS is small)
mkdir -p /root/autodl-tmp/home_cache /root/autodl-tmp/flexflow_cache
if [[ ! -L /root/.cache ]]; then
  rm -rf /root/.cache 2>/dev/null || true
  ln -sfn /root/autodl-tmp/home_cache /root/.cache || true
fi
ln -sfn /root/autodl-tmp/flexflow_cache /root/autodl-tmp/home_cache/flexflow || true

BUILD=./FlexFlow/build
INCR="$BUILD/inference/incr_decoding/incr_decoding"
SPEC="$BUILD/inference/spec_infer/spec_infer"

mkdir -p ./FlexFlow/inference/output/basic_test ./FlexFlow/inference/prompt
echo '["Three tips for staying healthy are: "]' > ./FlexFlow/inference/prompt/test.json

# RTX 6000D ~85GB; use large fsize/zsize
ncpus=8
ngpus=1
bs=1
# ~70GB device memory for Legion framebuffer
fsize=70000
zsize=40000

llm_model_name="huggyllama/llama-7b"
ssm_model_name="JackFram/llama-68m"

echo "Running single node tests on 1x GPU..."

CACHE=/root/autodl-tmp/flexflow_cache

echo ">>> Incremental decoding test..."
"$INCR" \
  -ll:cpu "$ncpus" -ll:util "$ncpus" -ll:gpu "$ngpus" \
  -ll:fsize "$fsize" -ll:zsize "$zsize" \
  -llm-model "$llm_model_name" \
  -cache-folder "$CACHE" \
  -prompt ./FlexFlow/inference/prompt/test.json \
  --max-requests-per-batch "$bs" \
  -tensor-parallelism-degree "$ngpus" \
  --fusion \
  -output-file ./FlexFlow/inference/output/basic_test/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.txt \
  > ./FlexFlow/inference/output/basic_test/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.out
echo ">>> Incremental decoding OK"

echo ">>> Sequence-based speculative decoding test..."
"$SPEC" \
  -ll:cpu "$ncpus" -ll:util "$ncpus" -ll:gpu "$ngpus" \
  -ll:fsize "$fsize" -ll:zsize "$zsize" \
  -llm-model "$llm_model_name" \
  -ssm-model "$ssm_model_name" \
  -cache-folder "$CACHE" \
  -prompt ./FlexFlow/inference/prompt/test.json \
  --max-requests-per-batch "$bs" \
  --expansion-degree -1 \
  -tensor-parallelism-degree "$ngpus" \
  --fusion \
  -output-file ./FlexFlow/inference/output/basic_test/1_machine-${ngpus}_gpu-${bs}_batchsize-sequence_specinfer.txt \
  > ./FlexFlow/inference/output/basic_test/1_machine-${ngpus}_gpu-${bs}_batchsize-sequence_specinfer.out
echo ">>> Sequence-based speculative decoding OK"

echo ">>> Tree-based speculative decoding test..."
"$SPEC" \
  -ll:cpu "$ncpus" -ll:util "$ncpus" -ll:gpu "$ngpus" \
  -ll:fsize "$fsize" -ll:zsize "$zsize" \
  -llm-model "$llm_model_name" \
  -ssm-model "$ssm_model_name" \
  -cache-folder "$CACHE" \
  -prompt ./FlexFlow/inference/prompt/test.json \
  --max-requests-per-batch "$bs" \
  -tensor-parallelism-degree "$ngpus" \
  --fusion \
  -output-file ./FlexFlow/inference/output/basic_test/1_machine-${ngpus}_gpu-${bs}_batchsize-tree_specinfer.txt \
  > ./FlexFlow/inference/output/basic_test/1_machine-${ngpus}_gpu-${bs}_batchsize-tree_specinfer.out
echo ">>> Tree-based speculative decoding OK"

# Offloading (needs opt-13b)
if [[ -d "$CACHE/weights/facebook/opt-13b/half-precision" ]] && \
   [[ -d "$CACHE/weights/facebook/opt-125m/half-precision" ]] && \
   [[ -f "$CACHE/weights/facebook/opt-13b/half-precision/.done" || $(ls "$CACHE/weights/facebook/opt-13b/half-precision" | wc -l) -gt 10 ]]; then
  echo ">>> Offloading test..."
  "$SPEC" \
    -ll:cpu "$ncpus" -ll:util "$ncpus" -ll:gpu "$ngpus" \
    -ll:fsize 21000 -ll:zsize "$zsize" \
    -llm-model "facebook/opt-13b" \
    -ssm-model "facebook/opt-125m" \
    -cache-folder "$CACHE" \
    -prompt ./FlexFlow/inference/prompt/test.json \
    -offload -offload-reserve-space-size 500 \
    --max-sequence-length 128 \
    --max-requests-per-batch "$bs" \
    -output-file ./FlexFlow/inference/output/basic_test/offloading.txt \
    > ./FlexFlow/inference/output/basic_test/offloading.out
  echo ">>> Offloading OK"
else
  echo ">>> Skipping offloading (opt-13b not ready yet)"
fi

echo ""
echo "Single-node Test passed!"
echo "Outputs:"
ls -la ./FlexFlow/inference/output/basic_test/
echo "--- sample incr_dec output ---"
head -c 800 ./FlexFlow/inference/output/basic_test/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.txt || true
echo
