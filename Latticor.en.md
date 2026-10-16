# Run SpecInfer with this machine's `flexflow_cache`

This document covers only:

- Code: `/root/autodl-tmp/IEEE_Conference_smart/specinfer-ae-main`
- Model cache: `/root/autodl-tmp/flexflow_cache`

It does not use `flexflow-train-master` or `flexflow-work`. The checks below were run on this machine. As of 2026-10-15 the cache is complete, but the prebuilt `FlexFlow/build` binaries cannot run on the current GPU. Rebuild as in section 4, then use the commands in section 6.

Chinese version: `RUN_WITH_FLEXFLOW_CACHE.zh.md`

## 1. What was checked on this machine

| Item | Measured result |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32760 MiB, compute capability **8.9** |
| Driver / CUDA | Driver 595.71.05, `/usr/local/cuda/bin/nvcc` is CUDA **12.8** (V12.8.93) |
| Prebuilt binaries | `FlexFlow/build/libflexflow.so.1` contains **sm_120** cubins only |
| Running those binaries | Process aborts with `CUDA_ERROR_NO_BINARY_FOR_GPU (209)`: `no kernel image is available for execution on the device` |
| RAM / disk | About 503 GB RAM; about 36 GB free on `/root/autodl-tmp` (95% used). Check disk before compiling |
| Toolchain | `gcc-10` / `g++-10`, CMake 3.22.1, 12 cores. `cmake --version` may first print `libcurl.so.4: no version information`; ignore that line if a version follows |
| Last successful run | 2026-08-11 on an RTX 6000D (about 85 GB, sm_120). Incremental, sequence-speculative, and tree-speculative decoding each printed `OK`. That `basic_test_single_gpu.sh` sets `-ll:fsize` to **70000**, which fits only that larger GPU |

On this 32 GB card both of the following are required:

1. Rebuild with `FF_CUDA_ARCH=89`. Launching `FlexFlow/build` will abort while loading the kernel image.
2. Do not pass `-ll:fsize 70000`. That flag is the Legion GPU memory pool in MB. On a 32760 MiB card use **28000**.

Do not run `./basic_test.sh`. It downloads models again, expects multi-node UCX, and its single-node commands do not pass `-cache-folder`. Do not run `./basic_test_single_gpu.sh` as-is either: it points at the sm_120 tree and uses `fsize=70000`.

## 2. How the two directories fit together

`specinfer-ae-main/FlexFlow` is the FlexFlow snapshot vendored by the SpecInfer artifact (commit `8d82c91a8`, inference line). The programs that actually run are:

- `inference/incr_decoding/incr_decoding`: incremental decoding, target model only
- `inference/spec_infer/spec_infer`: speculative decoding. `--expansion-degree -1` is sequence speculation; omitting it is tree speculation

`flexflow_cache` is the `-cache-folder` for both. Lookups use the lowercased model id, so `JackFram/llama-68m` maps to `jackfram/llama-68m`:

```text
flexflow_cache/
  configs/<org>/<name>/config.json
  tokenizers/<org>/<name>/
  weights/<org>/<name>/half-precision/    # raw float16, not safetensors
```

The cache on this machine is already complete. Inference does not need the network. Do not run `download_models.sh`; that script downloads into the default cache again. The four models are:

| Hugging Face id | Cache directory | Role | half-precision files | tokenizer files |
|---|---|---|---|---|
| `JackFram/llama-68m` | `jackfram/llama-68m` | small model / draft | 22 | 4 |
| `huggyllama/llama-7b` | `huggyllama/llama-7b` | target | 292 | 4 |
| `facebook/opt-125m` | `facebook/opt-125m` | draft | 198 | 6 |
| `facebook/opt-13b` | `facebook/opt-13b` | target (offload test) | 646 | 6 |

Each config directory contains `config.json`. Each weight directory contains `rev_sha.txt`. The binaries do not fill in missing weights.

## 3. Preflight

Every command below must succeed. Stop if any check fails.

```bash
set -euo pipefail
SPEC=/root/autodl-tmp/IEEE_Conference_smart/specinfer-ae-main
CACHE=/root/autodl-tmp/flexflow_cache

test -x "$SPEC/FlexFlow/build/inference/incr_decoding/incr_decoding"
test -d "$CACHE/weights/huggyllama/llama-7b/half-precision"
test -f "$CACHE/configs/huggyllama/llama-7b/config.json"
test -f "$CACHE/tokenizers/huggyllama/llama-7b/tokenizer.model"
test -f "$CACHE/tokenizers/jackfram/llama-68m/tokenizer.model"
test -f "$CACHE/configs/facebook/opt-13b/config.json"
test -x /usr/local/cuda/bin/nvcc
test -x /usr/bin/gcc-10
test -f /root/miniconda3/envs/flexflow/lib/libittnotify.so

echo "weight files:"
for m in jackfram/llama-68m huggyllama/llama-7b facebook/opt-125m facebook/opt-13b; do
  printf '  %-28s %s\n' "$m" "$(find "$CACHE/weights/$m/half-precision" -type f | wc -l)"
done
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
df -h /root/autodl-tmp | awk 'NR==2{print "disk avail:", $4}'
```

Expect:

- Weight-file counts **22, 292, 198, 646**. Do not run a model whose directory is short.
- `compute_cap` is `8.9`. If the machine is later switched back to an RTX 6000D (12.0), use the original `FlexFlow/build`, not `build-sm89`.
- At least about **15 GB** free for `build-sm89`. Free space first if the disk is already at 95%.

## 4. Rebuild for this GPU

Keep the existing `FlexFlow/build` (sm_120). Put the new binaries in `FlexFlow/build-sm89` so the August build that ran on the 6000D is not overwritten.

The block below is the 2026-08-11 script `/root/autodl-tmp/specinfer_do_build.sh` with three changes: architecture `120` becomes `89`, the build directory becomes `build-sm89`, and the job count follows `nproc`. Single-GPU tests do not need UCX. Do not `source /etc/network_turbo`; the weights are already local.

Expect 30 to 60 minutes.

```bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate flexflow

export PATH="/usr/bin:/usr/local/cuda/bin:${CONDA_PREFIX}/bin:/bin"
export CC=gcc-10
export CXX=g++-10
export CUDA_DIR=/usr/local/cuda
export CUDNN_DIR=/usr
export FF_CUDA_ARCH=89
export FF_LEGION_NETWORKS=
unset UCX_DIR || true
export BUILD_TYPE=Release
export TMPDIR=/root/autodl-tmp/tmp
export FF_USE_PYTHON=ON
mkdir -p "$TMPDIR"

SPEC=/root/autodl-tmp/IEEE_Conference_smart/specinfer-ae-main
cd "$SPEC/FlexFlow"
rm -rf build-sm89
mkdir build-sm89
cd build-sm89

../config/config.linux
make -j"$(nproc)" install

test -x inference/incr_decoding/incr_decoding
test -x inference/spec_infer/spec_infer
echo "==== cubin arch (must include sm_89) ===="
/usr/local/cuda/bin/cuobjdump -lelf libflexflow.so.1 | awk '/sm_/ {print; exit}'
```

The last line must show `sm_89`. If it still shows `sm_120`, the build landed in the old directory. Check `pwd` and stop.

If `../config/config.linux` fails on the Python path and there is no `Makefile`, configure once with the fallback below (the August fallback, with the architecture set to 89), then `make` again:

```bash
PY="${CONDA_PREFIX}/bin/python"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-10 \
  -DCMAKE_CXX_COMPILER=g++-10 \
  -DFF_CUDA_ARCH=89 \
  -DCUDA_PATH=/usr/local/cuda \
  -DCUDNN_PATH=/usr \
  -DFF_USE_PYTHON=ON \
  -DFF_BUILD_ALL_INFERENCE_EXAMPLES=ON \
  -DFF_USE_NCCL=ON \
  -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
  -DLegion_HIJACK_CUDART=OFF \
  -DPython3_EXECUTABLE="$PY" \
  -DPython_EXECUTABLE="$PY" \
  -DPython3_ROOT_DIR="${CONDA_PREFIX}" \
  -DPython_ROOT_DIR="${CONDA_PREFIX}" \
  -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}"
make -j"$(nproc)" install
```

## 5. Environment for every run

Open a new shell. `LD_PRELOAD` must point at `libittnotify.so` inside the conda env, or FlexFlow can fail at startup looking for Intel ITT symbols. `LD_LIBRARY_PATH` must include **build-sm89**, not only the old `build`.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate flexflow

SPEC=/root/autodl-tmp/IEEE_Conference_smart/specinfer-ae-main
CACHE=/root/autodl-tmp/flexflow_cache
BUILD="$SPEC/FlexFlow/build-sm89"

export PATH="/usr/local/cuda/bin:${CONDA_PREFIX}/bin:${PATH}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${BUILD}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR" "$SPEC/FlexFlow/inference/output/basic_test" "$SPEC/FlexFlow/inference/prompt"

# GPU pool: total MiB minus 4 GiB, rounded down to 1000 MB. This machine: 32760 → 28000.
MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
FSIZE="$(( (MEM_MIB - 4096) / 1000 * 1000 ))"
echo "Using -ll:fsize $FSIZE on ${MEM_MIB} MiB"

PROMPT="$SPEC/FlexFlow/inference/prompt/test.json"
printf '%s\n' '["Three tips for staying healthy are: "]' > "$PROMPT"

INCR="$BUILD/inference/incr_decoding/incr_decoding"
SPECINF="$BUILD/inference/spec_infer/spec_infer"
OUT="$SPEC/FlexFlow/inference/output/basic_test"
```

`source activate_specinfer.sh` does set `FF_CACHE_PATH` correctly, but its `si-incr` and `si-spec` aliases still point at `FlexFlow/build`. Do not use those aliases on this sm_89 GPU.

## 6. Three runs

Run all three in the same shell as section 5. Each step must print its `OK` line, and the output file must contain `end-to-end latency`. Do not background them; they share one GPU.

### 6.1 Prove the binary, the cache, and the GPU

68M only, sequence length 32, 8000 MB pool. This should finish within about a minute. It does not replace the 7B run.

```bash
"$INCR" \
  -ll:cpu 4 -ll:util 4 -ll:gpu 1 \
  -ll:fsize 8000 -ll:zsize 8000 \
  -llm-model JackFram/llama-68m \
  -cache-folder "$CACHE" \
  -prompt "$PROMPT" \
  --max-requests-per-batch 1 \
  --max-sequence-length 32 \
  -tensor-parallelism-degree 1 \
  --fusion \
  -output-file "$OUT/smoke_llama68m.txt" \
  | tee "$OUT/smoke_llama68m.out"

grep -n "end-to-end latency" "$OUT/smoke_llama68m.txt"
echo "SMOKE OK"
```

If `CUDA_ERROR_NO_BINARY_FOR_GPU` appears again, `INCR` still points at `FlexFlow/build`. Run `ls -l "$INCR"` and confirm the path contains `build-sm89`.

### 6.2 Incremental decoding (LLaMA-7B)

Same models and prompt as the successful 11 August run. The memory pool is the `FSIZE` from section 5 (28000 on this machine). 7B half-precision weights are about 13 GB, so a 28000 MB pool holds the weights and the KV cache for this prompt.

```bash
"$INCR" \
  -ll:cpu 8 -ll:util 8 -ll:gpu 1 \
  -ll:fsize "$FSIZE" -ll:zsize 40000 \
  -llm-model huggyllama/llama-7b \
  -cache-folder "$CACHE" \
  -prompt "$PROMPT" \
  --max-requests-per-batch 1 \
  -tensor-parallelism-degree 1 \
  --fusion \
  -output-file "$OUT/llama7b_incr.txt" \
  | tee "$OUT/llama7b_incr.out"

grep -n "end-to-end latency" "$OUT/llama7b_incr.txt"
echo "INCR OK"
```

The 11 August sample decoded to: `Three tips for staying healthy are: 1) Eat a healthy diet, 2) Exercise regularly, and 3) Get enough sleep.` The text does not have to match word for word. It should be readable English that continues the prompt.

### 6.3 Sequence speculation, then tree speculation

Target `huggyllama/llama-7b`, draft `JackFram/llama-68m`. Both are already in `flexflow_cache`. Sequence mode must pass `--expansion-degree -1`. Tree mode must not.

```bash
"$SPECINF" \
  -ll:cpu 8 -ll:util 8 -ll:gpu 1 \
  -ll:fsize "$FSIZE" -ll:zsize 40000 \
  -llm-model huggyllama/llama-7b \
  -ssm-model JackFram/llama-68m \
  -cache-folder "$CACHE" \
  -prompt "$PROMPT" \
  --max-requests-per-batch 1 \
  --expansion-degree -1 \
  -tensor-parallelism-degree 1 \
  --fusion \
  -output-file "$OUT/llama7b_sequence.txt" \
  | tee "$OUT/llama7b_sequence.out"
grep -n "end-to-end latency" "$OUT/llama7b_sequence.txt"
echo "SEQUENCE OK"

"$SPECINF" \
  -ll:cpu 8 -ll:util 8 -ll:gpu 1 \
  -ll:fsize "$FSIZE" -ll:zsize 40000 \
  -llm-model huggyllama/llama-7b \
  -ssm-model JackFram/llama-68m \
  -cache-folder "$CACHE" \
  -prompt "$PROMPT" \
  --max-requests-per-batch 1 \
  -tensor-parallelism-degree 1 \
  --fusion \
  -output-file "$OUT/llama7b_tree.txt" \
  | tee "$OUT/llama7b_tree.out"
grep -n "end-to-end latency" "$OUT/llama7b_tree.txt"
echo "TREE OK"
```

`Applying fusion optimizations` and the before/after operator counts are expected. On 11 August, 7B incremental decoding fused 424 operators down to 198. With the 68M draft, a second smaller pair is printed (35 fused to 18).

### 6.4 Optional: OPT-13B offload

Run this only after the first three steps succeed. OPT-13B half precision is about 25 GB, so it does not fit in a 32 GB card the way 7B does. `-offload` is required. `-ll:fsize 21000` is the artifact's offload pool, and it is below this card's 32760 MiB.

```bash
"$SPECINF" \
  -ll:cpu 8 -ll:util 8 -ll:gpu 1 \
  -ll:fsize 21000 -ll:zsize 40000 \
  -llm-model facebook/opt-13b \
  -ssm-model facebook/opt-125m \
  -cache-folder "$CACHE" \
  -prompt "$PROMPT" \
  -offload -offload-reserve-space-size 500 \
  --max-sequence-length 128 \
  --max-requests-per-batch 1 \
  -output-file "$OUT/opt13b_offload.txt" \
  | tee "$OUT/opt13b_offload.out"
grep -n "end-to-end latency" "$OUT/opt13b_offload.txt"
echo "OFFLOAD OK"
```

## 7. What counts as a successful run

All four of the following:

1. Section 6.1 prints `SMOKE OK` and does not print `CUDA_ERROR_NO_BINARY_FOR_GPU`.
2. Section 6.2 prints `INCR OK`. `llama7b_incr.txt` contains `end-to-end latency`, `num decoding steps`, `token IDs`, and English that starts with the prompt.
3. Section 6.3 prints `SEQUENCE OK` and then `TREE OK`. Both txt files contain `end-to-end latency`.
4. `echo $?` is 0 after each command. FlexFlow sometimes reports errors on stdout, so a txt file alone is not enough; it must contain the latency line.

At that point `specinfer-ae-main` is running against `flexflow_cache` on one GPU. Section 6.4 is an extra offload path, not a prerequisite.

## 8. If it fails

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA_ERROR_NO_BINARY_FOR_GPU` or `no kernel image is available` | The sm_120 `FlexFlow/build` binary was launched | Use the binaries under `build-sm89`, and put that directory on `LD_LIBRARY_PATH` |
| Framebuffer / `fsize` allocation failure, or the process dies while requesting 70000 MB | The pool is larger than the 32 GB card | Use `FSIZE` from section 5. Do not run `basic_test_single_gpu.sh` |
| Missing weights, tokenizer, or config | Wrong `-cache-folder`, or the model id does not match the directory | The folder must be `/root/autodl-tmp/flexflow_cache`. Pass ids such as `huggyllama/llama-7b` |
| `libittnotify.so` or an ITT symbol error | `LD_PRELOAD` is unset | Use the `LD_PRELOAD` line in section 5 |
| `libflexflow.so.1: cannot open shared object file` | The library path points at the old build, or not at the new one | Put `${BUILD}` at the front of `LD_LIBRARY_PATH` |
| `cmake` prints `libcurl.so.4: no version information` | The conda libcurl and the system cmake disagree about symbol versions | Continue if `cmake version` is printed next |
| Disk fills up during the build | About 36 GB was free on `/root/autodl-tmp` | Stop make, free space, and rebuild. Do not put the second build tree inside the 38 GB `flexflow_cache` |
