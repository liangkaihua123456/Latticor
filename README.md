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

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 用本机 `flexflow_cache` 运行 SpecInfer

这份文档只覆盖两个目录：

- 程序：`/root/autodl-tmp/IEEE_Conference_smart/specinfer-ae-main`
- 模型缓存：`/root/autodl-tmp/flexflow_cache`

它不使用 `flexflow-train-master`，也不使用 `flexflow-work`。下面的检查命令已在本机执行过。2026-10-15 的结论是：缓存是齐的，但现成的 `FlexFlow/build` 二进制不能在当前 GPU 上运行。按本文第 4 节重编译之后，再用第 6 节的命令，才是这条机器上能跑通的路径。

英文版：`RUN_WITH_FLEXFLOW_CACHE.en.md`

## 1. 本机已经核对过的事实

| 项目               | 实测结果                                                     |
| ------------------ | ------------------------------------------------------------ |
| GPU                | NVIDIA GeForce RTX 4080 SUPER，32760 MiB，compute capability **8.9** |
| 驱动 / CUDA        | Driver 595.71.05，`/usr/local/cuda/bin/nvcc` 为 CUDA **12.8**（V12.8.93） |
| 现成二进制         | `FlexFlow/build/libflexflow.so.1` 里只有 **sm_120** cubin    |
| 直接运行现成二进制 | 进程 abort，日志为 `CUDA_ERROR_NO_BINARY_FOR_GPU (209)`：`no kernel image is available for execution on the device` |
| 内存 / 磁盘        | 内存约 503 GB；`/root/autodl-tmp` 剩余约 36 GB（95% 已用）。重编译前先看磁盘 |
| 编译器             | `gcc-10` / `g++-10`，CMake 3.22.1，12 核。`cmake --version` 可能先打印一行 `libcurl.so.4: no version information`，只要后面打出版本号就可以忽略 |
| 上次跑通           | 2026-08-11，当时 GPU 是 RTX 6000D（约 85 GB，sm_120）。增量解码、序列投机、树形投机都打印了 `OK`。那次的 `basic_test_single_gpu.sh` 把 `-ll:fsize` 设成 **70000**，只适用于那张大卡 |

当前这张 32 GB 的卡有两件事和旧脚本冲突，缺一不可：

1. 必须用 `FF_CUDA_ARCH=89` 重新编译。继续跑 `FlexFlow/build` 一定会在加载 kernel 时 abort。
2. `-ll:fsize` 不能再用 70000。这是 Legion 向 GPU 申请的显存池，单位是 MB。32760 MiB 的卡上应使用 **28000**。

不要执行仓库自带的 `./basic_test.sh`。它会重新下载模型、走多机 UCX，而且单机命令没有显式传入 `-cache-folder`。也不要直接执行 `./basic_test_single_gpu.sh`：它指向 sm_120 的 `FlexFlow/build`，并且 `fsize=70000`。

## 2. 两个目录怎么配合

`specinfer-ae-main/FlexFlow` 是 SpecInfer 论文工件里钉住的 FlexFlow（提交 `8d82c91a8`，推理分支）。真正被调用的程序是：

- `inference/incr_decoding/incr_decoding`：增量解码，只有目标模型
- `inference/spec_infer/spec_infer`：投机解码。`--expansion-degree -1` 是序列投机，不传该参数是树形投机

`flexflow_cache` 是这两个程序的 `-cache-folder`。程序按模型名的小写形式找文件，例如参数 `JackFram/llama-68m` 对应目录 `jackfram/llama-68m`：

```text
flexflow_cache/
  configs/<org>/<name>/config.json
  tokenizers/<org>/<name>/
  weights/<org>/<name>/half-precision/    # 原始 float16，不是 safetensors
```

本机缓存已经齐，运行时不需要联网，也不要再跑 `download_models.sh`（那个脚本会往默认缓存里重新下载）。四个模型：

| Hugging Face 名       | 缓存目录              | 角色             | half-precision 文件数 | tokenizer 文件数 |
| --------------------- | --------------------- | ---------------- | --------------------- | ---------------- |
| `JackFram/llama-68m`  | `jackfram/llama-68m`  | 小模型 / 候选    | 22                    | 4                |
| `huggyllama/llama-7b` | `huggyllama/llama-7b` | 目标             | 292                   | 4                |
| `facebook/opt-125m`   | `facebook/opt-125m`   | 候选             | 198                   | 6                |
| `facebook/opt-13b`    | `facebook/opt-13b`    | 目标（卸载测试） | 646                   | 6                |

`configs/` 里每份都有 `config.json`。权重目录里还有 `rev_sha.txt`。缺权重时程序不会自己补全。

## 3. 运行前检查

下面每一条都要成功。任何一条失败就先停，不要进入编译或推理。

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

期望：

- 四个权重目录的文件数分别是 **22、292、198、646**。少一个文件就不要跑对应模型。
- `compute_cap` 为 `8.9`。若以后换回 RTX 6000D（12.0），应改用原来的 `FlexFlow/build`，不要用本节编译出的 `build-sm89`。
- 磁盘至少再留出约 **15 GB** 给 `build-sm89`。不够就先清理，不要在 95% 的盘上硬编。

## 4. 为当前 GPU 重编译

保留原来的 `FlexFlow/build`（sm_120）。新二进制放到 `FlexFlow/build-sm89`，避免盖掉 8 月那次能在 6000D 上跑通的构建。

这一段就是 2026-08-11 已经成功的 `/root/autodl-tmp/specinfer_do_build.sh`，只改了三处：架构 `120` 改成 `89`，构建目录改成 `build-sm89`，并行度改成机器核数。单机测试不需要 UCX。不要 `source /etc/network_turbo`，缓存已经在本地。

预计 30 到 60 分钟。

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

最后一行必须出现 `sm_89`。如果仍是 `sm_120`，说明编到了旧目录，停下来检查 `pwd`。

`../config/config.linux` 若因 Python 路径失败，而目录里没有 `Makefile`，用下面这条补一次配置后再 `make`。这是当天构建脚本里的备用命令，架构已改为 89：

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

## 5. 每次运行前的环境

新开一个 shell 再执行。`LD_PRELOAD` 必须指向 conda 环境里的 `libittnotify.so`，否则 FlexFlow 可能在启动时找不到 Intel ITT 符号。`LD_LIBRARY_PATH` 必须包含 **build-sm89**，不能只包含旧的 `build`。

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

# 显存池：总显存减去 4 GB，再向下取整到 1000 MB。本机 32760 → 28000。
MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
FSIZE="$(( (MEM_MIB - 4096) / 1000 * 1000 ))"
echo "Using -ll:fsize $FSIZE on ${MEM_MIB} MiB"

PROMPT="$SPEC/FlexFlow/inference/prompt/test.json"
printf '%s\n' '["Three tips for staying healthy are: "]' > "$PROMPT"

INCR="$BUILD/inference/incr_decoding/incr_decoding"
SPECINF="$BUILD/inference/spec_infer/spec_infer"
OUT="$SPEC/FlexFlow/inference/output/basic_test"
```

`source activate_specinfer.sh` 会把 `FF_CACHE_PATH` 设对，但它的别名 `si-incr` / `si-spec` 仍指向旧的 `FlexFlow/build`。在 sm_89 的机器上不要用那两个别名。

## 6. 三步运行

三步都在第 5 节的同一个 shell 里执行。每一步结束都要看到对应的 `OK`，并且输出文件里有 `end-to-end latency`。不要把三步用 `&` 并行，它们会抢同一块 GPU。

### 6.1 先证明二进制、缓存和 GPU 接上了

只用 68M，序列长度 32，显存池 8000 MB。这一步几十秒内应结束。它不代替 7B 测试。

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

若这里再次出现 `CUDA_ERROR_NO_BINARY_FOR_GPU`，说明 `INCR` 仍指向 `FlexFlow/build`。执行 `ls -l "$INCR"` 确认路径含 `build-sm89`。

### 6.2 增量解码（LLaMA-7B）

这是 8 月 11 日已经跑通的同一组模型与同一句 prompt。显存池改为本节算出的 `FSIZE`（本机 28000）。7B 半精度权重大约 13 GB，28000 MB 的池子装得下权重和这一句 prompt 的 KV。

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

8 月 11 日的成功样本解码成：`Three tips for staying healthy are: 1) Eat a healthy diet, 2) Exercise regularly, and 3) Get enough sleep.` 文本不必逐字相同，但应是可读英文，并且从这句 prompt 接下去。

### 6.3 序列投机，然后树形投机

目标 `huggyllama/llama-7b`，候选 `JackFram/llama-68m`。两者都在 `flexflow_cache` 里。序列模式必须带 `--expansion-degree -1`。树形模式不要加这个参数。

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

日志里出现 `Applying fusion optimizations` 以及 `operators before fusion` / `operators after fusion` 是正常的。8 月 11 日 7B 增量解码是 424 个算子融合到 198 个；加上 68M 候选时还会再打印一组较小的数字（35 融合到 18）。

### 6.4 可选：OPT-13B 卸载

只在前三步都成功后再做。13B 半精度大约 25 GB，不能和 7B 一样整模放进 32 GB 卡，所以必须加 `-offload`。`-ll:fsize 21000` 是工件脚本里的卸载显存池，小于本机 32760 MiB。

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

