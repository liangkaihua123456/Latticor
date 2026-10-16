#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export UCX_DIR="$PWD/ucx-1.15.0/install"
export PATH=$UCX_DIR/bin:$PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH

start_time=$(date +%s)

python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only JackFram/llama-68m
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only huggyllama/llama-7b 
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only huggyllama/llama-65b
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only facebook/opt-125m
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only facebook/opt-13b
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only facebook/opt-30b

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total download time: $execution_time seconds"
