#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export UCX_DIR="$PWD/ucx-1.15.0/install"
export PATH=$UCX_DIR/bin:$PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH

./download_dataset.sh
./download_models.sh

batch_sizes=( 1 2 4 8 16 )

# rm -rf ./FlexFlow/inference/output || true
mkdir -p ./FlexFlow/inference/output

start_time=$(date +%s)

# single node, single GPU
ncpus=8
ngpus=1
fsize=21890
zsize=80000
max_sequence_length=128
llm_model_name="huggyllama/llama-7b"
ssm_model_name="JackFram/llama-68m"
for bs in "${batch_sizes[@]}"
do
    # Incremental decoding
    ./FlexFlow/build/inference/incr_decoding/incr_decoding -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length  -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/server_small-${bs}_batchsize-incr_dec.txt > ./FlexFlow/inference/output/server_small-${bs}_batchsize-incr_dec.out
    # Sequence-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length  --expansion-degree -1 -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/server_small-${bs}_batchsize-sequence_specinfer.txt > ./FlexFlow/inference/output/server_small-${bs}_batchsize-sequence_specinfer.out
    # Tree-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length  -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/server_small-${bs}_batchsize-tree_specinfer.txt > ./FlexFlow/inference/output/server_small-${bs}_batchsize-tree_specinfer.out
done

# single node, multiple GPU
ncpus=8
ngpus=4
fsize=21800
zsize=80000
max_sequence_length=128
llm_model_name="facebook/opt-30b"
ssm_model_name="facebook/opt-125m"
for bs in "${batch_sizes[@]}"
do
    # Incremental decoding
    ./FlexFlow/build/inference/incr_decoding/incr_decoding -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length  -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/server_medium-${bs}_batchsize-incr_dec.txt > ./FlexFlow/inference/output/server_medium-${bs}_batchsize-incr_dec.out
    # Sequence-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length  --expansion-degree -1 -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/server_medium-${bs}_batchsize-sequence_specinfer.txt > ./FlexFlow/inference/output/server_medium-${bs}_batchsize-sequence_specinfer.out
    # Tree-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length  -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/server_medium-${bs}_batchsize-tree_specinfer.txt > ./FlexFlow/inference/output/server_medium-${bs}_batchsize-tree_specinfer.out
done

exit 0

# tensor + pipeline parallelism
ncpus=8
ngpus=8
fsize=21800
zsize=80000
max_sequence_length=128
tp_degree=4
pp_degree=2
llm_model_name="huggyllama/llama-65b"
ssm_model_name="JackFram/llama-68m"
for bs in "${batch_sizes[@]}"
do
    # Incremental decoding
    ./FlexFlow/build/inference/incr_decoding/incr_decoding -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length -tensor-parallelism-degree $tp_degree -pipeline-parallelism-degree $pp_degree --fusion -output-file ./FlexFlow/inference/output/server_large-${bs}_batchsize-incr_dec.txt > ./FlexFlow/inference/output/server_large-${bs}_batchsize-incr_dec.out
    # Sequence-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length  --expansion-degree -1 -tensor-parallelism-degree $tp_degree -pipeline-parallelism-degree $pp_degree --fusion -output-file ./FlexFlow/inference/output/server_large-${bs}_batchsize-sequence_specinfer.txt > server_large-${bs}_batchsize-sequence_specinfer.out
    # Tree-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize $fsize -ll:zsize $zsize -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --max-sequence-length $max_sequence_length -tensor-parallelism-degree $tp_degree -pipeline-parallelism-degree $pp_degree --fusion -output-file ./FlexFlow/inference/output/server_large-${bs}_batchsize-tree_specinfer.txt > ./FlexFlow/inference/output/server_large-${bs}_batchsize-tree_specinfer.out
done

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total server gpu test time: $execution_time seconds"