#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export UCX_DIR="$PWD/ucx-1.15.0/install"
export PATH=$UCX_DIR/bin:$PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH

FLEXFLOW_DIR=/home/ubuntu/specinfer-ae/FlexFlow/build

export REALM_UCP_BOOTSTRAP_PLUGIN=$FLEXFLOW_DIR/deps/legion/lib/realm_ucp_bootstrap_mpi.so
export LD_LIBRARY_PATH=$FLEXFLOW_DIR/deps/legion/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FLEXFLOW_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/conda/envs/flexflow/lib:$LD_LIBRARY_PATH
export FF_LEGION_NETWORKS=ucx

# Create test prompt file
# mkdir -p ./FlexFlow/inference/prompt
# echo '["Three tips for staying healthy are: "]' > ./FlexFlow/inference/prompt/test.json

mpirun -x REALM_UCP_BOOTSTRAP_PLUGIN -x PATH -x LD_LIBRARY_PATH --hostfile ~/hostfile --mca btl_tcp_if_include ens5 -N 1 -np 2 gdb -ex run -ex bt --args /home/ubuntu/specinfer-ae/FlexFlow/build/inference/incr_decoding/incr_decoding -ll:cpu 8 -ll:util 8 -ll:gpu 4 -ll:fsize 20000 -ll:zsize 20000 -llm-model JackFram/llama-68m -prompt /home/ubuntu/specinfer-ae/FlexFlow/inference/prompt/test.json --max-requests-per-batch 1 -tensor-parallelism-degree 4 -pipeline-parallelism-degree 2 --fusion --disable-control-replication
