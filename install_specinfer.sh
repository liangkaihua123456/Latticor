#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export FF_LEGION_NETWORKS=ucx
export UCX_DIR="$PWD/ucx-1.15.0/install"
BUILD_TYPE=${BUILD_TYPE:-Release}

cd FlexFlow
mkdir -p build
cd build
../config/config.linux
make -j install
