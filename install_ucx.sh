#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

wget https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz
tar -xf ucx-1.15.0.tar.gz
cd ucx-1.15.0
export CUDA_PATH=/usr/local/cuda
export PREFIX=$PWD/install
./contrib/configure-release-mt --prefix="$PREFIX" --without-go --enable-mt --with-cuda="$CUDA_PATH"
make -j install

export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH