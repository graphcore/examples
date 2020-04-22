#!/usr/bin/env sh
# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Checking..."
FILES="t10k-images-idx3-ubyte t10k-labels-idx1-ubyte train-images-idx3-ubyte train-labels-idx1-ubyte"
sum $FILES | cmp mnist.checksums && { echo "MNIST data already downloaded"; exit; }

echo "Downloading..."
BASE=http://yann.lecun.com/exdb/mnist

mkdir -p data
for f in $FILES
do
  curl $BASE/${f}.gz | gunzip > data/$f
done

echo "Done."
