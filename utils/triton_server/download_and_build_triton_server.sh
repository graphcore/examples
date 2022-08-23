#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#Build triton backend
TAG=r21.05

cd $1 && \
    git clone https://github.com/triton-inference-server/server.git && \
        cd server && \
        git checkout ${TAG} && \
        mkdir -p mybuild/tritonserver/install && \
        python3 build.py \
            --build-dir mybuild \
            --no-container-build \
            --endpoint=grpc \
            --enable-logging \
            --enable-stats \
            --cmake-dir `pwd`/build \
            --repo-tag=common:${TAG} \
            --repo-tag=core:${TAG} \
            --repo-tag=backend:${TAG} \
            --repo-tag=thirdparty:${TAG}
