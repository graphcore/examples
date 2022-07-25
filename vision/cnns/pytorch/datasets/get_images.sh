#!/usr/bin/env bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

RESOURCES_DIR='../../../../utils/resources/'
GET_FILE='get.py'
IMAGES='images224.tar.gz'

python "${RESOURCES_DIR}${GET_FILE}" ${IMAGES}
echo "Unpacking ${IMAGES}"
mkdir -p ../data
tar xzf ${IMAGES} -C ../data
rm ${IMAGES}