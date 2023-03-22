#!/usr/bin/env bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

RESOURCES_DIR='../../../../utils/resources/'
GET_FILE='get.py'
IMAGES='mnist-data.tar.gz'


python3 "${RESOURCES_DIR}${GET_FILE}" ${IMAGES}

echo "Unpacking ${IMAGES}"
tar xzf ${IMAGES}
