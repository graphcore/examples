#!/usr/bin/env bash
# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

RESOURCES_DIR='../../../../utils/resources/'
GET_FILE='get.py'
IMAGES='mnist-data.tar.gz'

# Cleanup existing files
rm -rf data
rm *.pvti

# Retrieve MNIST data tarball
python "${RESOURCES_DIR}${GET_FILE}" ${IMAGES}

# Unpack tarball
echo "Unpacking ${IMAGES}"
tar xzf ${IMAGES}
rm ${IMAGES}
