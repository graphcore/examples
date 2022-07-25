#!/usr/bin/env bash

RESOURCES_DIR='../../../../utils/resources/'
GET_FILE='get.py'
IMAGES='images224.tar.gz'

python "${RESOURCES_DIR}${GET_FILE}" ${IMAGES}
echo "Unpacking ${IMAGES}"
tar xzf ${IMAGES}
