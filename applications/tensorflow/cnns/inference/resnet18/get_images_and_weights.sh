#!/usr/bin/env bash

RESOURCES_DIR='../../../../../utils/resources/'
GET_FILE='get.py'
IMAGES='images224.tar.gz'
WEIGHTS='ResNet18.tar.gz'


python "${RESOURCES_DIR}${GET_FILE}" ${IMAGES}
python "${RESOURCES_DIR}${GET_FILE}" ${WEIGHTS}

echo "Unpacking ${IMAGES}"
tar xzf ${IMAGES}
echo "Unpacking ${WEIGHTS}"
tar xzf ${WEIGHTS}
