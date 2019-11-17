#!/usr/bin/env bash

RESOURCES_DIR='../../../utils/resources/'
GET_FILE='get.py'
DATAFILE='returns_and_features_for_mcmc.tar.gz'


python "${RESOURCES_DIR}${GET_FILE}" ${DATAFILE}

echo "Unpacking ${DATAFILE}"
tar -xzf ${DATAFILE}
