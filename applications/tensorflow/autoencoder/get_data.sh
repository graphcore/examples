#!/usr/bin/env bash

RESOURCES_DIR='../../../utils/resources/'
GET_FILE='get.py'
DATAFILE='netflix_data.tar.gz'


python "${RESOURCES_DIR}${GET_FILE}" ${DATAFILE}

echo "Unpacking ${DATAFILE}"
tar -xzf ${DATAFILE}
