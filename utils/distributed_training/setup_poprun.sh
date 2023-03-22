#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All Rights Reserved.
# This script configure poprun options and export them to POPRUN_PREFIX
# assuming that config_pod.sh script has been run successfully


display_usage() {
	echo "Usage: . set_up_poprun.sh ALLOCATION_NAME NUM_INSTANCES NUM_REPLICAS IPUS_PER_REPLICA"
      echo "ALLOCATION_NAME can be retrieved with vipu-admin list partition. "
}

if [ "${BASH_SOURCE[0]}" == "${0}" ]
then
    display_usage
    return 1
fi

if [[ $# != 4 ]]; then
   display_usage
   return 1
fi

ALLOCATION_NAME="$1"
NUM_INSTANCES="$2"
NUM_REPLICAS="$3"
IPUS_PER_REPLICA="$4"

if [ -z "$HOSTS_LIST" ]
then
      echo "\$HOSTS_LIST is empty : run the script config_pod.sh before this one"
      return 1
elif [ -z "$VIPU_SERVER_IP" ] || [ -z "$VIPU_PARTITION_NAME" ]
then
      echo "\$VIPU_SERVER_IP or \$VIPU_PARTITION_NAME is empty : run the script config_pod.sh before this one"
      return 1
elif [ -z "$MPI_GLOBAL_ARGS" ] || [ -z "$MPI_LOCAL_ARGS" ]
then
      echo "\$MPI_GLOBAL_ARGS or \$MPI_LOCAL_ARGS is empty : run the script config_pod.sh before this one"
      return 1
fi


HOSTS_LIST_ARRAY=(${HOSTS_LIST//,/ })

HOSTS_NUM=${#HOSTS_LIST_ARRAY[@]}

echo "The number of host is $HOSTS_NUM"


echo "NUM_INSTANCES is $NUM_INSTANCES"
echo "NUM_REPLICAS is $NUM_REPLICAS"
echo "IPUS_PER_REPLICA is $IPUS_PER_REPLICA"


POPRUN_OPTIONS="--only-output-from-instance 0 --host ${HOSTS_LIST} --num-instances ${NUM_INSTANCES} --num-replicas ${NUM_REPLICAS} --ipus-per-replica ${IPUS_PER_REPLICA} --vipu-server-host=${VIPU_SERVER_IP} --vipu-allocation=${ALLOCATION_NAME} --reset-partition=no --update-partition=yes --remove-partition=no --vipu-partition=${VIPU_PARTITION_NAME} --print-topology=yes  --vipu-server-timeout=500"

unset poprun_prefix
unset POPRUN_PREFIX

export POPRUN_PREFIX="poprun ${POPRUN_OPTIONS} \
--mpi-global-args=\"${MPI_GLOBAL_ARGS}\" \
--mpi-local-args=\"${MPI_LOCAL_ARGS}\" "

poprun_prefix () {
poprun ${POPRUN_OPTIONS} \
--mpi-global-args="${MPI_GLOBAL_ARGS}" \
--mpi-local-args="${MPI_LOCAL_ARGS}" \
 $*;
 }

export -f poprun_prefix



echo "The poprun prefix will be $POPRUN_PREFIX ."
echo "You can now use poprun_prefix as a prefix to your training script like : "
echo "poprun_prefix phyton3 train.py your_options ..."
