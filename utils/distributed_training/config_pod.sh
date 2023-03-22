#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All Rights Reserved.
# This script sources environment variables for pod configuration.
# It should be launched from the main host
# and it will sync all the data to the other specified hosts.

display_usage() {

	echo "Usage: . config_pod.sh partion-name HOST1 HOST2 ..."
      echo "partion-name : name of the partition accessible with vipu-admin list partitions (mandatory argument)"
	echo "HOSTn: a list of hosts (different from the main host) on which the data will be copied from the main host."
      echo "If Hostn are not specified, the pod will be configured to run only on the main host."
      echo "ssh-copy-id should have been performed before once from all the other hosts via the other script copy_ssh.sh."

}

if [ "${BASH_SOURCE[0]}" == "${0}" ]
then
    display_usage
    return 1
fi

# This one is not be stable as the printing of vipu-admin list partitions might change
# export VIPU_PARTITION_NAME="$(vipu-admin list partitions | awk '{print $5}' | sed -n '3 p')"

if [[ $# < 1 ]]; then
   display_usage
   return 1
fi

export VIPU_PARTITION_NAME=$1
shift

export IPUOF_VIPU_API_HOST="$(vipu-admin --server-version | sed -n '2 p' | awk '{ print $2; }' | awk -F: '{print $1}')"

export IPUOF_VIPU_API_PARTITION_ID=$VIPU_PARTITION_NAME

echo "Setting IPUOF_VIPU_API_PARTITION_ID to ${IPUOF_VIPU_API_PARTITION_ID}"

if [ "${IPUOF_VIPU_API_HOST}" == "localhost" ]; then
    export VIPU_SERVER_IP="$(getent hosts ${HOSTNAME} | awk '{ print $1 }')"
else
    export VIPU_SERVER_IP="$(getent hosts ${IPUOF_VIPU_API_HOST} | awk '{ print $1 }')"
fi

echo "Host ip address: $VIPU_SERVER_IP"

NETMASK_CIDR__IP_PART=(${VIPU_SERVER_IP//./ })

NETMASK_CIDR="${NETMASK_CIDR__IP_PART[0]}.${NETMASK_CIDR__IP_PART[1]}.0.0/16"

LOCAL_HOME="/localdata/${USER}"

if [ -d "${LOCAL_HOME}/exec_cache" ]
then
    echo "Directory ${LOCAL_HOME}/exec_cache exists."
else
    echo "Directory ${LOCAL_HOME}/exec_cache does not exists, creating one."
    mkdir ${LOCAL_HOME}/exec_cache
fi

unset MPI_GLOBAL_ARGS
unset MPI_LOCAL_ARGS
unset HOSTS_LIST
unset HOSTS
unset LISTED_HOST
# MPI options

export MPI_GLOBAL_ARGS="--tag-output --mca btl_tcp_if_include ${NETMASK_CIDR} --mca oob_tcp_if_include ${NETMASK_CIDR}"
export MPI_LOCAL_ARGS="-x TF_CPP_VMODULE='poplar_compiler=0, poplar_executor=0' -x HOROVOD_LOG_LEVEL=WARN -x IPUOF_LOG_LEVEL=WARN -x POPLAR_LOG_LEVEL=WARN -x CPATH -x TF_POPLAR_FLAGS=--executable_cache_path=${LOCAL_HOME}/exec_cache"

echo "Variables MPI_GLOBAL_ARGS and MPI_LOCAL_ARGS have been set up"


sync_venvs_and_code() {
   for host in ${HOSTS}; do
      echo "Syncing local code with ${host}"
      rsync --stats -av "${LOCAL_HOME}/public_examples" "${host}:${LOCAL_HOME}/"
      echo "Syncing local venvs with ${host}"
      rsync --stats -av "${LOCAL_HOME}/venvs" "${host}:${LOCAL_HOME}/"
      echo "Syncing local SDK with ${host}"
      rsync --stats -av "${LOCAL_HOME}/sdks" "${host}:${LOCAL_HOME}/"
   done
}

sync_imagenet() {
  if [ -d "/localdata/datasets/imagenet-data" ]; then
      for host in ${HOSTS}; do
                  echo "Syncing Imagenet with ${host}"
                  rsync --stats -av "/localdata/datasets/imagenet-data" "${host}:/localdata/datasets/"
      done
  fi
}

let NUM_HOSTS=$#+1

echo "Number of host (including main host) ${NUM_HOSTS}"

if [[ $# -gt 0 ]]; then
      HOSTS="$@"
      LISTED_HOST="${HOSTS}"
      echo "Listed hosts : ${LISTED_HOST}"

      let i=0
      for host in ${HOSTS}; do
            HOSTIP="$(getent hosts ${host} | awk '{ print $1 }')"
            if [ -z "$HOSTIP" ]
                  then
                        echo "${host} is a bad host name"
                        return 1
                  else
                        echo "$host ip is $HOSTIP"
                        LISTED_HOST[$i]=$HOSTIP
            fi
            let i+=1
      done

      LISTED_HOST="${LISTED_HOST[*]}"

      export HOSTS_LIST="${VIPU_SERVER_IP},${LISTED_HOST// /,}"

      if [ -d "${LOCAL_HOME}/venvs" ]
      then
      echo "Directory ${LOCAL_HOME}/venvs exists."
      else
      echo "Directory ${LOCAL_HOME}/venvs does not exists, please change your local popsdk envs."
      return 1
      fi

      if [ -d "${LOCAL_HOME}/sdks" ]
      then
      echo "Directory ${LOCAL_HOME}/sdks exists."
      else
      echo "Directory ${LOCAL_HOME}/sdks does not exists, please change your local popsdk sdks."
      return 1
      fi

      if [ -d "${LOCAL_HOME}/public_examples" ]
      then
      echo "Directory ${LOCAL_HOME}/public_examples exists."
      else
      echo "Directory ${LOCAL_HOME}/public_examples does not exists, please change your local public examples."
      return 1
      fi

      sync_venvs_and_code
      sync_imagenet
else
      export HOSTS_LIST="${VIPU_SERVER_IP}"
fi

echo "List of host IPs separated by comma: ${HOSTS_LIST} and exported to HOST_LIST"
