#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All Rights Reserved.

display_usage() {

	echo "Usage: $0 HOST1 HOST2 ..."
	echo "HOSTn: a list of remote hosts to which the ssh id will be copied."
   echo "Should be run once from  all the hosts connected together via MPI (poprun)."
   echo "At least one host should be specified"

}

if [[ $# -lt 1 ]]; then   
   display_usage
   exit 1
fi

HOSTS="$@"

for HOST in ${HOSTS}; do	
      echo "ssh-id copied into ${HOST}"
      ssh-copy-id "${HOST}"
      ssh-keyscan -H "${HOST}" >> "${HOME}/.ssh/known_hosts"
      HOSTIP="$(getent hosts ${HOST} | awk '{ print $1 }')"
      ssh-keyscan -H "${HOSTIP}" >> "${HOME}/.ssh/known_hosts"
done
