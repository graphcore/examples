# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#! /bin/bash

helpFunction()
{
   echo ""
   echo "Usage (int the case of 4 hosts): $0 -n host1,host2,host3,host4 " \
        "-s host0 -o interface1 -b interface2 -p partition_name -c allocation_name"
   echo -e "\t-n Hostnames/IPs of the hosts"
   echo -e "\t-s Hostname/IP of the controller server"
   echo -e "\t-p partition name"
   echo -e "\t-c allocation name"
   exit 1
}

while getopts "n:s:p:c:" opt
do
   case "$opt" in
      n ) hosts="$OPTARG" ;;
      s ) server="$OPTARG" ;;
      p ) partition="$OPTARG" ;;
      c ) cluster="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$hosts" ] || [ -z "$server" ] || [ -z "$partition" ] || [ -z "$allocation" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

commas="${hosts//[^,]}"
num=${#commas}

if [[ $num -eq 3 ]]
then
    echo "4 hosts are specified to use POD64."
    replicas=16
    instances=4
    batch_size=6
else
    echo "hosts are mal configured."
    exit 1
fi
# POPLAR options saves a bit of memory.
POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.hostSyncTimeout": 900}' \
poprun -vv \
        --num-replicas=16 \
        --num-instances=4 \
        --host=$hosts \
        --ipus-per-replica=4 \
        --vipu-server-host=$server \
        --vipu-allocation=$allocation \
        --vipu-server-timeout=3600 \
        --vipu-partition=$partition \
        --mpi-local-args=" -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x \
        IPUOF_VIPU_API_TIMEOUT=3600 -x POPLAR_LOG_LEVEL=WARN -x POPLAR_SDK_ENABLED -x POPLAR_ENGINE_OPTIONS" \
time=$(date "+%Y%m%d%H%M%S")
python main_pretrain.py --config vit_base_pod64 --data_path /path/to/imagenet1k 2>&1 | tee mae_pod64_$time.log
