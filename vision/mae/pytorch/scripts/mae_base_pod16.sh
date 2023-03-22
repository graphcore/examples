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
   echo "Usage: $0 -s host0 -p partition_name"
   echo -e "\t-s Hostname/IP of the controller server"
   echo -e "\t-p partition name"
   exit 1
}

while getopts "n:s:p:c:" opt
do
   case "$opt" in
      s ) server="$OPTARG" ;;
      p ) partition="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$server" ] || [ -z "$partition" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

echo "Training on a single host using POD16."

export IPUOF_VIPU_API_HOST=$server
export IPUOF_VIPU_API_PARTITION_ID=$partition
time=$(date "+%Y%m%d%H%M%S")
python main_pretrain.py --config vit_base_pod16 --data_path /path/to/imagenet1k 2>&1 | tee mae_pod16_$time.log
