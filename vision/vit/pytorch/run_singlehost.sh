#! /bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -s host0 -p partition_name"
   echo -e "\t-s Hostname/IP of the controller server"
   echo -e "\t-p partition name"
   exit 1
}

while getopts "n:s:o:b:p:c:" opt
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
python pretrain.py --config b16_in1k_pretrain \
        --dataset generated \
        --iterations 64 \
        --gradient-accumulation 1365 \
        --micro-batch-size 12 \
        --rebatched-worker-size 1365 \
        --replication-factor 4 \
        --dataloader-workers 64 \
        --enable-rts true \
        --optimizer-state-offchip false \
        --byteio true  2>&1 | tee vit_scalability_$time.log
