#! /bin/bash

helpFunction()
{
   echo ""
   echo "Usage (int the case of 4 hosts): $0 -n host1,host2,host3,host4 " \
        "-s host0 -o interface1 -b interface2 -p partition_name -c cluster_name"
   echo -e "\t-n Hostnames/IPs of the hosts"
   echo -e "\t-s Hostname/IP of the controller server"
   echo -e "\t-o network interface of the control plane"
   echo -e "\t-b network interface of the data plane"
   echo -e "\t-p partition name"
   echo -e "\t-c cluster name"
   exit 1
}

while getopts "n:s:o:b:p:c:" opt
do
   case "$opt" in
      n ) hosts="$OPTARG" ;;
      s ) server="$OPTARG" ;;
      o ) interface1="$OPTARG" ;;
      b ) interface2="$OPTARG" ;;
      p ) partition="$OPTARG" ;;
      c ) cluster="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done

if [ -z "$hosts" ] || [ -z "$server" ] || [ -z "$interface1" ] || \
    [ -z "$interface2" ] || [ -z "$partition" ] || [ -z "$cluster" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

commas="${hosts//[^,]}"
num=${#commas}

if [[ $num -eq 3 ]]
then
    echo "4 hosts are specified to use POD64."
    ilds=8
    replicas=16
    instances=8
    accumulation=341
    batch_size=12
    rebatched_worker_size=1364
    workers=32
    iterations=256
else
    echo "hosts are mal configured."
    exit 1
fi

time=$(date "+%Y%m%d%H%M%S")
export IPUOF_VIPU_API_TIMEOUT=1000
export POPLAR_LOG_LEVEL=WARN
export POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "3000"}'

poprun 
      --host $hosts \
      --vv \
      --reset-partition=no \
      --vipu-server-host=$server \
      --num-ilds=$ilds \
      --mpi-global-args="--output-filename output_poprun --mca oob_tcp_if_include $interface1 --mca btl_tcp_if_include $interface2" \
      --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x IPUOF_VIPU_API_TIMEOUT -x POPLAR_LOG_LEVEL -x POPLAR_SDK_ENABLED -x POPLAR_ENGINE_OPTIONS" \
      --vipu-partition=$partition \
      --vipu-cluster=$cluster \
      --ipus-per-replica 4 \
      --num-replicas=$replicas \
      --num-instances=$instances \
      --vipu-server-timeout=6000 \
      python pretrain.py --config b16_in1k_pretrain \
            --dataset generated \
            --iterations $iterations \
            --gradient-accumulation $accumulation \
            --micro-batch-size $batch_size \
            --rebatched-worker-size $rebatched_worker_size \
            --dataloader-workers $workers \
            --enable-rts true \
            --optimizer-state-offchip false \
            --byteio true 2>&1 | tee vit_scalability_$time.log
