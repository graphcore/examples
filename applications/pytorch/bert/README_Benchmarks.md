# Benchmarking on IPUs

This README describes how to run PyTorch BERT models for throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/pytorch/bert/README.md.

Follow the instructions at the same location for obtaining and processing the dataset. Ensure the $DATASETS_DIR environment variable points to the location of the dataset.

Run the following commands from inside the applications/pytorch/bert/ directory.

## Training

### Pretrain BERT-Base Sequence Length 128

#### 1 x IPU-POD16

Command:
```console
python3 run_pretraining.py \
   --config pretrain_base_128 \
   --training-steps 10 \
   --input-file $DATASETS_DIR/wikipedia/128/wiki_1[0-1]*.tfrecord \
   --disable-progress-bar
```

### Pretrain BERT-Base Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python3 run_pretraining.py \
   --config pretrain_base_384 \
   --training-steps 10 \
   --input-file $DATASETS_DIR/wikipedia/384/wiki_1[0-1]*.tfrecord \
   --disable-progress-bar
```

### Pretrain BERT-Large Sequence Length 128

#### 1 x IPU-POD16

Command:
```console
python3 run_pretraining.py \
   --config pretrain_large_128 \
   --training-steps 10 \
   --input-file $DATASETS_DIR/wikipedia/128/wiki_1[0-1]*.tfrecord \
   --disable-progress-bar
```

#### 1 x IPU-POD64

Command:
```console
python3 run_pretraining.py \
   --config pretrain_large_128_POD64 \
   --training-steps 10 \
   --input-file $DATASETS_DIR/wikipedia/128/wiki_1[0-1]*.tfrecord \
   --disable-progress-bar
```

#### 1 x IPU-POD128

#### 1 x IPU-POD128

export POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "600", "target.syncReplicasIndependently": "true"}'
export HOROVOD_STALL_CHECK_TIME_SECONDS=120
export HOROVOD_POPART_BROADCAST_TIMEOUT=120

$PARTITION, $IPUOF_VIPU_API_PARTITION_ID: ID of the Pod64 reconfigurable partition
$TCP_IF_INCLUDE: sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address.
$VIPU_SERVER_HOST: IP address as appropriate for the target hardware 
$HOSTS: IP address of the main host server

Command:
```console
poprun -vv --num-instances=2 --num-replicas=32 \
       --num-ilds=2 \
       --ipus-per-replica=4 \
       --vipu-server-host="$VIPU_SERVER_HOST" \
       --host=$HOSTS \
       --vipu-partition=gcl128 \
       --vipu-cluster=gcl \
       --update-partition=yes \
       --remove-partition=no \
       --reset-partition=no \
       --print-topology=yes \
       --mpi-global-args="--tag-output \
                          --allow-run-as-root \
                          --mca btl_tcp_if_include $TCP_IF_INCLUDE \
                          --mca oob_tcp_if_include $TCP_IF_INCLUDE \
       --mpi-local-args="-x OPAL_PREFIX \
                         -x CPATH \
                         -x IPUOF_VIPU_API_TIMEOUT=600 \
                         -x POPLAR_ENGINE_OPTIONS \
                         -x HOROVOD_STALL_CHECK_TIME_SECONDS \
                         -x HOROVOD_POPART_BROADCAST_TIMEOUT" \
python run_pretraining.py --config pretrain_large_128_POD64 --replication-factor 32 \
                          --loss-scaling 8192 \
                          --gradient-accumulation 256 \
                          --checkpoint-output-dir large-128-POD128-poprun-rep32 \
                          --checkpoint-steps 1000 \
                          --replicated-tensor-sharding True \
                          --random-seed 1984 \
                          --input-files  $DATASETS_DIR/wikipedia/torch_bert/128/*.tfrecord 

```

### Pretrain BERT-Large Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python3 run_pretraining.py \
   --config pretrain_large_384 \
   --training-steps 10 \
   --input-file $DATASETS_DIR/wikipedia/384/wiki_1[0-1]*.tfrecord \
   --disable-progress-bar
```

#### 1 x IPU-POD64

Command:
```console
python3 run_pretraining.py \
   --config pretrain_large_384_POD64 \
   --training-steps 10 \
   --input-file $DATASETS_DIR/wikipedia/384/wiki_1[0-1]*.tfrecord \
   --disable-progress-bar
```

#### 1 x IPU-POD128

export POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "600", "target.syncReplicasIndependently": "true"}'
export HOROVOD_STALL_CHECK_TIME_SECONDS=120
export HOROVOD_POPART_BROADCAST_TIMEOUT=120

$PARTITION, $IPUOF_VIPU_API_PARTITION_ID: ID of the Pod64 reconfigurable partition
$TCP_IF_INCLUDE: sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address.
$VIPU_SERVER_HOST: IP address as appropriate for the target hardware 
$HOSTS: IP address of the main host server

Command:
```console
poprun -vv --num-instances=2 --num-replicas=32 \
       --num-ilds=2 \
       --ipus-per-replica=4 \
       --vipu-server-host="$VIPU_SERVER_HOST" \
       --host=$HOSTS\
       --vipu-partition=gcl128 \
       --vipu-cluster=gcl \
       --update-partition=yes \
       --remove-partition=no \
       --reset-partition=no \
       --print-topology=yes \
       --mpi-global-args="--tag-output \
                          --allow-run-as-root \git add 
                          --mca btl_tcp_if_include $TCP_IF_INCLUDE \
                          --mca oob_tcp_if_include $TCP_IF_INCLUDE" \
       --mpi-local-args="-x OPAL_PREFIX \
                         -x CPATH \
                         -x IPUOF_VIPU_API_TIMEOUT=600 \
                         -x POPLAR_ENGINE_OPTIONS \
                         -x HOROVOD_STALL_CHECK_TIME_SECONDS \
                         -x HOROVOD_POPART_BROADCAST_TIMEOUT" \
python run_pretraining.py --config pretrain_large_384_POD64 --replication-factor 32 \
                          --loss-scaling 8192 \
                          --gradient-accumulation 256 \
                          --checkpoint-output-dir large-384-POD128-poprun-rep32-RTS \
                          --checkpoint-steps 1000 \
                          --replicated-tensor-sharding True \
                          --pretrained-checkpoint large-128-POD128-poprun-rep32-RTS/step_7037 \
                          --input-files $DATASETS_DIR/torch_bert/384/*.tfrecord 
```


### SQuAD BERT-Large Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python run_squad.py --config squad_large_384 --squad-do-validation False
```
