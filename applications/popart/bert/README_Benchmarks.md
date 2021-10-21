# Benchmarking on IPUs

This README describes how to run PopART BERT models for throughput and inference benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/popart/bert/README.md, including the instructions on obtaining and pre-processing the dataset. Ensure the $DATASETS_DIR environment variable points to the location of the /wikipedia/ directory.

## Training

Run the following command lines from inside the applications/popart/bert directory:

### BERT-Large Phase 1 Pre-training Sequence length 128

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/pretrain_large_128.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

#### 1 x IPU-POD64

Command:
```console
python bert.py --config configs/mk2/pretrain_large_128_POD64.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_*_tokenised --checkpoint-dir "checkpoint/phase1"
```

#### 1 x IPU-POD128
export POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "1200", "target.syncReplicasIndependently": "true"}'
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
       --vipu-cluster=c128 \
       --update-partition=yes \
       --remove-partition=no \
       --reset-partition=no \
       --print-topology=yes \
       --vipu-server-timeout=1200 \
       --mpi-global-args="--tag-output \
                          --allow-run-as-root \
                          --mca btl_tcp_if_include $TCP_IF_INCLUDE \
                          --mca oob_tcp_if_include $TCP_IF_INCLUDE" \
       --mpi-local-args="-x OPAL_PREFIX \
                         -x CPATH \
                         -x IPUOF_VIPU_API_TIMEOUT=1200 \
                         -x POPLAR_ENGINE_OPTIONS \
                         -x HOROVOD_STALL_CHECK_TIME_SECONDS \
                         -x HOROVOD_POPART_BROADCAST_TIMEOUT" \
python3 bert.py --config configs/mk2/pretrain_large_128.json  \
                --replication-factor 16 \
                --loss-scaling 4096 \
                --replicated-tensor-sharding True \
                --input-files $DATASETS_DIR/wikipedia/AA/sequence_128/* 
```

### BERT-Large Phase 2 Pre-training Sequence length 384

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/pretrain_large_384.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

#### 1 x IPU-POD64

Command:
```console
python bert.py --config configs/mk2/pretrain_large_384_POD64.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --epochs 2 
```

#### 1 x IPU-POD128

export POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "1200", "target.syncReplicasIndependently": "true"}'
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
       --vipu-cluster=c128 \
       --update-partition=yes \
       --remove-partition=no \
       --reset-partition=no \
       --print-topology=yes \
       --vipu-server-timeout=1200 \
       --mpi-global-args="--tag-output \
                          --allow-run-as-root \
                          --mca btl_tcp_if_include $TCP_IF_INCLUDE \
                          --mca oob_tcp_if_include $TCP_IF_INCLUDE" \
       --mpi-local-args="-x OPAL_PREFIX \
                         -x CPATH \
                         -x IPUOF_VIPU_API_TIMEOUT=1200 \
                         -x POPLAR_ENGINE_OPTIONS \
                         -x HOROVOD_STALL_CHECK_TIME_SECONDS \
                         -x HOROVOD_POPART_BROADCAST_TIMEOUT" \
python3 bert.py --config configs/mk2/pretrain_large_384.json  \
                --replication-factor 16 \
                --replicated-tensor-sharding True \
                --input-files $DATASETS_DIR/wikipedia/AA/sequence_384/* \
                --wandb \
                --onnx-checkpoint checkpoints/mk2/pretrain_large_rank_0/21-10-06-01-20-46/model.onnx
```

### BERT-Base Phase 1 Pre-training Sequence length 128

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/pretrain_base_128.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

### BERT-Base Phase 2 Pre-training Sequence length 384

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/pretrain_base_384.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

### BERT Large SQuAD Sequence length 384

#### 1 x IPU-POD16

Command:
```console
python run_squad.py --squad-do-validation False --config squad_large_384_POD16 --num-epochs 1
```

## Inference

Follow the installation instructions in applications/popart/bert/README.md.

We use generated data to obtain the inference throughput, so you do not need to
use pre-trained weights or download the SQuAD dataset to reproduce the benchmark results.

Run the following command lines from inside the applications/popart/bert directory:

### BERT-Large SQuAD v1.1 Inference Sequence length 128

#### 1 x IPU-M2000

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
mpirun --tag-output --np 4 --allow-run-as-root python bert.py --config configs/mk2/squad_large_128_inf.json           --micro-batch-size {batchsize} --generated-data=true --epochs-inference 20 --input-files=$DATASETS_DIR/squad/dev-v1.1.json
```

Set --micro-batch-size to 1, 2 or 3.

### BERT-Base SQuAD v1.1 Inference Sequence length 128

#### 1 x IPU-M2000

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
mpirun --tag-output --np 4 --allow-run-as-root python bert.py --config configs/mk2/squad_base_128_inf.json --micro-batch-size {batchsize} --generated-data=true --epochs-inference 10 --input-files=$DATASETS_DIR/squad/dev-v1.1.json
```

Set --micro-batch-size to 1, 2, 4, 8, 16, 32, 64, or 80 
for micro-batch-size = 80, also set --available-memory-proportion 0.55
