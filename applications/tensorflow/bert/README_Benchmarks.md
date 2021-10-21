# Benchmarking on IPUs

This README describes how to run TensorFlow BERT models for throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/tensorflow/bert/README.md.

Follow the instructions at the same location for obtaining and processing the dataset. Ensure the $DATASETS_DIR environment variable points to the location of the tf_wikipedia directory created. The benchmark scripts run on a subset of the full Wikipedia dataset. 

Run the following commands from inside the applications/tensorflow/bert/ directory.

In order to get a clear indication of the average throughput of a full training run, calculate the average of the results from these scripts, ignoring the first 3 results.

## Training

### Pretrain BERT-Base Sequence Length 128

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config configs/pretrain_base_128_phase1.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_128_dup5_mask20/wiki_00_cleaned.tfrecord --num-train-steps 4
```

### Pretrain BERT-Base Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config configs/pretrain_base_384_phase2.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask58/wiki_00_cleaned.tfrecord --num-train-steps 4
```

### Pretrain BERT-Large Sequence Length 128

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config configs/pretrain_large_128_phase1.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_128_dup5_mask20/wiki_00_cleaned.tfrecord --num-train-steps 4
```

#### 1 x IPU-POD64


Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
```

Command:
```console
python run_pretraining.py --config configs/pretrain_large_128_phase1_POD64.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_128_dup5_mask20/*.tfrecord --save-path "checkpoint/phase1/"
```



#### 1 x IPU-POD128

$PARTITION, $IPUOF_VIPU_API_PARTITION_ID: ID of the Pod64 reconfigurable partition
$TCP_IF_INCLUDE: sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address.
$VIPU_SERVER_HOST: IP address as appropriate for the target hardware 
$HOSTS: IP address of the main host server


Command:
```console
poprun --host ${HOSTS} --num-ilds 2 --num-instances 2 --num-replicas 32 --ipus-per-replica 4 --vipu-server-host=${VIPU_SERVER_HOST} --reset-partition=yes --update-partition=yes --remove-partition=no --vipu-partition=${PARTITION} --print-topology=yes --mpi-global-args="--tag-output --mca btl_tcp_if_include ${TCP_IF_INCLUDE} --mca oob_tcp_if_include ${TCP_IF_INCLUDE}" --mpi-local-args="-x TF_CPP_VMODULE='poplar_compiler=0, poplar_executor=0' -x HOROVOD_LOG_LEVEL=WARN -x IPUOF_LOG_LEVEL=WARN -x POPLAR_LOG_LEVEL=WARN -x CPATH -x GCL_LOG_LEVEL=WARN -x TF_POPLAR_FLAGS=--executable_cache_path=${LOCAL_HOME}/exec_cache" python run_pretraining.py --config configs/pretrain_large_128_phase1_POD128.json --train-file "$DATASETS_DIR/tf_wikipedia/tokenised_128_dup5_mask20/*.tfrecord" 
```


### Pretrain BERT-Large Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config configs/pretrain_large_384_phase2.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask58/wiki_00_cleaned.tfrecord --num-train-steps 4
```

#### 1 x IPU-POD64

Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
```

Command:
```console
python run_pretraining.py --config configs/pretrain_large_384_phase2_POD64.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask58/*.tfrecord --init-checkpoint "checkpoint/phase1/ckpt-7031" --save-path "checkpoint/phase2/"
```

#### 1 x IPU-POD128

$PARTITION, $IPUOF_VIPU_API_PARTITION_ID: ID of the Pod64 reconfigurable partition
$TCP_IF_INCLUDE: sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address.
$VIPU_SERVER_HOST: IP address as appropriate for the target hardware 
$HOSTS: IP address of the main host server


Command:
```console
poprun --host ${HOSTS} --num-ilds 2 --num-instances 2 --num-replicas 32 --ipus-per-replica 4 --vipu-server-host=${VIPU_SERVER_HOST} --reset-partition=yes --update-partition=yes --remove-partition=no --vipu-partition=${PARTITION_NAME} --print-topology=yes --mpi-global-args="--tag-output --mca btl_tcp_if_include ${TCP_IF_INCLUDE} --mca oob_tcp_if_include ${TCP_IF_INCLUDE}" --mpi-local-args="-x TF_CPP_VMODULE='poplar_compiler=0, poplar_executor=0' -x HOROVOD_LOG_LEVEL=WARN -x IPUOF_LOG_LEVEL=WARN -x POPLAR_LOG_LEVEL=WARN -x CPATH -x GCL_LOG_LEVEL=WARN -x TF_POPLAR_FLAGS=--executable_cache_path=${LOCAL_HOME}/exec_cache" python run_pretraining.py --config configs/pretrain_large_384_phase2_POD128.json --train-file "$DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask58/*.tfrecord" --init-checkpoint "${PHASE1_CHECKPOINT}" 
```

