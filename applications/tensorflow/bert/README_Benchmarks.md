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
python run_pretraining.py --config configs/pretrain_base_128.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_128_dup5_mask20/wiki_00_cleaned.tfrecord --num-train-steps 4
```

### Pretrain BERT-Base Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config configs/pretrain_base_384.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask56/wiki_00_cleaned.tfrecord --num-train-steps 4
```

### Pretrain BERT-Large Sequence Length 128

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config configs/pretrain_large_128_phase1.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_128_dup5_mask20/wiki_00_cleaned.tfrecord --num-train-steps 4
```

#### 1 x IPU-POD64

xxx.xxx.xxx.xxx: Replace with IP addresses as appropriate for the target hardware '--mca btl_tcp_if_include xxx.xxx.xxx.0/xx' sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address. Replace 'pod64_partition_name' with the name of your POD64 partition. Ensure the dataset, application and SDK are synced across hosts.

Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
```

Command:
```console
poprun --host xxx.xxx.xx1,xxx.xxx.xx2,xxx.xxx.xx3,xxx.xxx.xx4 --mpi-global-args="--allow-run-as-root --tag-output --mca oob_tcp_if_include xxx.xxx.xxx.0/xx --mca btl_tcp_if_include xxx.xxx.xxx.0/xx" --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=WARN -x TF_POPLAR_FLAGS -x DATASETS_DIR" --num-ilds 1 --num-instances 4 --num-replicas 16 --ipus-per-replica 4 --numa-aware=yes --vipu-server-host=xxx.xxx.xx1 --reset-partition=no --update-partition=no --vipu-partition=pod64_partition_name --vipu-server-timeout=600 python run_pretraining.py --config configs/pretrain_large_128_phase1_POD64.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_128_dup5_mask20/*.tfrecord --wandb
```

### Pretrain BERT-Large Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config configs/pretrain_large_384_phase2.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask56/wiki_00_cleaned.tfrecord --num-train-steps 4
```

#### 1 x IPU-POD64

xxx.xxx.xxx.xxx: Replace with IP addresses as appropriate for the target hardware '--mca btl_tcp_if_include xxx.xxx.xxx.0/xx' sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address. Replace 'pod64_partition_name' with the name of your POD64 partition. Ensure the dataset, application and SDK are synced across hosts.

Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
```

Command:
```console
poprun --host xxx.xxx.xx1,xxx.xxx.xx2,xxx.xxx.xx3,xxx.xxx.xx4 --mpi-global-args="--allow-run-as-root --tag-output --mca oob_tcp_if_include xxx.xxx.xxx.0/xx --mca btl_tcp_if_include xxx.xxx.xxx.0/xx" --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=WARN -x TF_POPLAR_FLAGS -x DATASETS_DIR" --num-ilds 1 --num-instances 4 --num-replicas 16 --ipus-per-replica 4 --numa-aware=yes --vipu-server-host=xxx.xxx.xx1 --reset-partition=no --update-partition=no --vipu-partition=pod64_partition_name --vipu-server-timeout=600 python run_pretraining.py --config configs/pretrain_large_384_phase2_POD64.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask56/wiki_00_cleaned.tfrecord --wandb
```
