# Benchmarking on IPUs

This README describes how to run the MiniDALL-E model for throughput benchmarking on the Mk2 IPU.


## Preparation

Follow the installation instructions in applications/pytorch/miniDALL-E/README.md.

## Training

### Pytorch MiniDALL-E Generated data training

#### 1 x IPU-M2000

Command:
```console
    python3 train.py \
        --config L16 \
        --generated-data True \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True \
```

#### 1 x IPU-POD16

Command:
```console
    python3 train.py \
        --config L16_POD16 \
        --generated-data True \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True \
```

#### 1 x IPU-POD64

Command:
```console
    python3 train.py \
        --config L16_POD64 \
        --generated-data True \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True \
```

#### 1 x IPU-POD256

Command:
```console
    poprun \
        -vv \
        --host $HOSTS \
        --vipu-partition=$PARTITION \
        --vipu-server-host=$VIPU_SERVER_HOST \
        --update-partition=yes \
        --print-topology=yes \
        --mpi-global-args="--tag-output --allow-run-as-root --mca oob_tcp_if_include $TCP_IF_INCLUDE --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
        --num-replicas=64 \
        --numa-aware=yes \
        --num-ilds=4 \
        --num-instances=16 \
        --ipus-per-replica=4 \
        --executable-cache-path=$PYTORCH_EXE_DIR \
    python3 train.py \
        --config L16_POD256 \
        --generated-data True \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True \
        --dataloader-workers 32 \
```

### Pytorch MiniDALL-E Real data training

#### 1 x IPU-M2000

Command:
```console
    python3 train.py \
        --config L16 \
        --input-folder $DATASETS_DIR/coco \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True \
```

#### 1 x IPU-POD16

Command:
```console
    python3 train.py \
        --config L16_POD16 \
        --input-folder $DATASETS_DIR/coco \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True \
```

#### 1 x IPU-POD64

Command:
```console
    python3 train.py \
        --config L16_POD64 \
        --input-folder $DATASETS_DIR/coco \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True \
```

#### 1 x IPU-POD256

Command:
```console
    poprun \
        -vv \
        --host $HOSTS \
        --vipu-partition=$PARTITION \
        --vipu-server-host=$VIPU_SERVER_HOST \
        --update-partition=yes \
        --print-topology=yes \
        --mpi-global-args="--tag-output --allow-run-as-root --mca oob_tcp_if_include $TCP_IF_INCLUDE --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
        --num-replicas=64 \
        --numa-aware=yes \
        --num-ilds=4 \
        --num-instances=16 \
        --ipus-per-replica=4 \
        --executable-cache-path=$PYTORCH_EXE_DIR \
    python3 train.py \
        --config L16_POD256 \
        --input-folder $DATASETS_DIR/coco \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True \
        --dataloader-workers 32 \
```


## Inference
