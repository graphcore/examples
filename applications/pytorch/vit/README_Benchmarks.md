# Benchmarking on IPUs

This README describes how to run Pytorch ViT models for throughput benchmarking on the IPU, for both training and inference.

## Preparation

Follow the environment setup instructions in applications/pytorch/vit/README.md.

Follow the dataset Preparation instructions in applications/pytorch/vit/README.md.

## Training

Run the following command lines from inside the applications/pytorch/vit directory.

### ViT finetune training generated

#### 1 x IPU-M2000

Command:
```console
python3 finetune.py \
    --config b16_imagenet1k \
    --training-steps 6 \
    --replication-factor 1 \
    --gradient-accumulation 120 \
    --optimizer-state-offchip \
    --enable-rts False \
    --dataset generated \
    --checkpoint-output-dir "" \
    --executable-cache-dir PYTORCH_EXE_DIR \
```

#### 1 x IPU-POD16

Command:
```console
python3 finetune.py \
    --config b16_imagenet1k \
    --training-steps 6 \
    --dataset generated \
    --checkpoint-output-dir "" \
    --executable-cache-dir PYTORCH_EXE_DIR \
```

### ViT pretrain training generated

#### 1 x IPU-M2000

Command:
```console
python3 pretrain.py \
    --config b16_in1k_pretrain \
    --replication-factor 1 \
    --iterations 64 \
    --gradient-accumulation 5460 \
    --micro-batch-size 12 \
    --rebatched-worker-size 2048 \
    --dataloader-workers 64 \
    --optimizer-state-offchip false \
    --byteio true \
    --dataset generated \
```

#### 1 x IPU-POD16

Command:
```console
python3 pretrain.py \
    --config b16_in1k_pretrain \
    --iterations 64 \
    --gradient-accumulation 1365 \
    --micro-batch-size 12 \
    --rebatched-worker-size 2048 \
    --dataloader-workers 64 \
    --optimizer-state-offchip false \
    --byteio true \
    --dataset generated \
```
#### 1 x IPU-POD64

Command:
```console
poprun \
    --vv \
    --numa-aware=yes \
    --reset-partition=no \
    --update-partition=yes \
    --remove-partition=false \
    --vipu-server-host=$VIPU_CLI_API_HOST \
    --mpi-global-args=" \
        --tag-output \
        --allow-run-as-root \
        --output-filename output_poprun \
        --mca oob_tcp_if_include $TCP_IF_INCLUDE \
        --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
    --mpi-local-args=" \
        -x OPAL_PREFIX \
        -x LD_LIBRARY_PATH \
        -x PATH \
        -x PYTHONPATH \
        -x CPATH \
        -x IPUOF_VIPU_API_TIMEOUT \
        -x POPLAR_LOG_LEVEL \
        -x POPLAR_SDK_ENABLED \
        -x POPLAR_ENGINE_OPTIONS" \
    --vipu-server-timeout=3600 \
    --vipu-partition=$PARTITION \
    --executable-cache-path PYTORCH_EXE_DIR \
    --host $HOSTS \
    --ipus-per-replica=4 \
    --num-ilds=8 \
    --num-replicas=16 \
    --num-instances=4 \
python3 pretrain.py \
    --config b16_in1k_pretrain \
    --dataset generated \
    --byteio true \
    --iterations 256 \
    --gradient-accumulation 341 \
    --dataloader-workers 32 \
    --rebatched-worker-size 1024 \
    --optimizer-state-offchip false \
    --micro-batch-size 8 \
    --enable-rts false \
```

#### 1 x IPU-POD128

Command:
```console
poprun \
    --vv \
    --numa-aware=yes \
    --reset-partition=no \
    --update-partition=yes \
    --remove-partition=false \
    --vipu-server-host=$VIPU_CLI_API_HOST \
    --mpi-global-args=" \
        --tag-output \
        --allow-run-as-root \
        --output-filename output_poprun \
        --mca oob_tcp_if_include $TCP_IF_INCLUDE \
        --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
    --mpi-local-args=" \
        -x OPAL_PREFIX \
        -x LD_LIBRARY_PATH \
        -x PATH \
        -x PYTHONPATH \
        -x CPATH \
        -x IPUOF_VIPU_API_TIMEOUT \
        -x POPLAR_LOG_LEVEL \
        -x POPLAR_SDK_ENABLED \
        -x POPLAR_ENGINE_OPTIONS" \
    --vipu-server-timeout=3600 \
    --vipu-partition=$PARTITION \
    --executable-cache-path PYTORCH_EXE_DIR \
    --host $HOSTS \
    --ipus-per-replica=4 \
    --num-ilds=2 \
    --num-replicas=32 \
    --num-instances=8 \
python3 pretrain.py \
    --config b16_in1k_pretrain \
    --dataset generated \
    --byteio true \
    --iterations 512 \
    --gradient-accumulation 146 \
    --dataloader-workers 64 \
    --rebatched-worker-size 1024 \
    --optimizer-state-offchip false \
    --micro-batch-size 8 \
    --enable-rts false \
```

#### 1 x IPU-POD256

Command:
```console
poprun \
    --vv \
    --numa-aware=yes \
    --reset-partition=no \
    --update-partition=yes \
    --remove-partition=false \
    --vipu-server-host=$VIPU_CLI_API_HOST \
    --mpi-global-args=" \
        --tag-output \
        --allow-run-as-root \
        --output-filename output_poprun \
        --mca oob_tcp_if_include $TCP_IF_INCLUDE \
        --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
    --mpi-local-args=" \
        -x OPAL_PREFIX \
        -x LD_LIBRARY_PATH \
        -x PATH \
        -x PYTHONPATH \
        -x CPATH \
        -x IPUOF_VIPU_API_TIMEOUT \
        -x POPLAR_LOG_LEVEL \
        -x POPLAR_SDK_ENABLED \
        -x POPLAR_ENGINE_OPTIONS" \
    --vipu-server-timeout=3600 \
    --vipu-partition=$PARTITION \
    --executable-cache-path PYTORCH_EXE_DIR \
    --host $HOSTS \
    --ipus-per-replica=4 \
    --num-ilds=4 \
    --num-replicas=64 \
    --num-instances=16 \
python3 pretrain.py \
    --config b16_in1k_pretrain \
    --dataset generated \
    --byteio true \
    --iterations 1024 \
    --gradient-accumulation 73 \
    --dataloader-workers 64 \
    --rebatched-worker-size 512 \
    --optimizer-state-offchip false \
    --micro-batch-size 8 \
    --enable-rts false \
```

## Inference
