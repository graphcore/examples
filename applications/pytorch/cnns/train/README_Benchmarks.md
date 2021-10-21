# Benchmarking on IPUs

This README describes how to run PyTorch CNN models for training throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/pytorch/cnns.
Follow the instructions in applications/pytorch/cnns/datasets/README.md to create a directory containing the ImageNet dataset in the WebDataset format.
Set the DATASETS_DIR environment variable to the parent directory of the dataset.

Run the following command lines from inside the applications/pytorch/cnns/training directory.

The first value reported will always be lower than the subsequent values. When calculating an average throughput, remove the first reported result from the calculation in order to get a fair indication of the throughput of a full training run.

## Training

### ResNet-50 v1.5


#### 1 x IPU-M2000

set:
	POPLAR_ENGINE_OPTIONS: '{"opt.enableMultiAccessCopies":"false"}'
   POPLAR_TARGET_OPTIONS: '{"gatewayMode":"false"}'

Command:
```console
	      poprun \
         --mpi-global-args="--allow-run-as-root --tag-output" \
         --numa-aware=1 \
         --ipus-per-replica=1 \
         --num-instances=4 \
         --num-replicas=4 \
      python train.py \
         --config resnet50_mk2 \
         --imagenet-data-path $DATASETS_DIR/imagenet_optimised \
         --gradient-accumulation 228 \
         --epoch 2 \
         --validation-mode none \
         --disable-metrics \
         --dataloader-worker 24 \
         --webdataset-memory-cache-ratio 0.2 \
         --dataloader-rebatch-size 256
```


#### 1 x IPU-POD16

set:
	POPLAR_ENGINE_OPTIONS: '{"opt.enableMultiAccessCopies":"false"}'
   POPLAR_TARGET_OPTIONS: '{"gatewayMode":"false"}'

Command:
```console
	      poprun \
         --mpi-global-args="--allow-run-as-root --tag-output" \
         --numa-aware=1 \
         --ipus-per-replica=1 \
         --num-instances=8 \
         --num-replicas=16 \
      python train.py \
         --config resnet50_mk2 \
         --imagenet-data-path $DATASETS_DIR/imagenet_optimised \
         --epoch 2 \
         --validation-mode none \
         --disable-metrics \
         --dataloader-worker 24 \
         --webdataset-memory-cache-ratio 0.4 \
         --dataloader-rebatch-size 256
```

#### 1 x IPU-POD64

set: POPLAR_ENGINE_OPTIONS: '{"opt.enableMultiAccessCopies":"false"}'

Command:
```console
	      poprun \
         --vv \
         --num-instances=16 \
         --num-replicas=64 \
         --ipus-per-replica=1 \
         --vipu-server-host=$VIPU_SERVER_HOST \
         --host=$HOSTS \
         --vipu-server-port 8090 \
         --num-ilds=1 \
         --vipu-partition=$PARTITION \
         --numa-aware=yes \
         --update-partition=no \
         --remove-partition=no \
         --reset-partition=no \
         --print-topology=yes \
         --executable-cache-path cache/rn50_pod64 \
         --mpi-global-args="--tag-output \
            --allow-run-as-root \
            --mca btl_tcp_if_include eno1" \
         --mpi-local-args="-x LD_LIBRARY_PATH \
            -x OPAL_PREFIX \
            -x PATH \
            -x CPATH \
            -x PYTHONPATH \
            -x POPLAR_ENGINE_OPTIONS \
            -x IPUOF_VIPU_API_TIMEOUT=800" \
      python3 train.py --config resnet50_mk2_pod64 \
         --dataloader-worker 24 \
         --webdataset-memory-cache-ratio 0.95 \
         --dataloader-rebatch-size 256 \
         --imagenet-data-path $DATASETS_DIR/imagenet_optimisede
```

### ### EfficientNet-B4 - Group Dim 16 

#### 1 x IPU-M2000

Command:
```console	    
    python train.py \
      --config efficientnet-b4-g16-gn-16ipu-mk2 \
      --replicas 1 \
      --gradient-accumulation 128 \
      --imagenet-data-path $DATASETS_DIR/imagenet_optimised \
      --epoch 2 \
      --warmup-epoch 0 \
      --validation-mode none \
      --weight-avg-strategy none \
      --disable-metrics
```


#### 1 x IPU-POD16

Command:
```console
	    python train.py \
      --config efficientnet-b4-g16-gn-16ipu-mk2 \
      --imagenet-data-path $DATASETS_DIR/imagenet_optimised \
      --epoch 2 \
      --warmup-epoch 0 \
      --validation-mode none \
      --weight-avg-strategy none \
      --disable-metrics
```