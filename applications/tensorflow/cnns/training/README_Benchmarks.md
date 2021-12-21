# Benchmarking on IPUs

This README describes how to run TensorFlow CNN models for throughput benchmarking on the Mk2 IPU, for both training and inference.

## Preparation

Follow the installation instructions in applications/tensorflow/cnns/training/README.md.

Set the DATASETS_DIR environment variable to the parent directory of the ImageNet dataset.
ImageNet must be in TFRecord format.

Run the following commands from inside the applications/tensorflow/cnns/training/ directory.

The scripts provide a string of throughput values. Calculate the average of these. The first value reported will always be lower than the subsequent values, therefore remove that number from the calculation in order to get a fair indication of the throughput of a full training run. In the case of multithreaded runs (using poprun), remove the first result from each thread.

## Training

To use these command lines for training instead of benchmarking you need to remove the `--epochs <n>` option.
You should remove the `--ckpts-per-epoch 0` option (where present) if you want to save checkpoints for future use (such as validation runs).
One checkpoint will then be stored per epoch.
If the command line does not use `poprun` then you can remove the `--no-validation` option if you want to run validation after training.
This will run validation on all available checkpoints.
If the training command line uses `poprun` then you will have to use the `validation.py` script to run validation separately.
See the [README](./README.md#popdist-and-poprun---distributed-training-on-ipu-pods) for details.

### ResNet-50 v1.5 Training

#### 1 x IPU-M2000

Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
```

Command:
```console
poprun \
   --mpi-global-args="--allow-run-as-root --tag-output" \
   --mpi-local-args="-x TF_POPLAR_FLAGS" \
   --numa-aware 1 \
   --ipus-per-replica 1 \
   --num-instances 2 \
   --num-replicas 4 \
python3 \
   train.py \
   --config mk2_resnet50_mlperf_pod16_bs20 \
   --epochs-per-sync 20 \
   --data-dir $DATASETS_DIR/ \
   --no-validation \
   --epochs 1 \
   --epochs-per-ckpt 0 \
   --gradient-accumulation-count 24
```

#### 1 x IPU-POD16

Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
```

Command:
```console
poprun \
   --mpi-global-args="--allow-run-as-root --tag-output" \
   --mpi-local-args="-x TF_POPLAR_FLAGS" \
   --numa-aware 1 \
   --ipus-per-replica 1 \
   --num-instances 8 \
   --num-replicas 16 \
python3 \
   train.py \
   --config mk2_resnet50_mlperf_pod16_bs20 \
   --epochs-per-sync 20 \
   --data-dir $DATASETS_DIR/ \
   --no-validation \
   --epochs 2 \
   --epochs-per-ckpt 0
```

#### 1 x IPU-POD64

for all Pod64 set:

$PARTITION, $IPUOF_VIPU_API_PARTITION_ID: ID of the Pod64 reconfigurable partition
$TCP_IF_INCLUDE: sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address.
$VIPU_SERVER_HOST: IP address as appropriate for the target hardware 
$HOSTS: IP address of the main host server

Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
export POPLAR_ENGINE_OPTIONS={"opt.enableMultiAccessCopies":"false"}
export POPLAR_TARGET_OPTIONS={"gatewayMode":"false"}
```

Command:
```console
      poprun \
         --host $HOSTS \
         --mpi-global-args=" \
            --allow-run-as-root \
            --tag-output \
            --mca oob_tcp_if_include $TCP_IF_INCLUDE \
            --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
         --mpi-local-args=" \
            -x OPAL_PREFIX \
            -x LD_LIBRARY_PATH \
            -x PATH \
            -x PYTHONPATH \
            -x IPUOF_VIPU_API_TIMEOUT=600 \
            -x POPLAR_LOG_LEVEL=WARN \
            -x TF_POPLAR_FLAGS \
            -x DATASETS_DIR \
            -x POPLAR_ENGINE_OPTIONS \
            -x POPLAR_TARGET_OPTIONS" \
         --update-partition=no \
         --reset-partition=no \
         --vipu-server-timeout 300 \
         --vipu-server-host "$VIPU_SERVER_HOST" \
         --vipu-partition=$PARTITION \
         --numa-aware 1 \
         --ipus-per-replica 1 \
         --num-instances 16 \
         --num-replicas 64 \
      python3 \
         train.py \
         --config mk2_resnet50_mlperf_pod64_bs20 \
         --epochs-per-sync 20 \
         --data-dir $DATASETS_DIR/ \
         --logs-path . \
         --no-validation
```

### ResNext-101 Training

#### 1 x IPU-M2000

Command:
```console
python train.py \
   --config mk2_resnext101_16ipus \
   --replicas 2 \
   --ckpts-per-epoch 0 \
   --epochs 2 \
   --logs-per-epoch 16 \
   --no-validation \
   --data-dir $DATASETS_DIR \
   --gradient-accumulation-count 64
```

#### 1 x IPU-POD16

Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
```

Command:
```console
poprun \
   --mpi-global-args=" \
      --allow-run-as-root \
      --tag-output" \
   --mpi-local-args="-x TF_POPLAR_FLAGS" \
   --numa-aware 1 \
   --num-replicas 8 \
   --ipus-per-replica 2 \
   --num-instances 8 \
python train.py \
   --config mk2_resnext101_16ipus \
   --ckpts-per-epoch 0 \
   --epochs 2 \
   --logs-per-epoch 16 \
   --no-validation \
   --data-dir $DATASETS_DIR
```



### EfficientNet-B4 Training Modified Group Dim 16

#### 1 x IPU-M2000


Command:
```console
poprun \
  --mpi-global-args="--allow-run-as-root --tag-output" \
  --numa-aware 1 \
  --num-replicas 2 \
  --num-instances 2 \
  --ipus-per-replica 2 \
python3 train.py \
  --config mk2_efficientnet_b4_g16_16ipus \
  --identical-replica-seeding \
  --epochs 1 \
  --data-dir $DATASETS_DIR/ \
  --logs-per-epoch 16 \
  --no-validation \
  --replicas 1 \
  --gradient-accumulation-count 160 \
  --dataset-percentage-to-use 15 \
  --eight-bit-io \
  --num-io-tiles 32 \
  --prefetch-depth 3 \
```

#### 1 x IPU-POD16

Command:
```console
poprun \
  --mpi-global-args="--allow-run-as-root --tag-output" \
  --numa-aware 1 \
  --num-replicas 8 \
  --num-instances 8 \
  --ipus-per-replica 2 \
python3 train.py \
  --config mk2_efficientnet_b4_g16_16ipus \
  --identical-replica-seeding \
  --epochs 1 \
  --data-dir $DATASETS_DIR/ \
  --logs-per-epoch 16 \
  --no-validation \
  --dataset-percentage-to-use 15 \
  --eight-bit-io \
  --num-io-tiles 32 \
  --prefetch-depth 3
```

#### 1 x IPU-POD64

xxx.xxx.xxx.xxx: Replace with IP addresses as appropriate for the target hardware '--mca btl_tcp_if_include xxx.xxx.xxx.0/xx' sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address. Replace 'partition_name' with the name of your partition. Ensure the dataset, application and SDK are synced across hosts.

Env:
```console
export TF_POPLAR_FLAGS=--executable_cache_path=/tmp/tf_cache/
export POPLAR_ENGINE_OPTIONS={"opt.enableMultiAccessCopies":"false"}
```

Command:
```console
poprun 
  --vv \
  --host xxx.xxx.xx1,xxx.xxx.xx2,xxx.xxx.xx3,xxx.xxx.xx4  \
  --mpi-global-args="--allow-run-as-root --tag-output --mca oob_tcp_if_include xxx.xxx.xxx.0/xx --mca btl_tcp_if_include xxx.xxx.xxx.0/xx"  \
  --mpi-local-args="-x OPAL_PREFIX  \
    -x LD_LIBRARY_PATH  \
    -x PATH  \
    -x PYTHONPATH  \
    -x IPUOF_VIPU_API_TIMEOUT=600  \
    -x POPLAR_LOG_LEVEL=WARN  \
    -x TF_POPLAR_FLAGS  \
    -x DATASETS_DIR  \
    -x POPLAR_ENGINE_OPTIONS  \
    -x POPLAR_TARGET_OPTIONS"  \
    --update-partition=no  \
    --reset-partition=no  \
    --vipu-server-timeout 300  \
    --vipu-server-host xxx.xxx.xx1  \
    --vipu-partition=$IPUOF_VIPU_API_PARTITION_ID  \
    --numa-aware 1  \
    --ipus-per-replica 2  \
    --num-instances 32  \
    --num-replicas 32  \
  python3 train.py \
    --config mk2_efficientnet_b4_g16_64ipus \
    --identical-replica-seeding \
    --epochs 350 \
    --logs-per-epoch 16 \
    --no-validation \
    --data-dir $DATASETS_DIR \
    --wandb \
    --eight-bit-io \
    --identical-replica-seeding \
    --seed 1 \
    --num-io-tiles 32 \
    --prefetch-depth 3
```

## Inference

Follow the installation instructions in applications/tensorflow/cnns/training/README.md.

Run the following command lines from inside the applications/tensorflow/cnns/training directory.


### ResNet-50 v1.5 - generated data

#### 1 x IPU-M2000

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
    mpirun \
      --tag-output \
      --allow-run-as-root \
      --np 4 \
      --bind-to socket \
    python3 inference_embedded.py \
      --model resnet \
      --config mk2_resnet50_base \
      --dataset imagenet \
      --generated-data \
      --iterations 8000 \
      --batches-per-step 100 \
      --micro-batch-size {batchsize} \
      --eight-bit-io  \
      --batch-norm \
      --enable-half-partials \
      --fused-preprocessing \
      --no-dataset-cache \
      --num-inference-thread 3 \
      --tmp-execs
```

Set micro_batch_size to 1,2,4,8,16,32, or 64. 

### ResNeXt-101 - generated data

#### 1 x IPU-M2000 

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
  mpirun \
      --tag-output \
      --allow-run-as-root \
      --np 4 \
      --bind-to socket \
    python3 inference_embedded.py \
      --model resnext \
      --model-size 101 \
      --dataset imagenet \
      --micro-batch-size {batchsize} \
      --generated-data \
      --batch-norm \
      --eight-bit-io \
      --enable-half-partials \
      --fused-preprocessing \
      --iterations 8000 \
      --batches-per-step 100 \
      --no-dataset-cache \
      --num-inference-thread 3 \
      --tmp-execs
```

Set micro_batch_size to 1, 2, 4, 8 or 16.  

### EfficientNet-B0 - Standard Group Dim 1 - generated data

#### 1 x IPU-M2000 

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
    mpirun \
      --tag-output \
      --allow-run-as-root \
      --np 4 \
      --bind-to socket \
    python3 inference_embedded.py \
      --model efficientnet \
      --model-size 0 \
      --dataset imagenet \
      --generated-data \
      --batch-norm \
      --iterations 8000 \
      --batches-per-step 100 \
      --micro-batch-size {batchsize} \
      --eight-bit-io \
      --enable-half-partials \
      --fused-preprocessing \
      --no-dataset-cache \
      --num-inference-thread 3 \
      --tmp-execs
```


Set --micro-batch-size to 1, 8, 16, 32, 40, 42. 

### EfficientNet-B4 - Standard Group Dim 1 - generated data

#### 1 x IPU-M2000 

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
    mpirun \
      --tag-output \
      --allow-run-as-root \
      --np 4 \
      --bind-to socket \
    python3 inference_embedded.py \
      --model efficientnet \
      --model-size 4 \
      --dataset imagenet \
      --generated-data \
      --batch-norm \
      --iterations 8000 \
      --batches-per-step 100 \
      --micro-batch-size {batchsize} \
      --eight-bit-io \
      --enable-half-partials \
      --fused-preprocessing \
      --no-dataset-cache \
      --num-inference-thread 3 \
      --tmp-execs
```

set --micro-batch-size to 1, 2, 4, or 5
