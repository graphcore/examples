# CNN Benchmarking on IPUs

This README describes how to run TensorFlow CNN models for throughput benchmarking on the Mk2 IPU, for both training and inference.

## Preparation

Follow the installation instructions in applications/tensorflow/cnns/training/README.md.

Set the DATASETS_DIR environment variable to the parent directory of the ImageNet dataset.
ImageNet must be in TFRecord format.

Run the following commands from inside the applications/tensorflow/cnns/training/ directory.

The scripts provide a string of throughput values. Calculate the average of these. The first value reported will always be lower than the subsequent values, therefore remove that number from the calculation in order to get a fair indication of the throughput of a full training run. In the case of multithreaded runs (using poprun), remove the first result from each thread.

### TensorFlow CNNs Training

#### ResNet-50 v1.5 Training

1 x IPU-M2000

```
python train.py --model resnet --model-size 50 --dataset imagenet --data-dir $DATASETS_DIR/ --ckpts-per-epoch 0 --batch-size 16 --replicas 1 --gradient-accumulation-count 1024 --epochs 1 --no-validation --xla-recompute --optimiser momentum --momentum 0.9 --normalise-input --stable-norm --internal-exchange-optimisation-target balanced --enable-half-partials --standard-imagenet --pipeline-num-parallel 32 --pipeline-schedule Grouped --no-dataset-cache --shards=4 --pipeline-splits b1/2/relu b2/3/relu b3/3/relu --pipeline --batch-norm --available-memory-proportion 0.15 --disable-variable-offloading --precision 16.32 --eight-bit-io
```

1 x IPU-POD16

```
poprun --numa-aware=yes --num-replicas=4 --num-instances=4 --ipus-per-replica 4 python train.py --model resnet --model-size 50 --dataset imagenet --data-dir $DATASETS_DIR/ --ckpts-per-epoch 0 --batch-size 16 --gradient-accumulation-count 256 --epochs 1 --no-validation --xla-recompute --optimiser momentum --momentum 0.9 --normalise-input --stable-norm --internal-exchange-optimisation-target balanced --enable-half-partials --standard-imagenet --pipeline-num-parallel 32 --pipeline-schedule Grouped --no-dataset-cache --shards=4 --pipeline-splits b1/2/relu b2/3/relu b3/3/relu --pipeline --batch-norm --available-memory-proportion 0.15 --disable-variable-offloading --precision 16.32 --eight-bit-io
```

#### ResNext-101 Training

1 x IPU-M2000

```
python train.py --model resnext --model-size 101 --dataset imagenet --data-dir $DATASETS_DIR --shards 2 --replicas 2 --batch-size 6 --pipeline --gradient-accumulation-count 64 --pipeline-splits b3/3/relu --epoch 2 --ckpts-per-epoch 0 --xla-recompute --normalise-input --internal-exchange-optimisation-target balanced --enable-half-partials --stable-norm --base-learning-rate -11 --pipeline-schedule Grouped --optimiser momentum --momentum 0.9 --lr-schedule cosine --label-smoothing 0.1 --no-validation --disable-variable-offloading --eight-bit-io
```

1 x IPU-POD16

```
poprun --numa-aware 1 --num-replicas 8 --ipus-per-replica 2 --num-instances 8 python train.py --model resnext --model-size 101 --dataset imagenet --data-dir $DATASETS_DIR --shards 2 --batch-size 6 --pipeline --gradient-accumulation-count 16 --pipeline-splits b3/3/relu --epoch 2 --ckpts-per-epoch 0 --xla-recompute --normalise-input --internal-exchange-optimisation-target balanced --enable-half-partials --stable-norm --base-learning-rate -11 --pipeline-schedule Grouped --optimiser momentum --momentum 0.9 --lr-schedule cosine --label-smoothing 0.1 --no-validation --disable-variable-offloading --standard-imagenet --eight-bit-io
```

#### EfficientNet-B4 Training

##### Standard (Group Dim 1)

1 x IPU-M2000

```
python3 train.py --model=efficientnet --model-size=4 --data-dir $DATASETS_DIR/ --precision=16.32 --group-dim=16 --expand-ratio=4 --dataset=imagenet --groups=4 --optimiser=RMSprop --lr-schedule=exponential --xla-recompute --enable-conv-dithering --available-memory-proportion=0.15 --pipeline-schedule Grouped --internal-exchange-optimisation-target balanced --weight-avg-exp 0.97 --enable-half-partials --cutmix-lambda 0.85 --mixup-alpha=0.2 --disable-variable-offloading --batch-size=5 --shards=4 --pipeline-split block2c block4c block6a --pipeline --gradient-accumulation-count=40 --no-validation --epochs 2
```

1 x IPU-POD16

```
poprun --numa-aware 1 --num-replicas 4 --ipus-per-replica 4 --num-instances 4 python3 train.py --model=efficientnet --model-size=4 --data-dir $DATASETS_DIR/ --precision=16.32 --group-dim=16 --expand-ratio=4 --dataset=imagenet --groups=4 --optimiser=RMSprop --lr-schedule=exponential --xla-recompute --enable-conv-dithering --available-memory-proportion=0.15 --pipeline-schedule Grouped --internal-exchange-optimisation-target balanced --weight-avg-exp 0.97 --enable-half-partials --cutmix-lambda 0.85 --mixup-alpha=0.2 --disable-variable-offloading --batch-size=5 --shards=4 --pipeline-split block2c block4c block6a --pipeline --gradient-accumulation-count=40 --no-validation --epochs 1
```

##### Modified (Group Dim 16)

1 x IPU-M2000

```
python3 train.py --model=efficientnet --model-size=4 --data-dir $DATASETS_DIR/ --precision=16.32 --group-dim=16 --expand-ratio=4 --dataset=imagenet --groups=4 --optimiser=RMSprop --lr-schedule=exponential --xla-recompute --enable-conv-dithering --available-memory-proportion=0.15 --pipeline-schedule Grouped --internal-exchange-optimisation-target balanced --weight-avg-exp 0.97 --enable-half-partials --cutmix-lambda 0.85 --mixup-alpha=0.2 --disable-variable-offloading --batch-size=5 --shards=4 --pipeline-split block2c block4c block6a --pipeline --gradient-accumulation-count=40 --no-validation --epochs 2
```

1 x IPU-POD16 (using poprun)

```
poprun --numa-aware 1 --num-replicas 4 --ipus-per-replica 4 --num-instances 4 python3 train.py --model=efficientnet --model-size=4 --data-dir $DATASETS_DIR/ --precision=16.32 --group-dim=16 --expand-ratio=4 --dataset=imagenet --groups=4 --optimiser=RMSprop --lr-schedule=exponential --xla-recompute --enable-conv-dithering --available-memory-proportion=0.15 --pipeline-schedule Grouped --internal-exchange-optimisation-target balanced --weight-avg-exp 0.97 --enable-half-partials --cutmix-lambda 0.85 --mixup-alpha=0.2 --disable-variable-offloading --batch-size=5 --shards=4 --pipeline-split block2c block4c block6a --pipeline --gradient-accumulation-count=40 --no-validation --epochs 1
```

### TensorFlow CNNs Inference

Follow the installation instructions in applications/tensorflow/cnns/training/README.md.

Run the following command lines from inside the applications/tensorflow/cnns/training directory.

#### ResNet-50 v1.5 - synthetic data

```
python3 validation.py --model resnet --model-size 50 --dataset imagenet --batch-size 1 --synthetic-data --repeat 10 --batch-norm --enable-half-partials
```

Change the --batch-size argument to be one of 1, 4, 16, 32, 64, 90

#### ResNeXt-101 - synthetic data 

1 x IPU

```
python3 validation.py --model resnext --model-size 101 --dataset imagenet --batch-size 1 --synthetic-data --repeat 10 --batch-norm --enable-half-partials
```

Change the --batch-size argument to be one of 1, 2, 4, 8, 16

#### EfficientNet-B0 - Standard (Group Dim 1) - synthetic data

1 x IPU

```
python3 validation.py --model efficientnet --model-size 0 --dataset imagenet --precision 16.16 --batch-size 1 --synthetic-data --repeat 10 --batch-norm --enable-half-partials
```

Change the --batch-size argument to be one of 1, 8, 16, 32


#### ResNet-50 v1.5 - data generated on the host; 4 IPUs

1 x IPU-M2000

```
poprun --num-replicas 4 --ipus-per-replica 1 --num-instances 4 --numa-aware 1 python3 validation.py --model resnet --model-size 50 --dataset imagenet --batch-size 1 --generated-data --repeat 10 --batch-norm --enable-half-partials --eight-bit-io
```

Change the --batch-size argument to be one of 1, 4, 16, 32, 64, 80

#### ResNeXt-101 - data generated on the host; 4 IPUs

1 x IPU-M2000

```
poprun --num-replicas 4 --ipus-per-replica 1 --num-instances 4 --numa-aware 1 python3 validation.py --model resnext --model-size 101 --dataset imagenet --batch-size 2 --generated-data --repeat 10 --batch-norm --enable-half-partials --eight-bit-io
```

Change the --batch-size argument to be one of 1, 2, 4, 8, 16

#### EfficientNet-B0 - Standard (Group Dim 1) - data generated on the host; 4 IPUs

1 x IPU-M2000

```
poprun --num-replicas 4 --ipus-per-replica 1 --num-instances 4 --numa-aware 1 python3 validation.py --model efficientnet --model-size 0 --dataset imagenet --precision 16.16 --batch-size 8 --generated-data --repeat 10 --batch-norm --enable-half-partials --eight-bit-io
```

Change the --batch-size argument to be one of 1, 8, 16, 32


