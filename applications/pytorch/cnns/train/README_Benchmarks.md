# CNN Benchmarking on IPUs

This README describes how to run PyTorch CNN models for training throughput benchmarking on the Mk2 IPU.

### PyTorch CNNs Training

Follow the installation instructions in applications/pytorch/cnns.
Follow the instructions in applications/pytorch/cnns/datasets/README.md to create a directory containing the ImageNet dataset in the WebDataset format.
Set the DATASETS_DIR environment variable to the parent directory of the dataset.

Run the following command lines from inside the applications/pytorch/cnns/training directory.

The first value reported will always be lower than the subsequent values. When calculating an average throughput, remove the first reported result from the calculation in order to get a fair indication of the throughput of a full training run.

#### ResNet-50 v1.5 Training

1 x IPU-M2000

```
python train.py --model resnet50 --data imagenet --imagenet-data-path $DATASETS_DIR/imagenet-webdata-format --replicas 4 --batch-size 6 --gradient-accumulation 44 --epoch 2 --optimizer sgd --momentum 0.85 --half-partial --validation-mode none --disable-metrics --precision 16.16 --device-iterations 1 --norm-type group --norm-num-groups 32  --enable-stochastic-rounding 
```

1 x IPU-POD16

```
python train.py --model resnet50 --data imagenet --imagenet-data-path $DATASETS_DIR/imagenet-webdata-format --replicas 16 --batch-size 6 --gradient-accumulation 11 --epoch 2 --optimizer sgd --momentum 0.85 --half-partial --validation-mode none --disable-metrics --precision 16.16 --device-iterations 1 --norm-type group --norm-num-groups 32 --enable-stochastic-rounding 
```

#### EfficientNet-B4  Modified (Group Dim 16) Training

1 x IPU-M2000

```
python train.py --model efficientnet-b4 --data imagenet --imagenet-data-path $DATASETS_DIR/imagenet-webdata-format --replicas 1 --batch-size 4 --gradient-accumulation 256 --epoch 2 --optimizer rmsprop --half-partial --validation-mode none --disable-metrics --precision 16.32 --device-iterations 1 --norm-type group --norm-num-groups 4 --efficientnet-group-dim 16 --efficientnet-expand-ratio 4 --pipeline-splits _blocks/3 _blocks/9 _blocks/22 --enable-stochastic-rounding
```

1 x IPU-POD16

```
python train.py --model efficientnet-b4 --data imagenet --imagenet-data-path $DATASETS_DIR/imagenet-webdata-format --replicas 4 --batch-size 4 --gradient-accumulation 64 --epoch 2 --optimizer rmsprop --half-partial --validation-mode none --disable-metrics --precision 16.32 --device-iterations 1 --norm-type group --norm-num-groups 4 --efficientnet-group-dim 16 --efficientnet-expand-ratio 4 --pipeline-splits _blocks/3 _blocks/9 _blocks/22 --enable-stochastic-rounding
```


