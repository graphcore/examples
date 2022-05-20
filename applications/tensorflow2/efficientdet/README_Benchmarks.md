# Benchmarking on IPUs

This README describes how to run the EfficientDet model for throughput and inference benchmarking on the Mk2 GC200 and BOW IPU.

## Preparation

Follow the installation instructions in applications/tensorflow2/efficientdet/README.md.

## Inference

The following commans should be run in the directory applications/tensorflow2/efficientdet/

### EfficientDet D0 batch-sizes 1 to 3 on 4 IPUs using host generated data

This benchmark spawns 4 host processes, 1 host per replica. The throughput can be obtaining by summing the throughput per iteration across processes. 

Batch sizes: 1,2,3

Command:
```console
    mpirun \
      --tag-output \
      --allow-run-as-root \
      --np 4 \
      --bind-to socket \
    python ipu_inference.py \
      --model-name efficientdet-d0 \
      --micro-batch-size {batchsize} \
      --dataset-type generated 

```

### EfficientDet-D0-D4 max batch size for 4 IPUs using host generated data

This benchmark spawns 4 host processes, 1 host per replica. The throughput can be obtaining by summing the throughput per iteration across processes. 

modelname: d0,d1,d2,d3,d4

Command:
```console
    mpirun \
      --tag-output \
      --allow-run-as-root \
      --np 4 \
      --bind-to socket \
    python ipu_inference.py \
      --model-name efficientdet-{modelname} \
      --dataset-type generated 
```