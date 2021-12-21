# Benchmarking on IPUs

This README describes how to run the Unet medical model for throughput benchmarking on the Mk2 IPU.


## Preparation

Follow the installation instructions in applications/tensorflow2/unet/README.md.

## Training

Command:
```console
    python main.py \
      --nb-ipus-per-replica 4 \
      --micro-batch-size 1 \
      --gradient-accumulation-count 24 \
      --train \
      --augment \
      --learning-rate 0.0024 \
      --kfold 1 \
      --num-epochs 200 \
      --data-dir $DATASETS_DIR/tif \
      --benchmark
```

## Inference over 4 replicase (Pod4)

Command:
```console

    mpirun \
      --tag-output \
      --allow-run-as-root \
      --np 4 \
      --bind-to socket \
      -x POPLAR_RUNTIME_OPTIONS \
    python main.py \
      --nb-ipus-per-replica 1 \
      --micro-batch-size {batchsize} \
      --steps-per-execution 400 \
      --infer \
      --host-generated-data \
      --benchmark \
```

for batchsize 1,2
