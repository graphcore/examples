# Benchmarking on IPUs

This README describes how to run the Conformer model for throughput benchmarking on the Mk2 IPU.


## Preparation

Follow the installation instructions in applications/pytorch/conformer/README.md.

## Training

Command:
```console
        python3 \
            main.py \
            train \
            --trainer.log_every_n_step 10 \
            --train_dataset.use_generated_data true \
            --trainer.num_epochs=3 \
            --ipu_options.num-replicas={replicas} \
```

for replicas 2,8
