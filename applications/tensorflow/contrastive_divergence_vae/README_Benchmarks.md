# Benchmarking on IPUs

This README describes how to run the TensorFlow contrastive divergence variational autoencoder for benchmarking on the Mk2 IPU in training.

## Preparation

Follow the installation instructions in applications/tensorflow/contrastive_divergence_vae/README.md.

Run the following command from inside the applications/tensorflow/contrastive_divergence_vae/ directory.

## Training

### 1 x IPU

Command:
```console
python3 main.py --config-file=configs/benchmark_config.json --no-testing --batch-size 3072
```
