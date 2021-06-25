# Benchmarking on IPUs

This README describes how to run the Deep Voice 3 throughput benchmark on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/popart/deep_voice/README.md

We use generated data to obtain the training throughput, so you do not
need to download the VCTK dataset to reproduce the benchmark result.

## Training

### Deep Voice 3

Run the following command line from inside the applications/popart/deep_voice directory.

#### 1 x IPU-M2000

Command:
```console
python3 deep_voice_train.py --data_dir TEST --model_dir TEST --generated_data --num_ipus 4 --replication_factor 4 --batch_size 128 --num_epochs 2 --no_validation
```
