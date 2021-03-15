# BERT Training Benchmarking on IPUs using TensorFlow

This README describes how to run TensorFlow BERT models for throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/tensorflow/bert/README.md.

Follow the instructions at the same location for obtaining and processing the dataset. Ensure the $DATASETS_DIR environment variable points to the location of the tf_wikipedia directory created. The benchmark scripts run on a subset of the full Wikipedia dataset. 

Run the following commands from inside the applications/tensorflow/bert/ directory.

In order to get a clear indication of the average throughput of a full training run, calculate the average of the results from these scripts, ignoring the first 3 results.

## Benchmarks

### Pretrain BERT-Base Sequence Length 128

1 x IPU-POD16

```
python run_pretraining.py --config configs/pretrain_base_128.json","$DATASETS_DIR/tf_wikipedia/tokenised_128_dup5/00.tfrecord --train-file $DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask58/wiki_00_cleaned.tfrecord --num-train-steps 4
```

### Pretrain BERT-Base Sequence Length 384

1 x IPU-POD16

```
python run_pretraining.py --config configs/pretrain_base_384.json","$DATASETS_DIR/tf_wikipedia/tokenised_128_dup5/00.tfrecord --train-file $DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask58/wiki_00_cleaned.tfrecord --num-train-steps 4
```

### Pretrain BERT-Large Sequence Length 128

1 x IPU-POD16

```
python run_pretraining.py --config configs/pretrain_large_128_phase1.json --train-file $DATASETS_DIR/tf_wikipedia/tokenised_128_dup5/00.tfrecord --num-train-steps 4
```

### Pretrain BERT-Large Sequence Length 384

1 x IPU-POD16

```
python run_pretraining.py --config configs/pretrain_large_384_phase2.json --train-file DATASETS_DIR/tf_wikipedia/tokenised_384_dup5_mask58/wiki_00_cleaned.tfrecord --num-train-steps 4
```
