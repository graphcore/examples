# Benchmarking on IPUs

This README describes how to run PyTorch BERT models for throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/pytorch/bert/README.md.

Follow the instructions at the same location for obtaining and processing the dataset. Ensure the $DATASETS_DIR environment variable points to the location of the dataset.

Run the following commands from inside the applications/pytorch/bert/ directory.

## Training

### Pretrain BERT-Base Sequence Length 128

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config pretrain_base_128 --training-steps 10 --input-file $DATASETS_DIR/wikipedia/128/wiki_1[0-1]*.tfrecord --disable-progress-bar
```

### Pretrain BERT-Base Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config pretrain_base_384 --training-steps 10 --input-file $DATASETS_DIR/wikipedia/384/wiki_1[0-1]*.tfrecord --disable-progress-bar
```

### Pretrain BERT-Large Sequence Length 128

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config pretrain_large_128 --training-steps 10 --input-file $DATASETS_DIR/wikipedia/128/wiki_1[0-1]*.tfrecord --disable-progress-bar
```

#### 1 x IPU-POD64

Command:
```console
python run_pretraining.py --config pretrain_large_128_POD64 --input-file $DATASETS_DIR/wikipedia/128/*.tfrecord --disable-progress-bar --wandb
```

### Pretrain BERT-Large Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python run_pretraining.py --config pretrain_large_384 --training-steps 10 --input-file $DATASETS_DIR/wikipedia/384/wiki_1[0-1]*.tfrecord --disable-progress-bar
```

### SQuAD BERT-Large Sequence Length 384

#### 1 x IPU-POD16

Command:
```console
python run_squad.py --config squad_large_384_POD16 --squad-do-validation False
```
