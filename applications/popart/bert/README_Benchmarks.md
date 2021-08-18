# Benchmarking on IPUs

This README describes how to run PopART BERT models for throughput and inference benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/popart/bert/README.md, including the instructions on obtaining and pre-processing the dataset. Ensure the $DATASETS_DIR environment variable points to the location of the /wikipedia/ directory.

## Training

Run the following command lines from inside the applications/popart/bert directory:

### BERT-Large Phase 1 Pre-training Sequence length 128

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/pretrain_large_128.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

#### 1 x IPU-POD64

Command:
```console
python bert.py --config configs/mk2/pretrain_large_128_POD64.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_*_tokenised --replication 16 --wandb
```

### BERT-Large Phase 2 Pre-training Sequence length 384

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/pretrain_large_384.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

#### 1 x IPU-POD64

Command:
```console
python3 bert.py --config configs/mk2/pretrain_large_384.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --replication 16 --loss-scaling 4096 --epochs 2 --wandb
```

### BERT-Base Phase 1 Pre-training Sequence length 128

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/pretrain_base_128.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```


### BERT-Base Phase 2 Pre-training Sequence length 384

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/pretrain_base_384.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

### BERT Large SQuAD Sequence length 384

#### 1 x IPU-POD16

Command:
```console
python bert.py --config configs/mk2/squad_large_384.json --input-files=$DATASETS_DIR/squad/train-v1.1.json --vocab-file=$DATASETS_DIR/ckpts/uncased_L-24_H-1024_A-16/vocab.txt --no-model-save --no-validation --steps-per-log 1
```

## Inference

Follow the installation instructions in applications/popart/bert/README.md.

We use generated data to obtain the inference throughput, so you do not need to
use pre-trained weights or download the SQuAD dataset to reproduce the benchmark results.

Run the following command lines from inside the applications/popart/bert directory:

### BERT-Large SQuAD v1.1 Inference Sequence length 128

#### 1 x IPU-M2000

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
mpirun --tag-output --allow-run-as-root --np 4 python bert.py --task=SQUAD --layers-per-ipu 24 --num-layers=24 --hidden-size=1024 --attention-heads=16 --sequence-length=128 --dtype=FLOAT16 --batches-per-step=16 --generated-data=true --no-model-save --host-embedding=NONE --minimum-latency-inference=true --input-files=$DATASETS_DIR/squad/dev-v1.1.json --inference --encoder-start-ipu=0 --use-default-available-memory-proportion=true --max-copy-merge-size=-1 --shuffle=false --micro-batch-size 1 --enable-half-partials --epochs-inference 20 --group-host-sync --no-outlining=false --steps-per-log=1
```

Set --micro-batch-size to 1, 2 or 3.

### BERT-Base SQuAD v1.1 Inference Sequence length 128

#### 1 x IPU-M2000

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
mpirun --tag-output --allow-run-as-root --np 4 python bert.py --task=SQUAD --layers-per-ipu 12 --num-layers=12 --hidden-size=768 --attention-heads=12 --sequence-length=128 --dtype=FLOAT16 --batches-per-step=16 --generated-data=true --no-model-save --host-embedding=NONE --minimum-latency-inference=true --input-files=$DATASETS_DIR/squad/dev-v1.1.json --inference --encoder-start-ipu=0 --use-default-available-memory-proportion=true --max-copy-merge-size=-1 --shuffle=false --micro-batch-size 1 --enable-half-partials --epochs-inference 10 --group-host-sync --no-outlining=false --steps-per-log=1
```

Set --micro-batch-size to 1, 2, 4, 8, 16, 32, 64 or 80.

### BERT 3-layer Base Inference Sequence length 128

#### 1 x IPU-M2000

This benchmark spawns multiple replicas using mpirun. To obtain the total throughput, sum the reported throughputs for each iteration.

Command:
```console
mpirun --tag-output --allow-run-as-root --np 4 python3 bert.py --task SQUAD --layers-per-ipu=3 --num-layers=3 --hidden-size=768 --attention-heads=12 --sequence-length=128 --dtype=FLOAT16 --batches-per-step=2048 --generated-data=true --no-model-save --host-embedding=NONE --low-latency-inference=false --minimum-latency-inference=true --input-files=$DATASETS_DIR/squad/dev-v1.1.json --inference --encoder-start-ipu=0 --use-default-available-memory-proportion=true --max-copy-merge-size=-1 --shuffle=false --micro-batch-size 1 --enable-half-partials --epochs-inference 10 --group-host-sync --no-outlining=false --steps-per-log=1
```

Set --micro-batch-size to 1, 2, 4, 8, 16, 32 or 64. Set --low-latency-inference to false or true. Set --minimum-latency-inference to true or false.
