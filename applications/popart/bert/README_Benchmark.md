# BERT Training Benchmarking on IPUs using PopART

This README describes how to run PopART BERT models for throughput and inference benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/popart/bert/README.md, including the instructions on obtaining and pre-processing the dataset. Ensure the $DATASETS_DIR environment variable points to the location of the /wikipedia/ directory.

### PopART BERT Training

Run the following command lines from inside the applications/popart/bert directory:

#### BERT-Large Phase 1 Pre-training (Sequence length 128)

1 x IPU-POD16

```
python bert.py --config configs/mk2/pretrain_large_128.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

1 x IPU-POD64

```
python3 bert.py --config configs/mk2/pretrain_large_128.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1 --gradient-accumulation-factor 512 --replication-factor 16
```

#### BERT-Large Phase 2 Pre-training (Sequence length 384)

1 x IPU-POD16

```
python bert.py --config configs/mk2/pretrain_large_384.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

1 x IPU-POD64

```
python3 bert.py --config configs/mk2/pretrain_large_384.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --epochs 1 --epochs 10 --no-model-save --no-validation --steps-per-log 1 --gradient-accumulation-factor 512 --replication-factor 16
```

1 x IPU-POD64

```
python bert.py --config configs/mk2/pretrain_large_384.json --replication-factor 16 --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

#### BERT-Base Phase 1 Pre-training (Sequence length 128)

1 x IPU-POD16

```
python bert.py --config configs/mk2/pretrain_base_128.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

1 x IPU-POD64

```
python bert.py --config configs/mk2/pretrain_large_128.json --replication-factor 16 --loss-scaling 128 --available-memory-proportion 0.15 0.2 0.2 0.2 --input-files=$DATASETS_DIR/wikipedia/AA/sequence_128/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```

#### BERT-Base Phase 3 Pre-training (Sequence length 384)

1 x IPU-POD16

```
python bert.py --config configs/mk2/pretrain_base_384.json --input-files=$DATASETS_DIR/wikipedia/AA/sequence_384/wiki_00_tokenised --epochs 1 --no-model-save --no-validation --steps-per-log 1
```


### PopART BERT Inference

Follow the installation instructions in applications/popart/bert/README.md.

We use generated data to obtain the inference throughput, so you do not need to
use pre-trained weights or download the SQuAD dataset to reproduce the benchmark results.

Run the following command lines from inside the applications/popart/bert directory:

#### BERT-Large SQuAD v1.1 Inference (Sequence length 128)

1 x IPU

```
python3 bert.py --task SQUAD --layers-per-ipu=3 --num-layers=3 --hidden-size=768 --attention-heads=12 --sequence-length=128 --dtype=FLOAT16 --batches-per-step=2048 --generated-data=true --no-model-save --host-embedding=NONE --low-latency-inference=true --minimum-latency-inference=false --input-files=$DATASETS_DIR/squad/dev-v1.1.json --inference --encoder-start-ipu=0 --use-default-available-memory-proportion=true --max-copy-merge-size=-1 --shuffle=false --micro-batch-size 1 --enable-half-partials --epochs-inference 10 --group-host-sync --no-outlining=false --steps-per-log=1
```

Change the --micro-batch-size argument to be one of 1, 2, 3

#### BERT 3-layer Base Inference (Sequence length 128)

1 x IPU

```
python3 bert.py --task SQUAD --layers-per-ipu=3 --num-layers=3 --hidden-size=768 --attention-heads=12 --sequence-length=128 --dtype=FLOAT16 --batches-per-step=2048 --generated-data=true --no-model-save --host-embedding=NONE --low-latency-inference=false --minimum-latency-inference=true --input-files=$DATASETS_DIR/squad/dev-v1.1.json --inference --encoder-start-ipu=0 --use-default-available-memory-proportion=true --max-copy-merge-size=-1 --shuffle=false --micro-batch-size 1 --enable-half-partials --epochs-inference 10 --group-host-sync --no-outlining=false --steps-per-log=1
```

Change the --micro-batch-size argument to be one of 1, 2, 4, 8, 16, 32, 64


