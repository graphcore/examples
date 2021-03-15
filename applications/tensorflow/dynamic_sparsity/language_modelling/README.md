**Copyright (c) 2020 Graphcore Ltd. All rights reserved.**

# Autoregressive Language Modelling
The model is a stack of transformer encoder layers with causal i.e. autoregressive masking in the self-attention layers (at times also loosely referred to as decoder layers).
It is sparsely trained on Wikitext-103 using the Rig-L optimiser from `Rigging the Lottery: Making All Tickets Winners` ([arxiv](https://arxiv.org/abs/1911.11134))

Important files in this folder:
- **train_sparse.py** The main training program.
- **program_options.py** Training program configuration options. 
- **tf_utils.py** the in-graph learning rate schedule

## Training a Sparse Transformer Based Language Model
### Setup the environment
1) Setup the environment as per the top level dynamic sparsity instructions: `applications/tensorflow/dynamic_sparsity/README.md`

2) Build the ipu_sparse_ops module:
```bash
make -j
PYTHONPATH=./ python ipu_sparse_ops/tools/sparse_matmul.py
```

### Prepare the Dataset (Wikitext-103)

The example uses wikitext-103 but any dataset that can be prepared with GPT2's custom BPE tokeniser could conceivably be used instead. In order to prepare
wikitext-103 data follow the instructions in `applications/tensorflow/dynamic_sparsity/dataset_encoder/README.md`. The example command below uses
`--sequence-length=256` so make sure you set that option when you run `dataset_encoder/encode_dataset.py`. 

You will need to set the `--data-dir` option in the steps below to the path specified during the generation of the encoded data. (Will be of the form: `<dataset-dir>/<dataset-name>-gpt2/`.)

### Train the Model:

For a quick start, try the following:
```bash
python train_sparse.py --num-shards=2 --encoder-layers=1 --source-sequence-length 256 --hidden-length=768 --ff-length=3072 --source-vocab-length=30000 --repeat-count=100 --warmup-steps=1000 --cooldown-steps=10000 --peak-learning-rate=2e-4 --min-learning-rate=8e-6 --nepochs=100 --prune-ratio 0.3 --block-size 16 --pooling-type=AVG --pipeline --gradient-accumulation-count 60 --data-dir <path-to-your-encoder-results>
```
Note that the model uses pipelining to allow scaling up to a large number of Transformer encoders.

### Larger Models
If you want to train a larger (e.g. GPT2 sized) sparse model then you will need to set numerous parameters that tune memory use on the IPU and enable recompute for the longer pipelines. For example:
```bash
python3 train_sparse.py --num-shards=16 --encoder-layers=45 --hidden-length=1600 --ff-length=6400 --source-vocab-length=50000 --warmup-steps=2000 --cooldown-steps=10000 --peak-learning-rate=0.00025 --decay-power 0.8 --min-learning-rate=8e-6 --nepochs=100 --pipeline --repeat-count 100 --data-dir <path-to-encoded-data> --source-sequence-length 256 --gradient-accumulation-count 376 --dtype=float16 --block-size=16 --sparse-matmul-options='{"metaInfoBucketOversizeProportion":0.2,"partialsType":"half","availableMemoryProportion":0.4}' --encoder-stage-dense-matmul-options='{"availableMemoryProportion":"0.15"}' --sparsity=0.9 --pooling-type=AVG --prune-ratio 0.5  --grad-acculation-mode Avg --scale-grad-pre-acc --recompute --extra-poplar-options-sync-enable
```

## Training and loss function
The loss is weighted sparse softmax **cross-entropy with logits**. At training time we report the unweighted mean cross-entropy loss (each token has equal weight in the loss) as **xentropy**. The reported **perplexity** is based on this unweighted cross-entropy loss.

Additionally, we report a **training loss** which is a weighted cross-entropy. The weights are designed to down-weight the re-occurrence of tokens in the sequence. For instance if a token occurs once it's weight in the training cross-entropy loss will be 1, if it occurs twice, each occurrence will have a weight of 1/2. Using such weights can help stabilize training.

## Optimizer
The default optimizer is Adam with a linear warm-up schedule (to `--peak-learning-rate`) followed by a square-root decay in learning rate to a specified minimum (`--min-learning-rate`).The schedule is implemented in **tf_utils.py**.

The optimizer implements gradient clipping to stabilize convergence.

### Sparse Optimiser

Because the model is sparse there are various optimiser wrappers that deal with nuances of the sparse training algorithm (e.g. returning the optimiser slots to the host efficiently
as they need to be modified at the same time as the network connectivity).

## Running experiments
It is possible to run some experiments and track it with weights and biases. Simply login to your wandb account with `wandb login`. Then on your wandb dashboard create a new project. Finally run with the following arguments: `--use-wandb --wandb-project-name <your-project-name>`. See the help for other wandb options.