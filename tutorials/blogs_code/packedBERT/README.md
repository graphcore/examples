<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# packedBERT algorithms for efficient sequence packing

This folder contains the histogram data for the Wikipedia and SQuAD dataset, as
well as the algorithms presented in ["Packing: Towards 2x NLP BERT
Acceleration"](https://arxiv.org/abs/2107.02027).

The training code for BERT with the packed dataset can be found in the [PopART
BERT
example](https://github.com/graphcore/examples/tree/v2.6.0/nlp/bert/popart). The
README in the
[bert_data](https://github.com/graphcore/examples/tree/v2.6.0/nlp/bert/popart/bert_data) folder
shows in step 5 how to prepare the packed dataset, and you can find
configurations which use packed training data in the
[configs/packed](https://github.com/graphcore/examples/tree/v2.6.0/nlp/bert/popart/configs/packed) folder.

## Contents

1. Sequence length histograms [histograms.py](./histograms.py)
2. Non-negative least squares histogram-packing [nnlshp.py](./nnlshp.py)
3. Shortest-pack-first histogram-packing [spfhp.py](./spfhp.py)
4. Longest-pack-first histogram-packing [lpfhp.py](./lpfhp.py)
5. Extended non-negative least squares histogram-packing [ennlshp.py](./ennlshp.py)

## Example use

All of the presented algorithms operate on histograms. The following snippets
demonstrate how each can be demoed on the Wikipedia pre-training dataset
histogram.

### Non-negative least-squares histogram-packing (NNLSHP)

```python3
from histograms import wikipedia_histogram, wikipedia_max_sequence_length
from nnlshp import pack_using_nnlshp
max_sequence_length = 512
max_sequences_per_pack = 3
strategy_set, strategy_repeat_count = pack_using_nnlshp(wikipedia_histogram, wikipedia_max_sequence_length, max_sequences_per_pack)
```

Which is expected to print:

```output
Packing efficiency (fraction of real tokens): 0.9975
 Speed-up theoretical limit: 2.0013
 Achieved speed-up over un-packed dataset: 1.99625
Runtime: Packed 16279552 sequences in 24.267 seconds.
```

### Shortest-pack-first histogram-packing (SPFHP)

```python3
from histograms import wikipedia_histogram, wikipedia_max_sequence_length
from spfhp import pack_using_spfhp
max_sequences_per_pack = 12
strategy_set, strategy_repeat_count = pack_using_spfhp(wikipedia_histogram, wikipedia_max_sequence_length, max_sequences_per_pack)
```

which is expected to print:

```output
Packing efficiency (fraction of real tokens): 99.6040
 Speed-up theoretical limit: 2.0013
 Achieved speed-up over un-packed dataset: 1.99340
 Runtime: Packed 16279552 sequences in 0.032 seconds.
```

### Longest-pack-first histogram-packing (LPFHP)

```python3
from histograms import wikipedia_histogram, wikipedia_max_sequence_length
from lpfhp import pack_using_lpfhp
max_sequences_per_pack = 12
strategy_set, strategy_repeat_count = pack_using_lpfhp(wikipedia_histogram, wikipedia_max_sequence_length, max_sequences_per_pack)
```

which is expected to print:

```output
Packing efficiency (fraction of real tokens): 99.8129
 Speed-up theoretical limit: 2.0013
 Achieved speed-up over un-packed dataset: 1.99758 Runtime: Packed 16279552 sequences in 0.048 seconds.
```

### Extended non-negative least squares histogram-packing (ENNLSHP)

```python3
from histograms import wikipedia_histogram, wikipedia_max_sequence_length
from ennlshp import pack_using_ennlshp
max_sequence_length = 512
max_sequences_per_pack = 3
strategy_set, strategy_repeat_count = pack_using_ennlshp(wikipedia_histogram, wikipedia_max_sequence_length, max_sequences_per_pack)
```

Which is expected to print:

```output
Packing efficiency (fraction of real tokens): 0.9975
 Speed-up theoretical limit: 2.0013
 Achieved speed-up over un-packed dataset: 1.99625
Runtime: Packed 16279552 sequences in 283.997 seconds.
```
