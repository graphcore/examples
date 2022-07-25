# Sparse softmax

> # Copyright (c) 2020 Graphcore Ltd. All rights reserved.

In the canonical expression of softmax the exponentiation of activations would transform all implicit zeros (empty blocks) in a block-sparse matrix into ones. The sparse matrix would then become fully dense.

A common approach to overcome this issue is to re-interpet the zeros (empty entries) as having a prior probability of 0. In sparse matrices this is achieved by applying the softmax only across the non-zero values of a row. This preserves the original sparsity pattern.

## Execution
Run the following examples to confirm functionality
```
python sparse_softmax_subblock_demo.py
python sparse_grouped_softmax_subblock_demo.py
```

## Testing
```
pytest
```
Expect to see the following on stdout: 
```
test_sparse_softmax_subblock.py::Test::test_output PASSED
