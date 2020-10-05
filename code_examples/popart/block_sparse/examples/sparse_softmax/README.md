# Sparse softmax

> Copyright 2019 Graphcore Ltd.

In the canonical expression of softmax the exponentiation of activations would transform all implicit zeros (empty blocks) in a block-sparse matrix into ones. The sparse matrix would then become fully dense.

A common approach to overcome this issue is to re-interpet the zeros (empty entries) as having a prior probability of 0. In sparse matrices this is achieved by applying the softmax only across the non-zero values of a row. This preserves the original sparsity pattern.

## Testing
Run the following example to confirm functionality
```
python test_block_sparse_softmax.py 
```
Expect to see the following on stdout: 
```
TEST: Verifying forward pass result
TEST FWD PASSED.
TEST: Verifying backward pass result
TEST BWD PASSED.
```
