# Integration tests for RNN-Transducer and Sparse LogSoftMax implementation

Follow the instructions in `../torch_reference/README` to build the reference transducer library.

Make sure to build the PopART custom RNNTLoss and SparseLogSoftmax ops by running the appropriate `make` program.

Then run the tests by doing:

```
pytest -v sparse_rnnt_loss_test.py
```
