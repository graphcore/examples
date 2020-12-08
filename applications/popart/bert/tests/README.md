# BERT Testing


First, the environment need to be prepared according to the instructions in the root README.

Then, the pytests packing need to be invoked from the root bert folder.

Run all tests in parallel:
```
pytest -n 16 --forked
```

Run selected tests e.g. custom_ops unit tests only:
```
pytest tests/unit/custom_ops
```
