# Graphcore

---

## Click Through Rate

Click through rate (CTR) is a binary classification problem. The inputs of this algorithm include User, Item and Context, and the output of this algorithm is whether the item will be clicked or the probability that the item will be clicked (Depends on whether the sigmoid function is used).        

This directory contains sample applications and code examples for typical CTR algorithms.

## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).



## The structure of this directory


| File                         | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| `din/`                       | DIN model definition, train and infer scripts, tests and other used code    |
| `dien/`                      | DIEN model definition, train and infer scripts, tests and other used code   |
| `common/`                    | The modules that can be used in several models                              |
| `requirements.txt/`          | Required packages                                                           |
| `README.md`                  | Structure of the folder and projects description                            |
| `LICENSE`                    | The license that applies to the files in the click_through_rate directory   |
| `NOTICE`                     | Notice of CTR projects                                                      |
