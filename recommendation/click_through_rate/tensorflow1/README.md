# Graphcore

---

## Click Through Rate

Click through rate (CTR) is a binary classification problem. The inputs of this algorithm include User, Item and Context, and the output of this algorithm is whether the item will be clicked or the probability that the item will be clicked (Depends on whether the sigmoid function is used).        

This directory contains sample applications and code examples for typical CTR algorithms.

## Benchmarking

To reproduce the benchmarks, please follow the setup instructions in this README to setup the environment, and then from this dir, use the `examples_utils` module to run one or more benchmarks. For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml
```

or to run a specific benchmark in the `benchmarks.yml` file provided:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --benchmark <benchmark_name>
```

For more information on how to use the examples_utils benchmark functionality, please see the <a>benchmarking readme<a href=<https://github.com/graphcore/examples-utils/tree/master/examples_utils/benchmarks>

## Profiling

Profiling can be done easily via the `examples_utils` module, simply by adding the `--profile` argument when using the `benchmark` submodule (see the <strong>Benchmarking</strong> section above for further details on use). For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --profile
```
Will create folders containing popvision profiles in this applications root directory (where the benchmark has to be run from), each folder ending with "_profile". 

The `--profile` argument works by allowing the `examples_utils` module to update the `POPLAR_ENGINE_OPTIONS` environment variable in the environment the benchmark is being run in, by setting:
```
POPLAR_ENGINE_OPTIONS = {
    "autoReport.all": "true",
    "autoReport.directory": <current_working_directory>,
    "autoReport.outputSerializedGraph": "false",
}
```
Which can also be done manually by exporting this variable in the benchmarking environment, if custom options are needed for this variable.

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
