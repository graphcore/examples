# Example name/acronym (full expanded name if applicable)
Give a one-line summary of the example and include images, if relevant.

(Table that summarises app usage and capability)
| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| one of them | NLP | name | list| Q&A / Classification, etc.. | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required  | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required | POD4/POD16/POD64 | link to paper/original implementation|

## (Optional) Extended introduction
Give a short description (~3-10 lines) about the example, the methods or technology used, applications, tasks etc. This should fill the role of an extended introduction to the example.

## Overview
Before you can run this model for training or inference you will need to:

1. Install and enable the Poplar SDK (see [Poplar SDK setup](#poplar-sdk-setup))
2. Install the system and Python requirements (see [Environment setup](#environment-setup))
3. Download the XX dataset (See [Dataset setup](#dataset-setup))
4. etc.

### Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is printed, then follow these steps to enable the Poplar SDK:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable
```

Detailed instructions on setting up your Poplar environment are available in the [<framework> quick start guide] (link from below)
PyTorch - https://docs.graphcore.ai/projects/pytorch-quick-start
TensorFlow 2 - https://docs.graphcore.ai/projects/tensorflow2-quick-start
PopXL - https://docs.graphcore.ai/projects/poplar-quick-start


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the framework-specific wheels:
```bash
cd <poplar sdk root dir>
pip3 install ...
pip3 install ...
```
Detailed instructions on setting up your <framework> environment are available in the [<framework> quick start guide](link from below)
PyTorch - https://docs.graphcore.ai/projects/pytorch-quick-start
TensorFlow 2 - https://docs.graphcore.ai/projects/tensorflow2-quick-start


4. Navigate to this example's root directory

5. Pre-requirements build steps
```bash
make
```

5. Install the Python requirements with:
```bash
pip3 install -r requirements.txt
```

6. Additional build steps:
```bash
make
```

### Additional build steps
Describe what these steps are, and what their purpose is.
```bash
sudo apt-get install xxx
make clean && make
```

## Dataset setup

### Dataset name
Describe (1-2 lines) what this dataset is and what it is used for.

Disk space required: XXGB
```bash
bash prep_data.sh etc.
```
Include the file structure tree for dataset in tree form, using "tree -L 3 --filelimit 5"
dataset_dir
├── README.md
├── train
|   ├── subdir_1
|   └── subdir_2
├── validate
    ├── subdir_1
    └── subdir_2

### Another dataset name
Describe (1-2 lines) what this dataset is and what it is used for.

Disk space required: XXGB
```bash
bash prep_data.sh etc.
```
Include the file structure tree for dataset in tree form, using "tree -L 3 --filelimit 5"
dataset_dir
├── README.md
├── train
|   ├── subdir_1
|   └── subdir_2
├── validate
    ├── subdir_1
    └── subdir_2


## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. The benchmarks are provided in the `benchmarks.yml` file in this example's root directory.

For example:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the `examples-utils` benchmarking module, please refer to the [module README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).


## Custom training
Give basic info on how to run your own custom training with commands

## Custom Inference
Give basic info on how to run your own custom inference with commands

## Other features
Describe or demonstrate any other aspects of the app that you want to highlight that don't fit into the sections above.

## License
License for this examples (standard across examples likely)
