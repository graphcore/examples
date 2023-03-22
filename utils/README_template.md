# Example name/acronym (full expanded name if applicable)
One line summary of example, as well as any images.

(Table that summarises app usage and capability)
| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| one of them | NLP | name | list| Q&A / Classification, etc.. | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required  | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required | POD4/POD16/POD64 | link to paper/original implementation|

## (Optional) Extended introduction:
~3-10 lines about the example, method/tech used, applications, tasks etc. To act as an extended introduction to the example.

## Instructions summary:
Before you can run this model for training or inference you will need to:

1. Install and enable the Poplar SDK (see Poplar SDK setup)
2. Install the system and Python requirements (see Environment setup)
3. Download the XX dataset (See Dataset setup)
4. etc.

### Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh (for Pytroch apps only)
```


More detailed instructions on setting up your poplar environment are available in the [poplar quick start guide] (https://docs.graphcore.ai/projects/poplar-quick-start)

## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the <Framework> and <Additional for framework> add-ons wheels:
```bash
cd <poplar sdk root dir>
pip3 install ...
pip3 install ...
```
For the CPU architecture you are running on (For TensorFlow 1/2 only)

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
A description of what these steps do, and what the purpose of these are.
```bash
sudo apt-get install xxx
make clean && make
```

More detailed instructions on setting up your <framework> environment are available in the [<framework> quick start guide] (link from below)
PyTorch - https://docs.graphcore.ai/projects/pytorch-quick-start
TensorFlow 2 - https://docs.graphcore.ai/projects/tensorflow2-quick-start

## Dataset setup

### Dataset name
One or two lines on what this dataset is and what it is used for
disk space required: XXGB
```bash
bash prep_data.sh etc.
```
File structure tree for dataset in tree form, using "tree -L 3 --filelimit 5"
dataset_dir
├── README.md
├── train
|   ├── subdir_1
|   └── subdir_2
├── validate
    ├── subdir_1
    └── subdir_2

### Another dataset name
One or two lines on what this dataset is and what it is used for
disk space required: XXGB
```bash
bash prep_data.sh etc.
```
File structure tree for dataset in tree form, using "tree -L 3 --filelimit 5"
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

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).


## Custom training
Statements in this section being about some basic info on how to run your own custom training with commands

## Custom Inference
Statements in this section being about some basic info on how to run your own custom inference with commands

## Other features
Any explanation/demonstration about other things that dont fit in sections above or how to use the app otherwise


## License
License for this examples (standard across examples likely)
