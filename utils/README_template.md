# Example name/acronym (full expanded name if applicable)
One line summary of example, as well as any images.

(Table that summarises app usage and capability)
| Framework | domain | Model | Datasets | Tasks| Training| Inference | Reference |
|-------------|-|------|-------|-------|-------|---| --------|
| one of them | NLP | name | list| Q&A / Classification, etc.. | ✅  | ✅ | link to paper/original implementation|

## Instructions summary: 
Before you can run this model for training or inference you will need to:

1. Install and enable the Poplar SDK (see Poplar SDK setup)
2. Install the system and python requirements (see Environment setup)
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


More detailed instructions on setting up your environment are available in the [poplar quick start guide](https://docs.graphcore.ai/projects/graphcloud-poplar-quick-start/en/latest/).

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
For the CPU architecture you are running on (For tensorflow only)

4. Navigate to this example's root directory

5. Pre-requirements build stepsL
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

## Dataset setup

### Dataset name
One or two lines on what this dataset is and what it is used for
disk space required: XXGB
```bash
bash prep_data.sh etc.
```
File struture tree for dataset in tree form, using "tree -L 3 --filelimit 5"
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
File struture tree for dataset in tree form, using "tree -L 3 --filelimit 5"
dataset_dir
├── README.md
├── train
|   ├── subdir_1
|   └── subdir_2
├── validate
    ├── subdir_1
    └── subdir_2


## Running and benchmark

Here are the following benchmarks that can be reproduced using this example:
| Model	| Num IPUs | Task |
|--------------|-------|----------|
| EfficientNet | M2000 | Training |
| EfficientNet | POD16 | Training |
| EfficientNet | POD64 | Training |
For a full list of all benchmarks available, please see our [performance results page](https://www.graphcore.ai/performance-results)

Specifying both default way to run the app and also how to benchmark due to benchmarks available on our site being tested setups/configs pre-release that are more likely to work well for all users.some explanation on examples_utils benchmarking and yaml files. Link to perf results page.
```bash
cd to root dir of a benchmarks.yml file
python3 -m examples_utils benchmark --benchmark <benchmark name> etc. or simpler commands
```

## Custom training/inference and other features
Statement on this section being about some basic info on how to run your own custom training/validation/inference commands and how to use other nice features

### Training
Some explanation here
```bash
some commands perhaps or args to add
```

### Validation
Some explanation here
```bash
some commands perhaps or args to add
```

### Inference
Some explanation here
```bash
some commands perhaps or args to add
```

### Additional feature
Some explanation here
```bash
some commands perhaps or args to add
```


## License
License for this examples (standard across examples likely)
