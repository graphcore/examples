# Model "Frozen️ in Time" implement in IPU
A Joint Video and Image Encoder for End-to-End Retrieval
----
[project page](https://www.robots.ox.ac.uk/~vgg/research/frozen-in-time/) | [paper](https://arxiv.org/abs/2104.00650) | [dataset](https://github.com/m-bain/webvid) |  [demo](http://meru.robots.ox.ac.uk/frozen-in-time/)
![alt text](arch.jpg)
Repository containing the code, models, data for end-to-end retrieval. WebVid data can be found [here](https://m-bain.github.io/webvid-dataset/)

----
## Installation instructions

1. Prepare the PopTorch environment. Install the Poplar SDK following the
   Getting Started(https://docs.graphcore.ai/en/latest/getting-started.html)guide for your IPU system. Make sure to source the
   `enable.sh` scripts for Poplar and PopART and activate a Python virtualenv with PopTorch installed.

2. Install the apt dependencies for Ubuntu 18.04 (requires admin privileges):

```console
sudo apt install $(< required_apt_packages.txt)
```

3. install the required python packages:
```console
pip install -r requirements.txt
```


## Prepare Datasets (evalution dataset: MSR-VTT, training dataset: WebVid-2M)

1. Download MSR-VTT `wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip -d data`

2. Download WebVid-2M (see https://github.com/m-bain/webvid), the downloading script 'download.py' included in this repo.
   a. Download 2.5M subset `wget http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv -P data/WebVid/release; wget http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_val.csv -P data/WebVid/release`
   b. Download videos
   ```console
   python3 datasets/download.py --csv_path data/WebVid/release/results_2M_train.csv --part 0
   python3 datasets/download.py --csv_path data/WebVid/release/results_2M_val.csv --part 0
   ```  
   c. Clean videos
   ```console
   mkdir data/WebVid/metadata
   python3 datasets/clean_videos.py --csv_path data/WebVid/release/results_2M_train.csv --video_path data/WebVid/videos/ --clean_csv_path data/WebVid/metadata/results_2M_training.csv
   python3 datasets/clean_videos.py --csv_path data/WebVid/release/results_2M_val.csv --video_path data/WebVid/videos/ --clean_csv_path data/WebVid/metadata/results_2M_inference.csv
   ``` 


## Run the application

After setting up your environment as explained above, you can run frozen-in-time(base_patch32_224) pretraining on WebVid-2M dataset.

Step1. Train Frozen-in-time on WebVid-2M dataset with 1-frame and 80 epochs:
```console
python3 run.py --config_name configs/webvid2m-8ipu-1f.json
```  
Step2. Set `"load_checkpoint":"From Step1 best ckeckpoint"`in the following config file. Then train frozen-in-time(base_patch32_224) on WebVid-2M dataset with 2-frame and 10 epochs:
```console
python3 run.py --config_name configs/webvid2m-8ipu-2f.json
```  
Step3. Set `"load_checkpoint":"From Step2 best ckeckpoint"`in the following config file. Then train frozen-in-time(base_patch32_224) on WebVid-2M dataset with 4-frame and 10 epochs:
```console
python3 run.py --config_name configs/webvid2m-8ipu-4f.json
``` 

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

## Licensing

This application is licensed under MIT license.
Please see the LICENSE file at the top-level of this repository.

The following files are created by Graphcore and are licensed under MIT License:
* README.md
* clean_videos.py
* modeling/model_patch.py
* test/*

The following files include code derived from the [Frozen-in-time repository](https://github.com/m-bain/frozen-in-time) which uses MIT license:
* arch.jpg
* run.py
* modeling/__init__.py
* modeling/loss.py
* modeling/metric.py
* modeling/model.py
* modeling/video_transformer.py
* modeling/trainer.py
* configs/*
* datasets/*


External packages:
- `torchvision` is licensed under BSD 3-Clause License
- `transformers` is licensed under Apache-2.0
