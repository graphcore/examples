PyTorch Object Detector reference application on IPUs
---

## Overview

Run Object Detector inference on Graphcore IPUs using PyTorch.

The following models are supported in inference:
1. YOLOv4-P5, implementation of [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036). [Original repository](https://github.com/WongKinYiu/ScaledYOLOv4).

### Folder structure

* `run.py` Reference script to run the application.
* `models` Models definition.
* `utils` Contains common code such as tools and dataset functionalities.
* `configs` Contains the default configuration for inference.
* `tests` Contains all the different tests for the model.
* `README.md` This file.
* `requirements.txt` Required Python packages.
* `conftest.py` Test helper functions.

### Installation instructions

1. Prepare the PopTorch environment. Install the Poplar SDK following the
   [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for your IPU system. Make sure to source the
   `enable.sh` scripts for Poplar and PopART and activate a Python virtualenv with PopTorch installed.

2. Install the pip dependencies:

```console
pip install -r requirements.txt
```

3. Download labels and sample images for inference and training:

Download the labels:

```console
curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -o coco2017labels.zip && unzip -q coco2017labels.zip -d '/localdata/datasets' && rm coco2017labels.zip
```

This command might need root access or sudo to unzip the folders.

Download the images:

```console
bash utils/download_coco_dataset.sh
```

4. Build the custom ops:

```console
make
```

## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

### Running inference

1. Running inference without the weight:

```console
python run.py
```
`run.py` will use the default config defined in `configs/inference-yolov4p5.yaml` which can be overridden by the following options:
```
  -h, --help            show this help message and exit
  --data DATA           Path to the dataset root dir (default:
                        /localdata/datasets/)
  --config CONFIG       Configuration of the model (default:
                        configs/inference-yolov4p5.yaml)
  --show-config         Show configuration for the program (default: False)
  --weights WEIGHTS     Pretrained weight path to use if specified (default:
                        None)
  --num-ipus NUM_IPUS   Number of IPUs to use (default: 1)
  --num-workers NUM_WORKERS
                        Number of workers to use (default: 20)
  --input-channels INPUT_CHANNELS
                        Number of channels in the input image (default: 3)
  --activation ACTIVATION
                        Activation function to use in the model (default:
                        mish)
  --normalization NORMALIZATION
                        Normalization function to use in the model (default:
                        batch)
  --num-classes NUM_CLASSES
                        Number of classes of the model (default: 80)
  --epochs EPOCHS       Number of training epochs (default: 300)
  --class-name-path CLASS_NAME_PATH
                        Path to the class names yaml (default:
                        ./configs/class_name.yaml)
  --image-size IMAGE_SIZE
                        Size of the input image (default: 896)
  --micro-batch-size MICRO_BATCH_SIZE
                        The number of samples calculated in one full
                        forward/backward pass (default: 1)
  --mode MODE           Mode to run the model (default: test)
  --precision {half,mixed,single}
                        Precision to run the model with (default: half)
  --benchmark           Run performance benchmark (default: False)
  --sharded             Use sharded execution, where IPU will sequentially
                        execute a part of the model, otherwise pipeline
                        execution will be used instead (default: False)
  --exec-cache EXEC_CACHE
                        Path to store training executable cache for the
                        compiled model (default: ./exec_cache)
  --cpu                 Use cpu to run model (default: True)
  --device-iterations DEVICE_ITERATIONS
                        Number of device iterations (default: 1)
  --gradient-accumulation GRADIENT_ACCUMULATION
                        Number of gradient accumulation, only relevant for
                        training (default: 1)
  --initial-lr INITIAL_LR
                        Initial learning rate (default: 0.01)
  --momentum-lr MOMENTUM_LR
                        Initial learning rate (default: 0.937)
  --class-conf-threshold CLASS_CONF_THRESHOLD
                        Minimum threshold for class prediction probability
                        (default: 0.4)
  --obj-threshold OBJ_THRESHOLD
                        Minimum threshold for the objectness score (default:
                        0.4)
  --iou-threshold IOU_THRESHOLD
                        Minimum threshold for IoU used in NMS (default: 0.65)
  --plot-step PLOT_STEP
                        Plot every n image (default: 250)
  --plot-dir PLOT_DIR   Directory for storing the plot output (default: plots)
  --dataset-name DATASET_NAME
                        Name of the dataset (default: coco)
  --max-bbox-per-scale MAX_BBOX_PER_SCALE
                        Maximum number of bounding boxes per image (default:
                        90)
  --train-file TRAIN_FILE
                        Path to the train annotations (default: train2017.txt)
  --test-file TEST_FILE
                        Path to the test annotations (default: val2017.txt)
  --no-eval             Dont compute the precision recall metrics (default:
                        True)
  --verbose             Print out class wise eval (default: False)
  --wandb               Log metrics to weight and biases (default: False)
  --logging-interval LOGGING_INTERVAL
                        Number of steps to log training progress on console
                        and/or to wandb (default: 200)
```
2. Running pre-trained model:

To download the pretrained weights, run the following commands:
```
mkdir weights
cd weights
curl https://gc-demo-resources.s3.us-west-1.amazonaws.com/yolov4_p5_reference_weights.tar.gz -o yolov4_p5_reference_weights.tar.gz && tar -zxvf yolov4_p5_reference_weights.tar.gz && rm yolov4_p5_reference_weights.tar.gz
cd ..
```
These weights are derived from the a pre-trained model shared by the [YOLOv4's author](https://github.com/WongKinYiu/ScaledYOLOv4).
We have post-processed these weights to remove the model description and leave a state_dict compatible with the IPU model description.

To run inference with the weights:

```console
python run.py --weights weights/yolov4_p5_reference_weights/yolov4-p5-sd.pt
```

### Evaluation

To compute evaluation metrics run:
``` console
python run.py --weights '/path/to/your/pretrain_weights.pt' --obj-threshold 0.001 --class-conf-threshold 0.001
```
You can use the `--verbose` flag if you want to print the metrics per class. Here is a comparison of our metrics against theirs on the COCO 2017 detection validation set:

| Model | Image Size | Type | Classes | Precision | Recall | mAP@0.5 | mAP@0.5:.95 |
|-------|------------|------|---------|-----------|--------|---------|-------------|
|  GPU  | 896        | FP32 | all     | 0.4501    | 0.76607| 0.6864  | 0.49034     |
|  GPU  | 896        | FP16 | all     | 0.44997   | 0.7663 | 0.68663 | 0.49037     |
|  IPU  | 896        | FP16 | all     | 0.45032   | 0.7674 | 0.68674 | 0.49159     |

We generate the numbers for the GPU by re-running the Scaled-YOLOv4 repo code on an AWS instance. Please note that these numbers are slightly different from what they report in their repo. This is attributed to the `rect` parameter. In their inference, this is set to be `True`. The IPU currently can not support different sized images, and therefore, we set this to `False` in their evaluation in order to draw a fair comparison. In that regard, we do perform at par with SOTA.


### Running the tests

After following installation instructions run:

```console
pytest
```

### Running training

To run the training you need to run ´run.py´in train mode.
It is recommended to use the provided config file.

``` console
python run.py --config /configs/training-yolov4p5.yaml --epochs 300
```
The training config can be overwritten with command-line interface.

The training will create multiple checkpoints. It is the files with `.pt` extension in the weights folder. To improve the portability of checkpoints `.pt.conf.yml` assigned to the file, which contains the configs of the experiment.

Any checkpoint can be validated with the following command:
``` console
python run.py --weights <path/checkpoint.pt> --epochs 0
```
It loads the pretrained model and skips the training.
