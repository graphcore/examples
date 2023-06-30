# YOLOv4-P5
YOLOv4-P5 (object detection reference application), based on [this repository](https://github.com/WongKinYiu/ScaledYOLOv4), optimised for Graphcore's IPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch | Vision | YOLOv4-P5 | COCO 2017 | Object detection | <p style="text-align: center;">❌ | <p style="text-align: center;">✅ <br> Min. 4 IPUs (POD4) required | [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036) |


## Instructions summary
1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Download the ImageNet LSVRC 2012 dataset (See Dataset setup)


## Poplar SDK setup
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
```

3. Additionally, enable PopART with:
```bash
cd popart-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the PopTorch (PyTorch) wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch...x86_64.whl
```

4. Navigate to this example's root directory

5. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

5. Build the custom ops:
```bash
make
```


More detailed instructions on setting up your PyTorch environment are available in the [PyTorch quick start guide](https://docs.graphcore.ai/projects/pytorch-quick-start).

## Dataset setup

### COCO 2017
Download the COCO 2017 dataset from [the source](http://images.cocodataset.org/zips/) or [via kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset), or via the script we provide:
```bash
bash utils/download_coco_dataset.sh
```

Additionally, also download  and unzip the labels:
```bash
curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -o coco2017labels.zip && unzip -q coco2017labels.zip -d '<dataset path>' && rm coco2017labels.zip
```

Disk space required: 26G

```bash
.
├── LICENSE
├── README.txt
├── annotations
├── images
├── labels
├── test-dev2017.txt
├── train2017.cache
├── train2017.txt
├── val2017.cache
└── val2017.txt

3 directories, 7 files
```


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


## Custom inference

### Inference with pre-trained weights
To download the pretrained weights, run the following commands:
```bash
mkdir weights
cd weights
curl https://gc-demo-resources.s3.us-west-1.amazonaws.com/yolov4_p5_reference_weights.tar.gz -o yolov4_p5_reference_weights.tar.gz && tar -zxvf yolov4_p5_reference_weights.tar.gz && rm yolov4_p5_reference_weights.tar.gz
cd ..
```
These weights are derived from the a pre-trained model shared by the [YOLOv4's author](https://github.com/WongKinYiu/ScaledYOLOv4). We have post-processed these weights to remove the model description and leave a state_dict compatible with the IPU model description.

To run:
```bash
python3 run.py --weights weights/yolov4_p5_reference_weights/yolov4-p5-sd.pt
```

### Inference without pre-trained weights

```console
python run.py
```
`run.py` will use the default config defined in `configs/inference-yolov4p5.yaml` which can be overridden by various arguments (`python run.py --help` for more info)

### Evaluation

To compute evaluation metrics run:
```bash
python run.py --weights '/path/to/your/pretrain_weights.pt' --obj-threshold 0.001 --class-conf-threshold 0.001
```
You can use the `--verbose` flag if you want to print the metrics per class. Here is a comparison of our metrics against the GPU on the COCO 2017 detection validation set:

| Model | Image Size | Type | Classes | Precision | Recall | mAP@0.5 | mAP@0.5:.95 |
|-------|------------|------|---------|-----------|--------|---------|-------------|
|  GPU  | 896        | FP32 | all     | 0.4501    | 0.76607| 0.6864  | 0.49034     |
|  GPU  | 896        | FP16 | all     | 0.44997   | 0.7663 | 0.68663 | 0.49037     |
|  IPU  | 896        | FP16 | all     | 0.45032   | 0.7674 | 0.68674 | 0.49159     |

We generate the numbers for the GPU by re-running the Scaled-YOLOv4 repo code on an AWS instance. Please note that these numbers are slightly different from what they report in their repo. This is attributed to the `rect` parameter. In their inference, this is set to be `True`. The IPU currently can not support different sized images, and therefore, we set this to `False` in their evaluation in order to draw a fair comparison. In that regard, we do perform at par with SOTA.

</br></br>

# Demonstration of object detection using YOLOv4 in Jupyter notebook

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch   | vision | yolo  | COCO     | object detection |  <p style="text-align: center;"> ❌ | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required | POD4/POD16/POD64 | link to paper/original implementation|

## Object detection with YOLOv4 on Graphcore IPU:
The [notebook](notebook_yolo.ipynb) demonstrates the object detection task with YOLOv4 model executed on Graphcore IPU.
The assumption is that the Poplar SDK is downloaded and activated.
