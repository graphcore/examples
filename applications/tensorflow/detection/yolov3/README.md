## YOLOv3
YOLOv3 - You Only Look Once - is a convolutional neural network model that performs object detection tasks.\
The directory contains an example of YOLOv3 training and inference.\
This YOLOv3 model implementation is based on the original work from Yun Yang <dreameryangyun@sjtu.edu.cn>. This model is based on the original paper "YOLOv3: An Incremental Improvement": https://arxiv.org/pdf/1804.02767.pdf

The instructions to run the model are as follows:

## 1. Prepare working environment

1. Prepare the TensorFlow environment. Install the Poplar SDK following the instructions in the Getting Started guide
   for your IPU system. Make sure to run the `enable.sh` script for Poplar and activate a Python 3 virtualenv with
   the TensorFlow 1 wheel from the Poplar SDK installed.
   Then install other python packages using the following command:
```bashrc
$ pip install -r requirements.txt
```
2. In your local version of the examples repo, do
```bashrc
$ cd examples/applications/tensorflow/detection/yolov3
```
3. Download the required data:
```bashrc
$ mkdir ckpt_init
$ cd ckpt_init
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
```

## 2. Train the model on the VOC dataset
Two dataset files are required:

- [`dataset.txt`](./data/dataset/voc_train.txt)(this file will be generated in section 2.1):
- [`class.names`](./data/classes/voc.names)


The VOC PASCAL training, validation and test datasets need to be downloaded:
```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory and rename them, which should have the following structure:

VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)

generate ./data/classes/voc.names
```
$ mkdir ./data/dataset
$ python scripts/parse_voc_annotation.py --data_path /to/your/VOC
```

Train from COCO weights, this script will also run evaluation code:
```bashrc
$ bash run_load_coco_weights_544.sh
```

To see the evaluation result:
```bashrc
$ cat mAP/mAP.log
```

On the VOC2007 dataset we have verified the model accuracy at 85.75%


## License information
This application is licensed under Apache License 2.0.
Please see the LICENSE file in this directory for full details of the license conditions.

The following files are licensed under MIT license and are derived from the work of YunYang1994 <dreameryangyun@sjtu.edu.cn>:\
./train.py \
./evaluate.py \
./core \
./mAP \
./scripts/show_bboxes.py \
./scripts/darknet53_to_yolov3.py \
./scripts/coco_weights_to_voc.py \
./scripts/parse_coco_annotation.py \
./scripts/parse_voc_annotation.py
./tests/original_model

opencv-python, pytest, tqdm are licensed under MIT license.
easydict is licensed under LGPL license.

The following files are licensed under Apache License 2.0. They are derived from NVIDIA Deep Learning Examples github and modified by Graphcore Ltd.: \
./ipu_optimizer.py (derived from NVIDIA Deep Learning Examples github)
./scripts/fp32_to_fp16.py (derived from NVIDIA Deep Learning Examples github)

The following files are created by Graphcore Ltd.  and are licensed under Apache 2.0:
./README.md \
./config/ \
./scripts/detection_to_coco_format.py \
./scripts/ckpt_to_pb.py \
./ipu_optimizer.py \
./ipu_utils.py \
./log.py \
./run_load_coco_weights_320.sh \
./run_load_coco_weights_544.sh \
./tests/optimizer_fixture.py
./tests/test_graph.py
./tests/test_optimizer.py
./tests/test_upsample.py
