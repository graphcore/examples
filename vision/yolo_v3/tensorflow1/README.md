## YOLOv3
YOLOv3 - You Only Look Once - is a convolutional neural network model that performs object detection tasks.\
The directory contains an example of YOLOv3 training and inference.\
This YOLOv3 model implementation is based on the original work from Yun Yang <dreameryangyun@sjtu.edu.cn>. This model is based on the original paper "YOLOv3: An Incremental Improvement": https://arxiv.org/pdf/1804.02767.pdf

The instructions to run the model are as follows:

## 1. Prepare working environment

1. Install requirements (Within poplarSDK TF1 environment)
```bashrc
$ pip install -r requirements.txt
$ export $DATASETS_DIR=/your/dataset/dir/
```

2. Go to your local version of the examples repo:
```bashrc
$ cd public_examples/vision/yolo_v3/tensorflow1/
```

3. (Optional, not required for benchmarking/testing) Download the checkpoint data:
```bashrc
$ mkdir ckpt_init
$ cd ckpt_init
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
```

## 2. Download and setup the VOC dataset

1. The VOC PASCAL training, validation and test datasets need to be downloaded:
```bashrc
$ mkdir $DATASETS_DIR/VOC
$ cd $DATASETS_DIR/VOC
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory, which should have the following structure:
```
VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
```
2. generate data reference files (voc_train.txt and voc_test.txt)
```
$ cd public_examples/vision/yolo_v3/tensorflow1/
$ mkdir ./data/dataset
$ python scripts/parse_voc_annotation.py --data_path $DATASETS_DIR/VOC
```

(`tf1_yolov3_training.yml` includes args that tell the model to look into `./data/dataset/voc_train.txt` etc. to find the data in your $DATASETS_DIR)

For the purposes of benchmarking/testing YoloV3 on VOC data, no further setup steps are required.

3. (Optional) Train from COCO weights, this script will also run evaluation code:
```bashrc
$ bash run_load_coco_weights_544.sh
```

4. (Optional) To see the evaluation result:
```bashrc
$ cat mAP/mAP.log
```

On the VOC2007 dataset we have verified the model accuracy at 85.75%

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
