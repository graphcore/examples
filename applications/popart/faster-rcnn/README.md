## Faster-RCNN
This is a IPU implementation of Faster-RCNN detection framework based on ruotianluo's [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn). This model is based on the original paper "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks": https://arxiv.org/abs/1506.01497

Currently, we get 79.7(mixed-precision) versus 80.3(Detectron2, FP32) on VOC, Faster-RCNN C4, Resnet50 training.

NOTE, different SDK versions will influence the mAP slightly.

The instructions to run the model are as follows:

## 1. Prepare working environment

1. Install requirements
```
pip install -r requirements.txt
make
```

2. Go to your local version of the examples repo:
```
cd public_examples/applications/popart/detection/faster-rcnn
```

## 2. Download and setup the dataset

You can run `bash prepare_dataset.sh` to install the dataset with one click. Of course, if you are interested in how the dataset is deployed, you can see the step-by-step tutorial below.

NOTE: The data is placed in the `./data` by default. If you don’t like to put the data and code together, you can modify the var $DATASETS_DIR in the script, and you also need to modify `_C.DATA_DIR` in `config.py`.

1. The VOC PASCAL training, validation and test datasets need to be downloaded:
```
DATASETS_DIR=${your_data_path}
mkdir $DATASETS_DIR
cd $DATASETS_DIR
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory, which should have the following structure:

DATASETS_DIR
```
DATASETS_DIR/
    VOCdevkit/
        VOC2007/ (from VOCtrainval_06-Nov-2007.tar)
        VOC2012/ (from VOCtrainval_11-May-2012.tar)
```

2. generate data reference files (voc_train.txt and voc_test.txt)
```
python3 split_trainval_voc.py ${DATASETS_DIR}/VOCdevkit/VOC2007
python3 split_trainval_voc.py ${DATASETS_DIR}/VOCdevkit/VOC2012
```

3. merge VOC2007 and VOC2012
We will train on VOC 2007 train+val + VOC 2012 train+val, testing on VOC 2007. This is a popular training method, called 07+12 in many papers. Another 07++12 (testing on 2012 test) requires submitting test data to the official VOC server. Those who are interested can try it by themselves. The mAP of 07+12 should be aournd 79.7 for Faster-RCNN-R50-C4 and Detectron2 is 80.3. 

To train on VOC 2007 train+val + VOC 2012 train+val, we need to put them together.
```
# merge VOC2007 and VOC2012 annotations
mkdir ${DATASETS_DIR}/VOC_annotrainval_2007_2012
cp ${DATASETS_DIR}/VOCdevkit/VOC2007/Annotations_trainval/* ${DATASETS_DIR}/VOC_annotrainval_2007_2012/
cp ${DATASETS_DIR}/VOCdevkit/VOC2012/Annotations_trainval/* ${DATASETS_DIR}/VOC_annotrainval_2007_2012/

# merge VOC2007 and VOC2012 images
mkdir ${DATASETS_DIR}/VOC_images
cp ${DATASETS_DIR}/VOCdevkit/VOC2007/JPEGImages/* ${DATASETS_DIR}/VOC_images/
cp ${DATASETS_DIR}/VOCdevkit/VOC2012/JPEGImages/* ${DATASETS_DIR}/VOC_images/
```

## 2. Train and evaluation

1. get pretrained weights for backbone ResNet-50 and convert to IPU format
The pretrained weights of backbone can be downloaded from here(https://drive.google.com/open?id=0B7fNdx_jAqhtbllXbWxMVEdZclE).
Then make a folder named weights and put in it.

```
# The script will convert weights from weights/resnet50-caffe.pth to weights/GC_init_weights.pth 
python3 get_init_weights.py 
```
The weights are from Detectron2(https://github.com/facebookresearch/detectron2), they are licensed under the Apache 2.0

2. train and evaluate model
For multi-batch, 4 IPUs, mixed-presision training on VOC 2007 train+val + VOC 2012 train+val and testing on VOC 2007, you will get mAP around 79.7 on VOC 2007 test. 
This is a popular training method, called 07+12 in many papers. Another 07++12 (testing on 2012 test) requires submitting test data to the official VOC server. Those who are interested can try it by themselves. The mAP of 07+12 should be aournd 79.7 for Faster-RCNN-R50-C4 and Detectron2 is 80.3. 
```
bash train.sh yamls/mixed_precision_VOC0712_16batch.yaml
```
Also you can try other configs contained in `yamls/`

## License information
This application is licensed under Apache License 2.0.
Please see the LICENSE file in this directory for full details of the license conditions.

The following files are licensed under MIT license and are derived from the work of Microsoft: 
./layer/anchor_target_layer.py 
./datasets/blob.py 
./datasets/imdb.py 
./datasets/factory.py 

The files contained in following folder are licensed under MIT license and are derived from the work of ONNX Project: 
./IPU/custom_ops/include/onnx

The files contained in following folder are licensed under google's license and are derived from the work of Google Inc.: 
./IPU/custom_ops/include/google

The following files are licensed under MIT license and are derived from the work of Ross Girshick: 
./datasets/ds_utils.py 

The following files are licensed under MIT license and are derived from the work of Bharath Hariharan: 
./datasets/voc_eval.py

The following files are licensed under Apache license 2.0 and are derived from the work of TensorFlow and modified by Graphcore Ltd.: 
./layer/balanced_positive_negative_sampler.py 
./layer/tpu_proposal_target_layer.py 

The files contained in following folder are licensed under Apache license 2.0 and are derived from the work of RangiLyu: 
./nanodata/

The following files are licensed under Apache license 2.0 and are derived from the work of RangiLyu and modified by Graphcore Ltd.: 
./nanodata/dataset/xml_dataset_for_rcnn.py 
./nanodata/dataset/coco_dataset_for_rcnn.py

The following files are created by Graphcore Ltd. and are licensed under Apache 2.0:
./README.md 
./evaluation.py 
./evaluation.sh 
./get_init_weights.py 
./keys_mappin.txt 
./Makefile 
./multi_train.sh 
./requirements.txt 
./config.py 
./split_trainval_voc.py 
./train.py 
./train.sh 
./yaml_parser.py 
./yamls/ 
./utils/ 
./models/ 
./layer/resnet.py 
./layer/rpn.py 
./layer/vgg.py 
./IPU/ 
./datasets/data_loader.py 
./tests

opencv-python, pytest, tensorboardX, wandb and PyYAML are licensed under MIT license.
pycocotools and torch are licensed under BSD-3-Clause license.
onnx is licensed under Apache 2.0 license.

easydict is licensed under LGPL license.