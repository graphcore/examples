## PerceiverIO 
https://arxiv.org/pdf/2107.14795.pdf 
> "Perceiver IO, a general-purpose architecture that handles data from arbitrary settings while scaling linearly with the size of inputs and outputs."


## Installation
1. activate SDK
2. download ImageNet-1k dataset (see a section below)
3. `pip install -r requirements.txt`
4. `python3 run_cpu.py configs/imagenet1k_classification.json`


## Unimodal
### Image classification based on ImageNet-1k

In order to obtain ImageNet-1k you have to download it manually. Please download it from (https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz). You will need to login and download the `imagenet_object_localization_patched2019.tar.gz` file, which is about 166 GB in size. Don't extract the file and use the `--dataset-path` argument to point the application to the file. 


## Optional arguments
```
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        Name of the dataset to use. Supported datasets:
                        ['cifar10', 'imagenet-1k']. (default: imagenet-1k)
  --dataset_path DATASET_PATH
                        Enabling training. (default: /localdata/datasets/image
                        net_object_localization_patched2019.tar.gz)
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
  --do_train [DO_TRAIN]
                        Whether to run training. (default: False)
  --do_eval [DO_EVAL]   Whether to run eval on the dev set. (default: False)
  --do_predict [DO_PREDICT]
                        Whether to run predictions on the test set. (default:
                        False)
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform. (default:
                        3.0)
```