PyTorch CNN dataset & data handling
---


### File structure

* `host_benchmark.py` Benchmark the host side throughput. 
* `data.py` Provides the dataloader.
* `preprocess.py` Optimized preprocess transformations.
* `README.md` This file.
* `get_images.sh` Download the real images dataset.
* `validate_dataset.py` Validate the imagenet dataset(checks whether the dataset is corrupted)
* `raw_imagenet.py` Helper functions for raw ImageNet dataset, which uses bounding boxes too.
* `augmentation.py` Contains custom augmentations, such as cutmix.

### Validate the correctness of the dataset

Use the following script to calculate the checksum of the dataset.
```
python validate_dataset.py --imagenet-data-path <path> 
```

### How to benchmark host-side data loading

Example:
```
python host_benchmark.py --data imagenet --batch-size 1024
```

Options:

`-h`                            Show usage information

`--batch-size`                  Batch size of the dataloader

`--data`                        Choose the dataset between: `real`, `generated`, `cifar10` or `imagenet`

`--imagenet-data-path`          The path of the downloaded imagenet dataset (only required, if imagenet is selected as data)

`--disable-async-loading`       Load data synchronously.

`--normalization-location`      Location of the input data normalization: `host` or `ipu`

It is possible to run it in distributed settings too:
```
poprun --offline-mode=yes --numa-aware=yes --num-instances 8 --num-replicas 8 python host_benchmark.py --data imagenet --batch-size 1024
```
