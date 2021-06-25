PyTorch CNN dataset & data handling
---


### File structure

* `create_webdataset.py` Converts raw imagenet dataset to WebDataset format.
* `host_benchmark.py` Benchmark the host side throughput. 
* `data.py` Provides the dataloader.
* `preprocess.py` Optimized preprocess transformations.
* `README.md` This file.
* `webdataset_format.py` WebDataset loading code.
* `get_images.sh` Download the real images dataset.
* `validate_dataset.py` Validate the imagenet dataset(checks whether the dataset is corrupted)
* `distributed_webdataset.py` Create a repartition of the webdataset for the distributed training.
* `tfrecord_format.py` Helper functions for TF Record format
* `raw_imagenet.py` Helper functions for raw ImageNet dataset, which uses bounding boxes too.

### Validate the correctness of the dataset

Use the following script to calculate the checksum of the dataset.
```
python validate_dataset.py --imagenet-data-path <path> 
```


### How to maximize the infeed performance

1) Convert the dataset WebDatasetformat.

2) Distribute the dataset for the number of distributed instances. This step helps to avoid skipping any samples during the training.

### How to create WebDataset format
[WebDataset](https://github.com/tmbdev/webdataset) is a PyTorch Dataset implementation, which provides access to tar archives. Using tar archives reduces the IO overhead, by using larger data chunks. This makes the file access sequential instead of random access.

```
python create_webdataset.py --source <raw imagenet folder> --target <target folder> --samples-per-shard <global batch size> --shuffle --format img
```

Options:

`-h`                            Show usage information

`--source`                      Source folder

`--target`                      Target folder

`--shuffle`                     Shuffle the dataset

`--samples-per-shard`           Number of samples in each shard

`--format`                      Saving format of the data: save it as jpeg(`img`) or PyTorch tensor(`tensor`).

`--seed`                        Seed for shuffling.

`--train-preprocess-steps`      Provide the preprocessing steps for training. Options: [Resize(<size>), CenterCrop(<size>)

`--validation-preprocess-steps` Provide the preprocessing steps for validation. Options: [Resize(<size>), CenterCrop(<size>)

Note `img` use less disk space as it is compressed for storing, but requires more compute than the `tensor` version. The optimal format depends on the actual system configuration, whether IO (or host memory) is the bottleneck or the CPU. It is recommended to generate both versions of the dataset and benchmark it with `host_benchmark.py` script.

### How to distribute WebDataset

 ```
python distributed_webdataset.py --target <imagenet in webdataset format path> --num-instances <number of instances> 
```

Options:

`-h`                            Show usage information

`--target`                      The webdataset path

`--num-instances`               Number of instances when running with poprun       


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
