Graphcore

## Image classification inference on IPU using Pytorch

The following models are supported using this inference harness.

1. Resnet18
2. Resnet34
3. ResNet50
4. EfficientNet B0


### File structure

* `run_benchmark.py` Driver script for running inference.
* `get_images.sh` Script to fetch sample test images.
* `README.md` This file.
* `data.py` Creates the DataLoader to load images.

### How to use this demo

1) Activate the Poptorch environment (if you haven't done already).
   
2) Download the images.

       ./get_images.sh

  This will create and populate the `images/` directory with sample test images.

3) Run the graph.
```
       python3 run_benchmark.py --data real --replicas 1  --batch-size 1 --model resnet18 --device-iteration 1
```

#### Options
The program has a few command-line options:

`-h` Show usage information.

`--batch-size`        Sets the batch size for inference.

`--model`             Select the model for inference.

`--data`              Choose the dataset between real and synthetic.

`--pipeline-splits`   List of layers to create stages of the pipeline. Each stage runs on different IPUs

`--replicas`          Number of replicas.

`--device-iteration`  Sets the device iteration: the number of inference steps before return to the host.

`--precision`         Sets the floating point precision: full or half.
