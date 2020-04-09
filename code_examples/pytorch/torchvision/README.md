# Graphcore

## Using the pytorch functionality within popart

This directory contains an example of how to use the `popart.torch.trainingsession` for conversion of various models in torchvision.

### File structure

* `torchvision_examples.py` The main examples file, see below on how to use this.
* `requirements.txt` The list of additional requirements, on top of popart. (Torch and Torchvision).
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

    If you haven't already, install Torch and Torchvision, either by `pip install Torch Torchvision` or `pip install -r requirements.txt`. You should already have popart installed, along with it's dependencies.

    Full instructions can be found on https://pytorch.org

2) Download the data.

    For Cifar10/100, the data will automatically be downloaded by the program if not present at
    the location defined by the '--data-dir' option (default `/tmp/data`).

    For ImageNet follow the instruction on how to download and prepare the data. 
    <https://github.com/pytorch/examples/tree/master/imagenet>

3) Run the examples with whichever options you require.
    Currently this example is designed to work with Resnets 18, 34 and 50. You may need multiple IPUs for the larger models. For example run `python3 torchvision_examples.py --model-name=resnet18 --num-ipus=2` to run a 2-IPU resnet18 with default options.
     
    Run `python3 torchvision_examples.py --help` to see what options are available.

## More Information
  This uses the `popart.torch.TrainingSession` wrapper to convert a pytorch module from torchvision to the ONNX IR for use in popart on an IPU. 
  
  Please see https://pytorch.org/docs/stable/torchvision/models.html for information on the different models. Alternatively, construct your own ONNX compaitible pytorch model and replicate the `popart.torch.trainingsession` logic to load your model into popart.