# Half and mixed precision in PopTorch

This tutorial shows how to use half and mixed precision in PopTorch with the example task of fine-tuning a pretrained ResNet-18 model on a single Graphcore Mk2 IPU.

Requirements:
   - an installed Poplar SDK. See the Getting Started guide for your IPU hardware for details of how to install the SDK;
   - Other Python modules: `pip install torchvision tqdm`

### Motives for mixed precision

Data is stored in memory, and some formats to store that data require less memory than others. In a device's memory, when it comes to numerical data, we use either integers or floating points. But there are several types of integers and floating points. In this tutorial, we are going to talk about floating points represented in 32 bits (FP32) and 16 bits (FP16), and how to use these types in PopTorch in order to reduce the memory requirements of a model and speed up its runtime.

### Import the packages

We will be downloading a dataset and a pretrained model from `torchvision`, and `tqdm` is simple library to create progress bars that we will use to visually monitor the progress of the training job.

```python
import torch
import poptorch
import torchvision
from torchvision import transforms
from tqdm import tqdm
```

### Build the model

Now we are going to fine-tune a pretrained ResNet-18 model provided by the package `torchvision` for a classification task. We have to wrap that model into a new child class of `torch.nn.Module` in order to include to our model a loss function so that PopTorch is able to build a computational graph for training.

Fine-tuning in our case means that we will freeze all the parameters of the model, so that they are not updated, except for the last `Linear` layer which we replace with a new trainable `Linear` layer whose input units is 512 and output units the number of classes we want.

```python
class CustomResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Download/upload a pretrained RestNet-18 model
        self.resnet18 = torchvision.models.resnet18(pretrained=True)

        # Freeze all the parameters in the model
        for param in self.resnet18.parameters():
            param.requires_grad = False 
        
        # Replace with the last layer `fc` with a trainable `Linear` layer
        self.resnet18.fc = torch.nn.Linear(512, num_classes)

        # Add a loss function 
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, labels=None):
        out = self.resnet18(x)
        if self.training:
            return out, self.loss(out, labels)
        return out
```

To convert all a model's weights in PyTorch:
```python
model = model.half()
```

To convert specific layers to half precision:
```python
model.conv1 = model.conv1.half()
```

### Prepare the data

If the data is not converted to FP16 too, the model will be converted back to FP16. However, if the data is in FP16 and the model is entirely in FP32, some layers of the model may be converted to FP16. Therefore, if the data can be

```
transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.ConvertImageDtype(torch.float16)])

train_dataset = torchvision.datasets.FashionMNIST("./datasets/", transform=transform, download=True, train=True)
test_dataset = torchvision.datasets.FashionMNIST("./datasets/", transform=transform, download=True, train=False)
classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
num_classes = len(classes)
```

### Optimizers and loss scaling

Loss scaling is a technique that improves the numerical stability of a model in half or mixed precision during training. It consists of scaling up the loss value right before the start of backpropagation to prevent numerical underflow of the gradients. The value of the loss scaling factor is a parameter for the optimisers in `poptorch.optim`.

```python
poptorch.optim.AdamW(model.parameters(), loss_scaling=100)
```

While higher values of `loss_scaling` minimizes underflows, values too high can also generate overflows as well as hurt convergence of the loss. The optimal value depends on the model and the training job. This is therefore a hyperparameter to be tuned.

### Set PopTorch's options

Some features specific to the IPU demands to be configured and we use the `poptorch.Options` class to configure them. We covered some of the available options in the [introductory tutorial for PopTorch](https://github.com/graphcore/examples/tree/master/tutorials/pytorch/tut1_basics). 

Let's initialise our options object before we talk about stochastic rounding:

```python
opts = poptorch.Options()
```

>**NOTE**: This tutorial has been designed to run on a single MK2 IPU. If you do not have access to a MK2 IPU, you can use the option `useIpuModel` to run a simulation on CPU instead.

#### Stochastic rounding

When training in half or mixed precision, numbers multiplied between each other will need to be rounded in order to fit into memory. Stochastic rounding is the process of using a probability equation for the rounding, so that the _expectation_ of the computed value is equal to the exact value. It is also highly recommended to enable this feature when training neural networks with exclusively FP16 weights. You can read more about stochasitc rounding [here]().

With the IPU, this feature is implemented directly in the hardware and requires the users to enable it. However, PopTorch surfaces this option from PopART, the lower level framework that underpins PopTorch, meaning we will be calling `poptorch.Options.Popart.set` to enable stochastic rounding. We thus do:

```python
opts.Popart.set("enableStochasticRounding", True)
```

### Train the model

```python
train_dataloader = poptorch.DataLoader(opts, 
                                       train_dataset, 
                                       batch_size=12, 
                                       shuffle=True, 
                                       num_workers=40, 
                                       mode=poptorch.DataLoaderMode.Async)
```

```python
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)

epochs = 5
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss
    print (f"Epoch {epoch+1}/{epochs} - Loss: {total_loss}")
```