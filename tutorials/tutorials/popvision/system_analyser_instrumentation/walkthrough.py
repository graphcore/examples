"""
Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
"""

# Tutorial: Instrumenting applications

In this tutorial you will learn to use:

- the [PopVision System
  Analyser](https://docs.graphcore.ai/projects/system-analyser-userguide/),
  a desktop tool for profiling the code that runs on the host processors and interacts with IPUs”;
- the [`libpvti`
  module](https://docs.graphcore.ai/projects/libpvti/en/latest/) in
  Python which can be used to profile, time, and log information from your IPU
  applications and plot it directly in the PopVision System Analyser.
"""
"""
## How to run this tutorial
"""
"""
To run the Python version of this tutorial:
1. Download and install the Poplar SDK, see the [Getting
  Started](https://docs.graphcore.ai/projects/pytorch-quick-start/en/latest/quick-start-beginners.html#enable-poplar-sdk) guide for your IPU system.
2. For repeatability we recommend that you create and activate a Python virtual environment. You can do this with:
   a. create a virtual environment in the directory `venv`: `virtualenv -p python3 venv`;
   b. activate it: `source venv/bin/activate`.
3. Install the Python packages that this tutorial needs with `python -m pip install -r requirements.txt`.

sst_ignore_jupyter
"""
"""
To run the Jupyter notebook version of this tutorial:

1. Enable a Poplar SDK environment (see the [Getting
  Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for
  your IPU system)
2. In the same environment, install the Jupyter notebook server:
   `python -m pip install jupyter`
3. Launch a Jupyter Server on a specific port:
   `jupyter-notebook --no-browser --port <port number>`
4. Connect via SSH to your remote machine, forwarding your chosen port:
   `ssh -NL <port number>:localhost:<port number>
   <your username>@<remote machine>`

For more details about this process, or if you need troubleshooting, see our
[guide on using IPUs from Jupyter
notebooks](../../standard_tools/using_jupyter/README.md).
"""
"""
## Introduction
"""
"""
The Graphcore PopVision™ System Analyser is a desktop tool for analysing the
execution of IPU-targeted software on your host system processors.
It shows an interactive timeline visualisation of the execution steps involved,
helping you to identify any bottlenecks between the CPUs and IPUs.
This is particularly useful when you are scaling models to run on multiple CPUs and IPUs.

For this tutorial we are going to use a PopTorch MNIST example and add
instrumentation that can be viewed using the PopVision System Analyser.
Make sure the PopVision System Analyser is installed on your local machine, it can
be downloaded from the [Downloads Portal](https://downloads.graphcore.ai/).
"""
"""
Run the MNIST example with profiling enabled
"""
# %pip install -r requirements.txt
# ! PVTI_OPTIONS='{"enable":"true"}' python3 poptorch_mnist.py
# sst_ignore_md
# sst_ignore_code_only
"""
"""
import subprocess
import os

mnist_path = "./poptorch_mnist.py"
os.environ["PVTI_OPTIONS"] = '{"enable":"true", "directory": "reports"}'
output = subprocess.run(["python3", mnist_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(output.stdout.decode("utf-8"))
"""
When this has completed you will find a pvti file in the working directory,
for example "Tue_Nov_24_11:59:17_2022_GMT_4532.pvti".

> **Note**: You can specify an output directory for the pvti files to be
> written to:
>
> ```bash
> PVTI_OPTIONS='{"enable":"true", "directory": "reports"}' python3 poptorch_mnist.py
> ```
"""
"""
Open the PopVision System Analyser and then select "Open a report" and select
the pvti file generated. You may need to copy the pvti file to your local
machine.

You should then see the following profile information.

![PopVision System Analyser screenshot of mnist](./screenshots/mnist.png)
"""
"""
## Profiling execution of epochs
"""
"""
We are now going to modify the MNIST example to add instrumentation to clearly
show the epochs. (You can find the completed tutorial in the `complete`
directory)
"""
"""
Firstly, we need to import the libpvti library.

Add the import statement at the top of `poptorch_mnist.py`:
```python
import libpvti as pvti
```
"""
"""
Next we will need to create a trace channel. Add the `mnistPvtiChannel` as a
global object.

```python
mnistPvtiChannel = pvti.createTraceChannel("MNIST Application")
```
"""
"""
We are going to use the Python `with` keyword with a Python context manager to
instrument the epoch loop.

> **Note**: You will need to indent the contents of the loop.

```python
# Training loop
print("Running training loop.")
epochs = 10
for epoch in trange(epochs, desc="epochs"):
    with pvti.Tracepoint(mnistPvtiChannel, f"Epoch:{epoch}"):
        ...
```
"""
"""
Run the MNIST example again with profiling enabled
"""
output = subprocess.run(["python3", mnist_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(output.stdout.decode("utf-8"))
"""
We leave it as an exercise for the reader to add instrumentation of the
training & evaluation phases. When added you will see the following profile in
the PopVision System Analyser.

> **Note**: You can nest the Tracepoint statements.

![PopVision System Analyser screenshot of instrumented mnist](./screenshots/mnist_instrumented.png)
"""
"""
## Logging the training and validation losses
"""
"""
In addition to displaying function profiling, the System Analyser can plot
numerical data captured by the `libpvti` library.

In this section, we are going to add instrumentation to our Python script to
allow the System Analyser to plot the loss reported by PopTorch.

We have added the libpvti import in the previous section, so we need first to
create a pvti Graph object and then create series in the graph.
"""
"""
To create the graph we call the `pvti.Graph` constructor passing the name of
the graph:

```python
loss_graph = pvti.Graph("Loss", "")
```
"""
"""
Then create the series to which we will add the data:

```python
training_loss_series = loss_graph.addSeries("Training Loss")
validation_loss_series = loss_graph.addSeries("Validation Loss")
```
"""
"""
Finally after each training batch we will record the
training and validation loss. We take the loss and add it to our `training_loss_series`:

```python
output, loss = poptorch_model(data, labels)
# Record the training loss
training_loss_series.add(loss.item())
...

output, loss = poptorch_model(data, labels)
# Record the validation loss
validation_loss_series.add(loss.item())
```
"""
"""
Run the MNIST example again with profiling enabled
"""
output = subprocess.run(["python3", mnist_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(output.stdout.decode("utf-8"))
"""
When we view the resulting pvti report in the System Analyser (you may need to
scroll to the bottom of the page) it will show the loss graph looking something
like this:

![PopVision System Analyser screenshot of instrumented mnist loss](./screenshots/mnist_instrumented_loss.png)
"""
"""
> **Note**: The option to `merge all charts` has been enabled to combine all
> threads into a single row, to make it easier to align the flame graph with
> the line graph.
"""
"""
## Generating and profiling instant events
"""
"""
You can get insight into when particular sequences in the host code are executed by adding 'instant events'. This feature can be used to log events that occur during the execution of the application, such as receiving a message, errors/warnings or a change in a parameter such as epoch or learning rate.

For these purposes you may use 'instant events', which are like checkpoints. This feature adds trace points corresponding to a single point in time rather than a block.
"""
"""
For example, we are going to log the epoch number each time a new epoch begins, by using instant events:

```python
mnistInstantEventsChannel = pvti.createTraceChannel("Instant Events")
print("Running training loop.")
for epoch in trange(epochs, desc="epochs"):
    pvti.Tracepoint.event(mnistInstantEventsChannel, f"Epoch {epoch} begin")
    ...
```
"""
"""
Run the MNIST example again with profiling enabled
"""
output = subprocess.run(["python3", mnist_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(output.stdout.decode("utf-8"))
"""
You can use an existing trace channel to capture instant events, but we are using a separate one for the purposes of this tutorial.

When added you will see the following profile in the PopVision System Analyser. Instant events are represented by flags at the top of the profile:

![PopVision System Analyser screenshot of instrumented mnist instant events](./screenshots/mnist_instrumented_instant_events.png)
"""
"""
## Generating heatmap data to visualise numerical stability of tensors
"""
"""
To help understand the numerical distribution of your tensors, you can add 'heatmaps'. This feature
can be used to log tensor data through the execution of the application. This can be though of as a
"histogram over time" and your tensor data will be aggregated into bins based on your defined edges when
you create the heatmap.

We're using anchors to capture the tensor information from PopTorch. See the [debugging
help](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/debugging.html) for more details.
"""
"""
In this example, we are going to capture heatmap information for two convolution weight tensors:

```python
opts.anchorTensor("conv1_weight", "conv1.weight")
opts.anchorTensor("conv2_weight", "conv2.weight")
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)
conv1_heatmap = pvti.HeatmapDouble(
    "conv1.weight", torch.linspace(-16, 16, 33).tolist(), "2^x"
)
conv2_heatmap = pvti.HeatmapDouble(
    "conv2.weight", torch.linspace(-16, 16, 33).tolist(), "2^x"
)
print("Running training loop.")
for epoch in trange(epochs, desc="epochs"):
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        conv1_tensor = torch.abs(
            poptorch_model.getAnchoredTensor("conv1_weight")
        ).flatten()
        conv1_heatmap.add(conv1_tensor[conv1_tensor != 0].tolist())
        conv2_tensor = torch.abs(
            poptorch_model.getAnchoredTensor("conv2_weight")
        ).flatten()
        conv2_heatmap.add(conv1_tensor[conv1_tensor != 0].tolist())
    ...
```
"""
"""
Run the MNIST example again with profiling enabled
"""
output = subprocess.run(["python3", mnist_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(output.stdout.decode("utf-8"))
"""
When added you will see the following profile in the PopVision System Analyser. New heatmaps can be
seen at the top of the profile:

![PopVision System Analyser screenshot of instrumented mnist heatmaps](./screenshots/mnist_instrumented_heatmaps.png)
"""
"""
## Going further
"""
"""
The completed example also calculates accuracy of the model, and CPU load using
the `psutil` library, and plots both of them.

![PopVision System Analyser screenshot of instrumented mnist loss, accuracy & cpuload](./screenshots/mnist_instrumented_loss_accuracy_cpuload.png)
"""
"""
This is a very simple use case for adding instrumentation. The PopVision trace
instrumentation library (libpvti) provides other functions, classes & methods
to instrument your Python and C++ code. For more information please see the
[PVTI library documentation](https://docs.graphcore.ai/projects/libpvti/en/latest/).
"""
