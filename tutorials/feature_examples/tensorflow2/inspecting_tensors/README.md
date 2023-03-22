<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# Inspecting Tensors Using Custom Outfeed Layers and a Custom Optimizer in TensorFlow 2

This example trains a choice of simple fully connected models on the MNIST
numeral dataset and shows how tensors (containing activations and gradients)
can be returned to the host via outfeeds for inspection.

This can be useful for debugging but can significantly increase the amount
of memory required on the IPU(s). When pipelining, use a small value for the
gradient accumulation count to mitigate this. Consider using a small
number of steps per epoch. Filters can be used to only return a subset of
the activations and gradients.

## File structure

* `mnist.py` The main Python script.
* `outfeed_callback.py` Contains a custom callback that dequeues an outfeed queue
  at the end of every epoch.
* `outfeed_layers.py` Custom layers that (selectively) add the inputs (for example,
  activations from the previous layer) to a dict that will be enqueued on an
  outfeed queue.
* `outfeed_optimizer.py` Custom optimizer that outfeeds the gradients generated
  by a wrapped optimizer.
* `outfeed_wrapper.py` Contains the `MaybeOutfeedQueue` class, see below.
* `README.md` This file.
* `requirements.txt` Required packages for the tests.
* `tests` Subdirectory containing test scripts.

### Class descriptions

This example uses the following classes:

* `outfeed_wrapper.MaybeOutfeedQueue` - a wrapper for an IPUOutfeedQueue that allows
  key-value pairs to be selectively added to a dictionary that can then be enqueued.
* `outfeed_optimizer.OutfeedOptimizer` - a custom optimizer that enqueues gradients
  using a `MaybeOutfeedQueue`,
  with the choice of whether to enqueue the gradients after they are computed
  (the pre-accumulated gradients) or before they are applied (the accumulated gradients).
* `outfeed_layers.Outfeed` - a Keras layer that puts the inputs into a dictionary
  and enqueues it on an IPUOutfeedQueue.
* `outfeed_layers.MaybeOutfeed` - a Keras layer that uses a MaybeOutfeedQueue to
  selectively put the inputs into a dict and optionally enqueues the dict. At the moment,
  this layer cannot be used with non-pipelined Sequential models.
* `outfeed_callback.OutfeedCallback` - a Keras callback to dequeue an outfeed
  queue at the end of every epoch, printing some statistics about the tensors.

See the `outfeed_*.py` files for further documentation.

## How to use this example

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the instructions in the Getting Started guide
   for your IPU system.
   Make sure to source the `enable.sh` script for Poplar and activate a Python
   virtualenv with a TensorFlow 2 wheel from the Poplar SDK installed
   (use the version appropriate to your operating system).

2) Train the graph, printing information via the callbacks

```
    python mnist.py
```

Example output:

```
Epoch 1/3

Gradients callback
key: Dense_128/bias:0_grad shape: (500, 128)
key: Dense_128/kernel:0_grad shape: (500, 256, 128)
Epoch 1 - Summary Stats
Index Name                         Mean         Std          Minimum      Maximum      NaNs    infs
0     Dense_128/bias:0_grad        -0.000678    0.019391     -0.151136    0.155004     False   False
1     Dense_128/kernel:0_grad      -0.000149    0.011651     -0.221480    0.222142     False   False

Multi-layer activations callback
key: Dense_128_acts shape: (2000, 32, 128)
key: Dense_10_acts shape: (2000, 32, 10)
Epoch 1 - Summary Stats
Index Name                Mean         Std          Minimum      Maximum      NaNs    infs
0     Dense_128_acts      0.729529     0.845325     0.000000     8.597726     False   False
1     Dense_10_acts       0.100000     0.265599     0.000000     1.000000     False   False

```

## Extra information

### Model

By default, the example runs a three layer fully connected model, pipelined over
two IPUs. Gradients for one of the layers, and activations for two of the layers,
are returned for inspection on the host. This can be changed using options.

For the single IPU models (Model and Sequential) gradients and activations are
returned for one layer.

### Options

The following command line options are available. See the code for other ways of
changing the behaviour of the example.

 * --model-type: One of "Model", "Sequential". Default is "Sequential".
 * --no-pipelining: If set, pipelining will not be used. Default is False (that is, pipelining is used).
 * --outfeed-pre-accumulated-gradients: If set then outfeed the pre-accumulated
   rather than accumulated gradients (only makes a difference when using gradient
   accumulation)
 * --use-gradient-accumulation: enables gradient accumulation even when not using pipelining. It is
   enabled by default when using pipelining.
 * --steps-per-epoch: The number of steps to run per epoch. Default is 2000.
 * --epochs: The number of epochs to run. Default is 3.
 * --gradients-filters: Space separated strings used to select which gradients
   should be added to the dict that is returned via an outfeed queue. Default is
   Dense_128. Set to none to get all gradients.
 * --activations-filters: Space separated strings used to select which activations
   from the second PipelineStage should be added to the dict that is returned via an outfeed queue. Set to none (default) to get the activations from
   both layers. Only applicable when using pipelined models.

## Tests

Some integration tests are included in the `tests` subdirectory.

Install the required packages:

```
    python3 -m pip install -r requirements.txt
```

Run the tests:

```
    python3 -m pytest tests
```
