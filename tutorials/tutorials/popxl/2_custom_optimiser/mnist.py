"""
Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
"""
# PopXL Custom Optimiser

## Introduction

We saw in the [PopXL basics tutorial](../1_basic_concepts) that we can
easily create graphs with an internal state using the `addons.Module` class and we
used modules to implement dense layers of a simple neural network.

In this tutorial, we will learn how to implement a custom optimiser: the [Adam
optimiser](https://paperswithcode.com/method/adam). Many optimisers, like
layers, must manage some persistent variables. To manage this internal state we
will re-use the same programming pattern with `addons.Module` as a base class to
the optimiser.

Once you've finished this tutorial, you will:

- be able to write your own custom optimiser for your models.
- understand graph caching and how it helps subgraph reuse.
- have used some of the built-in rules within the `popxl.ops.var_updates` module
  which are useful for a variety of tasks including updating your optimiser's
  internal state variables.

If you are unfamiliar with PopXL, you may want to try out
[the tutorial covering the basic concepts](../1_basic_concepts). You may also want
to refer to the [PopXL user guide](https://docs.graphcore.ai/projects/popxl/).
"""
"""
## Requirements

1. Install a Poplar SDK (version 2.6 or later) and source the `enable.sh`
   scripts for both PopART and Poplar as described in the [Getting Started
   guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU
   system.
2. Create a Python virtual environment: `python3 -m venv <virtual_env>`.
3. Activate the virtual environment: `. <virtual_env>/bin/activate`.
4. Update `pip`: `pip3 install --upgrade pip`.
5. Install requirements `pip3 install -r requirements.txt` (this will also
   install popxl.addons).

```bash
python3 -m venv virtual_env
. virtual_env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

# sst_ignore_jupyter
"""
"""
To run the Jupyter Notebook version of this tutorial:

1. Install a Poplar SDK (version 2.6 or later) and source the `enable.sh`
   scripts for both PopART and Poplar as described in the [Getting Started
   guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU
   system.
2. Create a virtual environment.
3. In the same virtual environment, install the Jupyter Notebook server: `python
   -m pip install jupyter`.
4. Launch a Jupyter Server on a specific port: `jupyter-notebook --no-browser
   --port <port number>`. Be sure to be in the virtual environment.
5. Connect via SSH to your remote machine, forwarding your chosen port: `ssh -NL
   <port number>:localhost:<port number> <your username>@<remote machine>`.

For more details about this process, or if you need troubleshooting, see our
[guide on using IPUs from Jupyter
Notebooks](../../standard_tools/using_jupyter/README.md).

If using VS Code, Intellisense can help you understand the tutorial code. It
will show function and class descriptions when hovering over their names and
lets you easily jump to their definitions. Consult the [VS Code setup
guide](../VSCodeSetup.md) to use Intellisense for this tutorial.
"""

# %pip install -q -r requirements.txt
# sst_ignore_md
# sst_ignore_code_only
# sst_hide_output

"""
## Imports

We start by importing all the modules we will need for this
tutorial:
"""

from typing import Dict, Mapping, Optional, Union
from functools import partial

import numpy as np
import torch
import torchvision
from tqdm import tqdm

import popxl
import popxl_addons as addons
import popxl.ops as ops

np.random.seed(42)

"""
## Defining the Adam optimiser

Below, we implement the Adam optimiser by deriving from the `addons.Module`
class. The `Adam` class defines the update step for a single variable in its
`build` method.
First we will see how to correctly deal with in-place operations, then we will
define the update step using PopXL's `var_updates` module,
and, finally we will test and inspect our optimiser on a single variable.

### Managing in-place ops

The update is performed in-place on the `weight` argument which contains the
model variable updated by the Adam optimiser. Because this operation is in-place, we pass the
argument as a `TensorByRef` and use the `@popxl.in_sequence()` decorator to
prevent the operations from being rearranged by the compiler. The rest of the
definition follows the same pattern used to add weights to our layers in the
[PopXL basics tutorial](../1_basic_concepts).
"""


class Adam(addons.Module):
    # We need to specify `in_sequence` because many operations are in-place
    # and their order shouldn't be changed
    @popxl.in_sequence()
    def build(
        self,
        weight: popxl.TensorByRef,
        grad: popxl.Tensor,
        *,
        lr: Union[float, popxl.Tensor],
        beta1: Union[float, popxl.Tensor] = 0.9,
        beta2: Union[float, popxl.Tensor] = 0.999,
        eps: Union[float, popxl.Tensor] = 1e-5,
        weight_decay: Union[float, popxl.Tensor] = 0.0,
        first_order_dtype: popxl.dtype = popxl.float16,
        bias_correction: bool = True,
    ):
        # Gradient estimator for the variable `weight` - same shape as the variable
        first_order = self.add_variable_input(
            "first_order",
            partial(np.zeros, weight.shape),
            first_order_dtype,
            by_ref=True,
        )
        ops.var_updates.accumulate_moving_average_(first_order, grad, f=beta1)

        # Variance estimator for the variable `weight` - same shape as the variable
        second_order = self.add_variable_input(
            "second_order", partial(np.zeros, weight.shape), popxl.float32, by_ref=True
        )
        ops.var_updates.accumulate_moving_average_square_(second_order, grad, f=beta2)

        # Adam is a biased estimator: provide the step variable to correct bias
        step = None
        if bias_correction:
            step = self.add_variable_input("step", partial(np.zeros, ()), popxl.float32, by_ref=True)

        # Calculate the weight increment with an Adam heuristic
        # Here we use the built-in `adam_updater`, but you can write your own.
        dw = ops.var_updates.adam_updater(
            first_order,
            second_order,
            weight=weight,
            weight_decay=weight_decay,
            time_step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=eps,
        )

        # in-place weight update: weight += (-lr)*dw
        ops.scaled_add_(weight, dw, b=-lr)


# sst_hide_output
"""
The Adam optimiser needs state to store the mean and uncentred variance (first and second
moments) of the gradients. These need to be of type `Variable`, hence we add them with
`Module.add_variable_input`, creating named inputs for them (`first_order` and
`second_order`).

We used `Module.add_variable_input` in the [PopXL introductory
tutorial](../1_basic_concepts) to add weights to our layers. However, in the
`Adam` implementation you should notice a few differences.

- We now have a `@popxl.in_sequence()` decorator on top of the `build` method.
  This forces all operations to be added in the exact order we define
  them, enforcing topological constraints between them. This is necessary here
  since most of the optimiser operations are **in-place**, hence their order of
  execution must be strictly preserved. Remember this whenever you have in-place
  operations.
- The `weight` input is a `popxl.TensorByRef`: any change made to this variable
  will be automatically copied to the parent graph. See
  [TensorByRef](https://docs.graphcore.ai/projects/popxl/en/3.1.0/api.html#popxl.Ir.create_graph)
  for more information.
- Some parameters, such as learning rate or weight decay, are defined as
  `Union[float, popxl.Tensor]`. If the parameter was provided as a simple
  `float`, it would be "baked" into the graph, with no possibility of changing
  it at run-time. Instead, if the parameter is a `Tensor` (or `TensorSpec`) it
  will appear as an input to the graph, which needs to be provided when calling
  the graph. If you plan to change a parameter (for example, because you have a
  learning rate schedule), this is the way to go.

The rest of the logic is straightforward:

- We update `first_order`, the mean estimator of the `weight` gradient.
- We update `second_order`, the uncentred variance estimator of `weight`
  gradient.
- We optionally correct the estimators, since they are biased.
- We compute the increment delta-weight `dw`.
- We update the variable `weight` with `scaled_add_` to implement the equation
  `weight -= lr * dw`.

### Using the `var_updates` module

The `ops.var_updates` module contains several useful update rules (you
can also create your own). In this example, we will use three of the built-in
rules:

- `ops.var_updates.accumulate_moving_average_(average, new_sample, coefficient)`
  updates `average` in-place with an exponential moving average rule:

  ```python
  average = (coefficient * average) + ((1 - coefficient) * new_sample)
  ```

- `accumulate_moving_average_square_(average, new_sample, coefficient)` updates
  `average` in-place but uses the square of the sample.
- `ops.var_updates.adam_updater(...)` returns the Adam increment `dw` required
  for the weight update. This is computed using the Adam internal state which
  comprises of the first and second moments.

### Using our custom optimiser

Let's inspect the optimiser graph and its use in a simple example.
"""

ir = popxl.Ir(replication=1)

with ir.main_graph:
    var = popxl.variable(np.ones((2, 2)), popxl.float32)
    grad = popxl.variable(np.full((2, 2), 0.1), popxl.float32)

    # create graph and factories - float learning rate
    adam_facts, adam = Adam(cache=True).create_graph(var, var.spec, lr=1e-3)

    # create graph and factories - Tensor learning rate
    adam_facts_lr, adam_lr = Adam().create_graph(var, var.spec, lr=popxl.TensorSpec((), popxl.float32))

    print("Adam with float learning rate\n")
    print(adam.print_schedule())
    print("\n Adam with tensor learning rate\n")
    print(adam_lr.print_schedule())

    # instantiate optimiser variables
    adam_state = adam_facts.init()
    adam_state_lr = adam_facts_lr.init()

    # optimization step for float lr: call the bound graph providing the
    # variable to update and the gradient
    adam.bind(adam_state).call(var, grad)

    # optimization step for tensor lr: call the bound graph providing the
    # variable to update, the gradient and the learning rate
    adam_lr.bind(adam_state_lr).call(var, grad, popxl.constant(1e-3))

ir.num_host_transfers = 1
session = popxl.Session(ir, "ipu_hw")
print("\n Before Adam update")
var_data = session.get_tensor_data(var)
state = session.get_tensors_data(adam_state.tensors)
print("Variable:\n", var)
print("Adam state:")
for name, data in state.items():
    print(name, "\n", state[name])

with session:
    session.run()

print("\n After Adam update")
var_data = session.get_tensor_data(var)
state = session.get_tensors_data(adam_state.tensors)
print("Variable:\n", var)
print("Adam state:")
for name, data in state.items():
    print(name, "\n", state[name])


"""
## MNIST with Adam

We can now refactor our MNIST example to incorporate the Adam optimiser. Note
that we need an optimiser for each variable: we first define the
`optimiser_step` function which creates the graph for each variable and performs
a full weight update for all the variables. Since the `Adam` module uses
`cache=True`, if two graphs happens to be the same, the same graph will be
re-used.

We will use a simple float learning rate (rather than `Tensor`), since we don't
plan to change its value during training.
"""


def optimiser_step(
    variables,
    grads: Dict[popxl.Tensor, popxl.Tensor],
    optimiser: addons.Module,
    learning_rate: popxl.float32 = 1e-3,
):
    """
    Update all variables creating per-variable optimisers.
    """
    for name, var in variables.named_tensors.items():
        # Create optimiser and state factories for the variable
        opt_facts, opt_graph = optimiser.create_graph(
            var, var.spec, lr=learning_rate, weight_decay=0.0, bias_correction=False
        )
        state = opt_facts.init()

        # Bind the graph to its state and call it.
        # Both the state and the variables are updated in-place and are passed
        # by ref, hence after the graph is called they are updated.
        opt_graph.bind(state).call(var, grads[var])


"""
We load the data, and define our network using exactly the same code as in the
PopXL basics tutorial:
"""


def get_mnist_data(test_batch_size: int, batch_size: int):
    training_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "~/.torch/datasets",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # mean and std computed on the training set.
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    validation_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "~/.torch/datasets",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        drop_last=True,
    )

    return training_data, validation_data


class Linear(addons.Module):
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        # add a state variable to the module
        w = self.add_variable_input(
            "weight",
            partial(np.random.normal, 0, 0.02, (x.shape[-1], self.out_features)),
            x.dtype,
        )
        y = x @ w
        if self.bias:
            # add a state variable to the module
            b = self.add_variable_input("bias", partial(np.zeros, y.shape[-1]), x.dtype)
            y = y + b

        return y


class Net(addons.Module):
    def __init__(self, cache: Optional[addons.GraphCache] = None):
        super().__init__(cache=cache)
        self.fc1 = Linear(512)
        self.fc2 = Linear(512)
        self.fc3 = Linear(512)
        self.fc4 = Linear(10)

    def build(self, x: popxl.Tensor):
        x = x.reshape((-1, 28 * 28))
        x = ops.gelu(self.fc1(x))
        x = ops.gelu(self.fc2(x))
        x = ops.gelu(self.fc3(x))
        x = self.fc4(x)
        return x


"""
The training code is almost unchanged from that of the [PopXL basics
tutorial](../1_basic_concepts), the only difference being that we now call our
`Adam` class and `optimiser_step` function as the optimiser instead of the
simple `scaled_add_`:
"""


def train_program(batch_size, device, learning_rate):
    ir = popxl.Ir(replication=1)

    with ir.main_graph:
        # Create input streams from host to device
        img_stream = popxl.h2d_stream((batch_size, 28, 28), popxl.float32, "image")
        img_t = ops.host_load(img_stream)  # load data
        label_stream = popxl.h2d_stream((batch_size,), popxl.int32, "labels")
        labels = ops.host_load(label_stream, "labels")

        # Create forward graph
        facts, fwd_graph = Net().create_graph(img_t)

        # Create backward graph via autodiff transform
        bwd_graph = addons.autodiff(fwd_graph)

        # Initialise variables (weights)
        variables = facts.init()

        # Call the forward graph with call_with_info because we want to retrieve
        # information from the call site
        fwd_info = fwd_graph.bind(variables).call_with_info(img_t)
        x = fwd_info.outputs[0]  # forward output

        # Compute loss and starting gradient for backpropagation
        loss, dx = addons.ops.cross_entropy_with_grad(x, labels)

        # Setup a stream to retrieve loss values from the host
        loss_stream = popxl.d2h_stream(loss.shape, loss.dtype, "loss")
        ops.host_store(loss_stream, loss)

        # Retrieve activations from the forward graph
        activations = bwd_graph.grad_graph_info.inputs_dict(fwd_info)

        # Call the backward graph providing the starting value for
        # backpropagation and activations
        bwd_info = bwd_graph.call_with_info(dx, args=activations)

        # Adam optimiser, with cache
        grads_dict = bwd_graph.grad_graph_info.fwd_parent_ins_to_grad_parent_outs(fwd_info, bwd_info)
        optimiser = Adam(cache=True)
        optimiser_step(variables, grads_dict, optimiser, learning_rate)

    ir.num_host_transfers = 1
    return popxl.Session(ir, device), [img_stream, label_stream], variables, loss_stream


"""
You will notice above that we created the `Adam` module using `cache=True`. This
will enable graph reuse, if possible, when calling `optimiser.create_graph`. For
our optimiser this would be when there are multiple variables with the same
shape and data type.
"""
"""
Now let's run a training session.

Since we are using the Adam optimiser, we need to use a smaller learning rate than before.
"""

train_batch_size = 8
test_batch_size = 80
device = "ipu_hw"
learning_rate = 1e-3
epochs = 1

training_data, test_data = get_mnist_data(test_batch_size, train_batch_size)
train_session, train_input_streams, train_variables, loss_stream = train_program(
    train_batch_size, device, learning_rate
)

num_batches = len(training_data)
with train_session:
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        bar = tqdm(training_data, total=num_batches)
        for data, labels in bar:
            inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                zip(train_input_streams, [data.squeeze().float(), labels.int()])
            )
            loss = train_session.run(inputs)[loss_stream]
            bar.set_description(f"Loss:{loss:0.4f}")

"""
"""

# Retrieve the trained weights to use during inference
train_vars_to_data = train_session.get_tensors_data(train_variables.tensors)

"""
## Validation

As we did previously, to test our model we need to create an inference-only
program and run it on the test dataset.
"""


def test_program(test_batch_size, device):
    ir = popxl.Ir(replication=1)

    with ir.main_graph:
        # Inputs
        in_stream = popxl.h2d_stream((test_batch_size, 28, 28), popxl.float32, "image")
        in_t = ops.host_load(in_stream)

        # Create graphs
        facts, graph = Net().create_graph(in_t)

        # Initialise variables
        variables = facts.init()

        # Forward
        (outputs,) = graph.bind(variables).call(in_t)
        out_stream = popxl.d2h_stream(outputs.shape, outputs.dtype, "outputs")
        ops.host_store(out_stream, outputs)

    ir.num_host_transfers = 1
    return popxl.Session(ir, device), [in_stream], variables, out_stream


"""
We create the test program and copy the trained weights to it:
"""

test_session, test_input_streams, test_variables, out_stream = test_program(test_batch_size, device)

train_vars_to_test_vars = train_variables.to_mapping(test_variables)

test_vars_to_data = {
    test_var: train_vars_to_data[train_var].copy() for train_var, test_var in train_vars_to_test_vars.items()
}

test_session.write_variables_data(test_vars_to_data)

"""
Finally, let's run the test session and measure the accuracy:
"""


def accuracy(predictions: np.ndarray, labels: np.ndarray):
    ind = np.argmax(predictions, axis=-1).flatten()
    labels = labels.detach().numpy().flatten()
    return np.mean(ind == labels) * 100.0


num_batches = len(test_data)
sum_acc = 0.0
with test_session:
    for data, labels in tqdm(test_data, total=num_batches):
        inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
            zip(test_input_streams, [data.squeeze().float(), labels.int()])
        )
        output = test_session.run(inputs)
        sum_acc += accuracy(output[out_stream], labels)

test_set_accuracy = sum_acc / len(test_data)
print(f"Accuracy on test set: {test_set_accuracy:0.2f}%")

"""
## Conclusion

In this tutorial we wrote a custom optimiser using the popxl.addons API. We
achieved the following:

- built an Adam Optimiser (by subclassing `addons.Module`) and ran it with an
  MNIST model.
- used the `popxl.in_sequence()` function and learnt why it is needed, to
  prevent operations from being rearranged by the compiler.
- used `popxl.TensorByRef` to pass variable updates back to the parent graph.
- explored some of the functions within the `popxl.ops.var_updates` module.
- became familiar with using `Tensor` parameters versus simpler (built-in) types
  in `addons.Module.build`, when declaring dynamic parameters.
- exploited graph caching in `addons.Module`: `addons.Module(cache=True)` to
  enable graph reuse.

To try out more features in PopXL [look at our other
tutorials](../../README.md).

You can also read our [PopXL User
Guide](https://docs.graphcore.ai/projects/popxl/en/3.1.0/) for more
information.

As the PopXL API is still experimental, we would love to hear your feedback on it
([support@graphcore.ai](mailto:support@graphcore.ai?subject=PopXL%20Feedback)).
Your input could help drive its future direction.
"""
