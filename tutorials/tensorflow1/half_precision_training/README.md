
TensorFlow 1 on the IPU: training a model using half- and mixed-precision
=========================================================================


Table of Contents
=================
* [Introduction](#introduction)
* [Using FP16 in practice](#using-fp16-in-practice)
    + [Support for FP16 in TensorFlow](#support-for-fp16-in-tensorflow)
    + [Review of common numerical issues](#review-of-common-numerical-issues)
    + [Inaccuracies in parameter updates](#inaccuracies-in-parameter-updates)
      - [Method 1: Using stochastic rounding](#method-1-using-stochastic-rounding)
      - [Method 2: Store and update the parameters in FP32](#method-2-store-and-update-the-parameters-in-fp32)
    + [Underflowing gradients and loss scaling](#underflowing-gradients-and-loss-scaling)
    + [Diagnosing numerical issues](#diagnosing-numerical-issues)
    + [Other considerations](#other-considerations)
* [Code examples](#code-examples)
    + [Running the examples](#running-the-examples)
    + [Stochastic rounding example](#stochastic-rounding-example)
    + [FP32 master weights example](#fp32-master-weights-example)
    + [Optional command line arguments](#optional-command-line-arguments)
    + [Other examples](#other-examples)

# Introduction

On computing devices, real numbers are represented by one of several floating point formats, which vary in how many bits they use to represent each number. Using more bits allows for greater precision and a wider range of representable numbers, whereas using fewer bits allows for faster calculations and reduces memory and power usage. In deep learning applications, where less precise calculations are acceptable and throughput is critical, using a lower precision format can provide substantial gains in performance.

The Graphcore IPU provides native support for two floating-point formats:

- IEEE single-precision, which uses 32 bits for each number
- IEEE half-precision, which uses 16 bits for each number

These are commonly known as "FP32" and "FP16", respectively, and we refer to them as such throughout this tutorial.

Some applications which use FP16 do all calculations in FP16, whereas others use a mix of FP16 and FP32. The latter approach is known as *mixed precision*.

The IPU also provides native support for stochastic rounding, a technique which makes some operations in FP16 more accurate on average by making the rounding operation statistically unbiased.

In this tutorial, we will discuss how to use FP16 and stochastic rounding using TensorFlow 1 for the IPU, as well as best practices for ensuring stability. Two code examples, `stochastic_rounding.py` and `float32_master_weights.py`, are also provided, which demonstrate different solutions to the same numerical issue. Currently, this tutorial does not cover models which are run using the `IPUEstimator` API.

If you are not familiar with how to run TensorFlow 1 models on the IPU, you may wish to read our [introductory tutorial](../basics) on the subject.

# Using FP16 in practice

## Support for FP16 in TensorFlow

TensorFlow provides the native data type `tf.float16`, which corresponds to the IEEE half-precision type supported by the IPU. You can use `tf.cast` to convert tensors between numerical types. You can pass a `dtype` argument to `tf.constant` to create a constant tensor with a particular type.

If you are using NumPy, you can convert the type of an `ndarray` (the standard representation of a multi-dimensional array in NumPy) with its `astype` method. This will return a copy of the array with the specified data type. You can either pass a NumPy data type (such as `np.float16`) or a string (such as `'float16'`) as input to this method.

Most binary operations in TensorFlow will only work if the inputs to the operation have the same type. You cannot, for example, have an operation that adds a `tf.float16` value to a `tf.float32` value, or does this as a subcomputation. Operations with inputs with mismatching types will raise a `TypeError`, such as this one:

```
TypeError: Input 'y' of 'AddV2' Op has type float32 that does not match type float16 of argument 'x'.
```

The output of a TensorFlow operation will have the same type as its inputs (unless that operation is a cast, for example). If the operation is deemed too unstable for computation purely in FP16, the inputs will be upcast to FP32 for the compute, and the outputs will be downcast back to FP16 before being returned. Since outputs have the same type as inputs, it is often sufficient to provide your inputs in FP16 for the whole model to be executed in FP16.

For many models, it is sufficient to perform all computations in FP16, without making any modifications for stability. However, there are a number of numerical issues that can arise when using FP16, which may affect your model. In the remainder of this tutorial, we will discuss some common issues that can occur and how you can address them.

## Review of common numerical issues

Two well-known issues that can arise when using floating-point numbers are overflow and underflow. Overflow occurs when a number of large magnitude exceeds the range that can be represented. Underflow occurs when a number of small magnitude is too small to be represented, and so is approximated as 0. Both of these are more likely to occur in FP16: the maximum representable value in FP16 is 65504 and the minimum representable positive value in FP16 is approximately `6.0e-08`, compared to approximately `1.7e+38` and `1.7e-38` respectively in FP32.

Another problem with all floating-point formats is that often, the result of an addition or subtraction is not a representable number, so the result has to be rounded to the nearest representable number. For example, let's calculate `0.0004 + 0.25 - 0.25` in FP16:

```python
import numpy as np

x_in_float16 = np.array([0.0004]).astype(np.float16)
x_in_float16 += 0.25
x_in_float16 -= 0.25
print('0.0004 + 0.25 - 0.25 = ', x_in_float16[0])
```

The output of this is:

```
0.0004 + 0.25 - 0.25 = 0.0004833
```

We get `0.0004833` instead of `0.0004` because the nearest representable number to `0.2504` in FP16 is approximately `0.2504833`.

Generally speaking, the greater the difference in magnitude between the addends, the less accurate the rounded result will be. For example:

```python
import numpy as np

big_number = .25

for small_number in [0.2, 0.02, 0.002, 0.0002]:
    x_in_float16 = np.array([small_number]).astype(np.float16)
    x_in_float16 += big_number
    x_in_float16 -= big_number
    print(f'{small_number:.4f} + {big_number} - {big_number} = {x_in_float16[0]:.7f}, '
          f'relative error: {(x_in_float16[0] - small_number)/small_number:+.1%}')
```

The output of this is:

```
0.2000 + 0.25 - 0.25 = 0.1999512, relative error: -0.0%
0.0200 + 0.25 - 0.25 = 0.0200205, relative error: +0.1%
0.0020 + 0.25 - 0.25 = 0.0019531, relative error: -2.3%
0.0002 + 0.25 - 0.25 = 0.0002441, relative error: +22.1%
```

This can be particularly damaging if the nearest representable number is one of the addends. For example, let's calculate `0.0001 + 0.25 - 0.25` in FP16:

```python
import numpy as np

x_in_float16 = np.array([0.0001]).astype(np.float16)
x_in_float16 += 0.25
x_in_float16 -= 0.25
print('0.0001 + 0.25 - 0.25 = ', x_in_float16[0])
```

The output of this is:

```
0.0001 + 0.25 - 0.25 = 0.0
```

We get `0.0` because the nearest representable number to `0.2501` in FP16 is `0.25`. This is known as *swamping*.

If a calculation consists of a sum of many values, the aggregate of many small errors in individual additions can result in very significant errors. For example, we can try starting with `0.0` and incrementing by `0.0001` 10,000 times:

```python
import numpy as np

x_in_float16 = np.array([0.0]).astype(np.float16)
for _ in range(10000)
    x_in_float16 += 0.0001
print('Sum: ', x_in_float16[0])
```

The correct result would be `0.0001 * 10000 = 1.0`, but when we run this code we get:

```
Sum: 0.25
```

This is because each time `0.0001` is added to `0.25`, the result is rounded to `0.25`, so we never progress beyond this value, giving a highly inaccurate result.


## Inaccuracies in parameter updates

When training a neural network, the product of the gradients and the learning rate is often many orders of magnitude smaller than the parameters. This can lead to inaccuracies as described in the previous section, which can lead to poorer results. In some cases, swamping can occur, and parts of the network or even the entire network will completely fail to train.

There are two main techniques which are used to address this:

- Using stochastic rounding
- Storing and updating the parameters in FP32


### Method 1: Using stochastic rounding

The IPU has native support for stochastic rounding, a technique which makes some operations in FP16 more accurate on average by making the rounding operation statistically unbiased. The idea of stochastic rounding is that instead of always rounding to the nearest representable number, we round up or down with a probability such that the expected value after rounding is equal to the value before rounding. Since the expected value of an addition after rounding is equal to the exact result of the addition, the expected value of a sum is also its exact value.

This means that on average, the values of the parameters of a network will be close to the values they would have had if a higher-precision format had been used. The added bonus of using stochastic rounding is that the parameters can be stored in FP16, which means the parameters can be stored using half as much memory. This can be especially helpful when training with small batch sizes, where the memory used to store the parameters is proportionally greater than the memory used to store parameters when training with large batch sizes.

When stochastic rounding is used, it is enabled during the IPU configuration step. When you enable stochastic rounding, you should also explicitly enable or disable floating-point exceptions and NaN-on-overflow mode. See [the section of this tutorial on diagnosing numerical issues](#diagnosing-numerical-issues) for further details. In a neural network which is trained using stochastic rounding, all computations can be performed in FP16.

Here's an example of how stochastic rounding can be used, that also demonstrates its effectiveness against swamping. We repeat our calculation from the end of the last section, starting with `0.0` and incrementing by `0.0001` 10,000 times, but using stochastic rounding this time. We repeat our calculation 20 times.


```python
# Configure device with 1 IPU and compile

ipu_configuration = ipu.utils.create_ipu_config()

ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration, num_ipus=1)

# Enable stochastic rounding
# We also explicitly enable all floating-point exceptions
ipu_configuration = ipu.utils.set_floating_point_behaviour_options(opts=ipu_configuration,
                                                                   esr=True,
                                                                   nanoo=True,
                                                                   oflo=True, inv=True, div0=True)

ipu.utils.configure_ipu_system(config=ipu_configuration)

# Define addition function and compile it for IPU

def add_two_numbers(x, y):
    return x + y

x = tf.placeholder(tf.float16, [20])
y = tf.placeholder(tf.float16, [20])

with ipu.scopes.ipu_scope('/device:IPU:0'):
    add_on_ipu = ipu.ipu_compiler.compile(add_two_numbers, [x, y])

with tf.Session() as sess:

    running_total = [0. for _ in range(20)]
    small_numbers = [0.0001 for _ in range(20)]

    for _ in range(10000):

        running_total = sess.run(add_on_ipu, feed_dict={x: running_total, y: small_numbers})[0]

    print(running_total)
```

Here's some example output:

```python
[1.005  0.993  0.991  0.9966 0.986  0.997  1.019  0.9985 1.004  1.03
 0.991  0.9873 0.9946 1.013  1.006  0.9995 1.02   0.9653 0.9854 0.9985]
```

All of the results are reasonably close to the correct result of `1.0`, and are certainly much closer than the result without using stochastic rounding. Indeed, the average is `0.999`, which corresponds with the fact that the expected value of the sum is equal to the true value. There is some variance in the results, which is due to the inherent randomness of stochastic rounding.

An example of training ResNet models on CIFAR-10 in FP16 using stochastic rounding is provided in `stochastic_rounding.py`.


### Method 2: Store and update the parameters in FP32

An alternative approach to dealing with inaccuracies in the parameter update step is to store and update the parameters in FP32. FP32 computations are much more precise than those in FP16, so any inaccuracies in the parameter update will be much smaller. Also, casting the parameters back and forth between steps comes with little computational overhead, and there is much less randomness involved than using stochastic rounding.

For these reasons, using this method is recommended over using stochastic rounding unless memory constraints prevent you from doing so.

One drawback of this method is that the Keras APIs for defining models do not come with the low-level facilities needed to implement it. If you are porting a Keras model, you may wish to experiment with the `tf.keras.mixed_precision` API (see [here](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/mixed_precision/experimental) for the documentation), which will allow you to do something very similar. However, this is experimental in TensorFlow 1 and has not been extensively tested on the IPU.

One way to implement this method with a model defined in pure TensorFlow is to create the variables for the parameters in FP32 and then cast them to FP16 before doing any compute. Here's an example of how a dense layer might be implemented using this technique:

```python
# Dense layer with FP16 compute and FP32 variables
#     where input is a 1D tensor of FP16 values
def method_2_dense(inputs_float16, units_out):

    units_in = inputs_float16.get_shape().as_list()[0]
    
    # tf.matmul requires both arguments to be 2D tensors
    inputs_float16 = tf.reshape(inputs_float16, [units_in, 1])
    
    # Create weights in FP32
    weights = tf.get_variable(
        name="weights",
        shape=[units_out, units_in],
        dtype=tf.float32,
        trainable=True
    )
    
    # Cast to FP16 before doing the compute    
    weights_float16 = tf.cast(weights, tf.float16)

    # Do the same for the biases
    biases = tf.get_variable(
        name="biases",
        shape=[units_out, 1],
        dtype=tf.float32,
        trainable=True
    )
    
    biases_float16 = tf.cast(biases, tf.float16)

    # Do compute in FP16
    output = tf.matmul(weights_float16, inputs_float16) + biases_float16
    
    # Return output to 1D
    output = tf.reshape(output, [units_out])
    
    return output    
```

When implemented as above, the forward pass for this layer is performed in FP16. The backward pass is done in the same format as the forward pass, so the backward pass is also performed in FP16. As part of the backward pass, the casting of the parameters from the forward pass is reversed, so the parameter update step is performed in FP32.

The problem with this is that this function is *only* an implementation of this layer with FP16 compute and FP32 storage and updates. We would need to separately define a function that does, say, all compute in FP32. Ideally, we would like to be able to separate the definition of the layer from the floating-point format we want to use.

This can be achieved using a "custom getter" as long as the parameters in a layer are created using `tf.get_variable`. This is because it is possible to customise how the `tf.get_variable` function works within the context of a particular `tf.variable_scope`. Whenever `tf.get_variable` is called within a variable scope with a custom getter, the custom getter is called instead, with `tf.get_variable` as the first argument and the arguments given to `tf.get_variable` as the subsequent arguments. This allows for a great deal of control over how the parameters in a model are created and used.

Using a custom getter will not work with models defined using Keras because Keras does not use `tf.get_variable` internally. 

To use this method, we first define a custom getter. As stated above, this will take the function `tf.get_variable` as its first argument, and the inputs to `tf.get_variable` as its subsequent arguments. We can provide `tf.get_variable` as an argument and call it within our custom getter because functions are first-class objects in Python. Here is an example taken from `float32_master_weights.py`:

```python
# FP32 parameter getter
# This function creates FP32 weights no matter what the compute dtype is

def fp32_parameter_getter(getter, name, dtype, trainable, shape=None, *args, **kwargs):

    if trainable and dtype != tf.float32:
        parameter_variable = getter(name, shape, tf.float32, *args, trainable=trainable, **kwargs)
        return tf.cast(parameter_variable, dtype=dtype, name=name + "_cast")

    else:
        parameter_variable = getter(name, shape, dtype, *args, trainable=trainable, **kwargs)
        return parameter_variable
```

For trainable parameters, if the data type of the compute is not FP32, the parameters are created in FP32 and then cast to the data type of the compute before the computations are performed.

We then define the layers of our model using `tf.get_variable` to create the variables for the parameters. For example, here is a general convolution function as defined in `float32_master_weights.py`:

```python
# Define a convolution that uses tf.get_variable to create the kernel
def conv(feature_map, kernel_size, stride, filters_out, padding='SAME', op_name):
    
    # We use NHWC format
    filters_in = feature_map.get_shape().as_list()[-1]
    
    # Resource variables must be used on the IPU
    with tf.variable_scope(op_name, use_resource=True):
        
        kernel = tf.get_variable(
            name="conv2d/kernel",
            shape=[kernel_size, kernel_size, filters_in, filters_out],
            dtype=feature_map.dtype,
            trainable=True
        )
        
        return tf.nn.conv2d(
            x,
            filters=kernel,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format="NHWC",
        )
```

This definition of the layer has no code specific to any data type, so we can use it for training in FP16, FP32, or a mix of the two. Each distinct variable we create using `tf.get_variable` must be given a distinct name, so we provide an `op_name` argument which allows us to do this.

Our variable scope also has the argument `use_resource=True` to explicitly specify that resource variables should be used. We do this because non-resource variables are not supported by the XLA compiler that is used to run TensorFlow programs on the IPU.

We can then define a model using these layers. When we call the model function to apply the forward pass, we do so within a `tf.variable_scope` with our custom getter. We must again explicitly specify that resource variables should be used. Here's part of the training loop in `float32_master_weights.py` as an example:

```python
# Define the body of the training loop, to pass to `ipu.loops.repeat`
def training_loop_body(loss_running_total, x, y):

    # Apply the model function to the inputs
    # Using the chosen variable getter as our custom getter
    with tf.variable_scope('all_vars', use_resource=True, 
                           custom_getter=fp32_parameter_getter):
        logits = model_function(x)
        
    loss = loss_function(logits, labels)
    
    # (Rest of training loop excluded for brevity)
```

When this method is used, each layer in the model will create its variables using our custom getter. We then only need to change the data type of the inputs to switch between training in pure FP32 and training in mixed precision.

Since custom getters change the functionality of a standard TensorFlow function, they can be a source of errors. For example, when no custom getter is used, `tf.get_variable` returns a `tf.Variable` object. However, the `tf.cast` function in our mixed-precision custom getter returns a `tf.Tensor` object. This would cause an error if a user of `tf.get_variable` expected it to return a `tf.Variable` object, and then tried to call a method of the output which `tf.Tensor` objects don't have, such as `assign()`.

A full example of using this method to train a simple convolutional model on the Fashion-MNIST dataset is provided in `float32_master_weights.py`.


## Underflowing gradients and loss scaling

Another numerical issue that can occur when training a model in half-precision is that the gradients can underflow. This can be difficult to debug because the model will simply appear to not be training, and can be especially damaging because any gradients which underflow will propagate a value of 0 backwards to other gradient calculations.

The standard solution to this is known as *loss scaling*. To use loss scaling, multiply the loss by some constant factor before calculating the gradients. Because each gradient is just the derivative of the loss with respect to some parameter, scaling the loss by a constant factor also scales all of the gradients by the same factor.

As a rule of thumb, you should use the highest loss scaling factor that does not cause numerical overflow.

The problem with this is that it makes the parameter update step incorrect, because the resulting gradients have all been scaled up. We must therefore scale the gradients back down by the loss scaling factor after calculating them.  

To do this, we make use of the fact that the `minimize` method of a TensorFlow optimizer is just the composition of its `compute_gradients()` and `apply_gradients()` methods. The `compute_gradients()` method returns a list of `(gradient, variable)` tuples of corresponding gradients and variables and the `apply_gradients()` method takes this list as input and performs the parameter update step. This means we can divide the gradients by the loss scaling factor between these two steps.

This is the method implemented in the training loop in `float32_master_weights.py`, which is shown below:

```python
# Define the body of the training loop, to pass to `ipu.loops.repeat`
def training_loop_body(loss_running_total, x, y):

    # We use the chosen variable getter as our custom getter
    with tf.variable_scope('all_vars', use_resource=True, custom_getter=chosen_getter):
        logits = model_function(x)

    logits = tf.cast(logits, tf.float32)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    # When using Adam in FP16, you should check
    #     the default value of epsilon and ensure
    #     that it does not underflow
    optimizer = tf.train.AdamOptimizer(0.01, epsilon=1e-4)
    
    # Scale loss
    loss *= LOSS_SCALING_FACTOR
    
    # Calculate gradients with scaled loss
    grads_and_vars = optimizer.compute_gradients(loss=loss)
    
    # Rescale gradients to correct values
    grads_and_vars = [(gradient/LOSS_SCALING_FACTOR, variable)
                      for gradient, variable in grads_and_vars]
    
    # Apply gradients
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

    # Return loss to original value before reporting it
    loss /= LOSS_SCALING_FACTOR
    
    return([loss_running_total + loss, train_op])
```

If you are doing all computations in FP16, gradients may underflow after they have been divided by the learning rate. Dividing gradients by the learning rate can only be done completely safely when the parameters are stored and updated in FP32, in which case the gradients are cast to FP32 before the parameter update.

If the relationship between the gradients and the size of the parameter update step is linear, as in stochastic gradient descent (with or without momentum), we can instead choose to divide the learning rate by the loss scaling factor to make the parameter update step correct. This is demonstrated in the training loop of the `stochastic_rounding.py` example included with this tutorial:

```python
def training_loop_body(loss_running_total, x, y):

    logits = model(x, training=True)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    # Apply loss scaling
    loss *= LOSS_SCALING_FACTOR

    # Divide learning rate by loss scaling factor
    #     so the parameter update step is correct
    optimizer = tf.train.MomentumOptimizer(LEARNING_RATE/LOSS_SCALING_FACTOR, momentum=0.9)

    train_op = optimizer.minimize(loss=loss)

    # Return loss to original value before reporting it
    loss /= LOSS_SCALING_FACTOR

    return([loss_running_total + loss, train_op])
```

This saves some compute, but can lead to some problems, such as causing the learning rate to underflow in FP16.


## Diagnosing numerical issues

If your model is not performing as you would expect, you may find it useful to inspect the outputs of some of the intermediate calculations. This can be done either by using `ipu.internal_ops.print_tensor` to print the value of a tensor or by using an outfeed queue. Please refer to the API reference for details of how to [print tensors](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/api.html#tensorflow.python.ipu.internal_ops.print_tensor) and how to use an [outfeed queue](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/api.html#tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue).

You can also configure how the floating-point unit should respond to floating-point exceptions. This can be done using the function `ipu.utils.set_floating_point_behaviour_options`. Setting `nanoo=True` will enable "NaN on overflow" mode, in which case operations that overflow return `NaN`. If you set `nanoo=False`, overflowing operations will "saturate", which means that they will return the highest representable number in FP16, which may give unusual results. Setting `oflo`, `div0`, and `inv` to `True` will cause the floating-point unit to raise exceptions and stop execution on overflows, divisions by 0, and invalid operations, respectively. For example:

```python
# Configure device with 1 IPU and compile

ipu_configuration = ipu.utils.create_ipu_config()

ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration, num_ipus=1)

# Enable floating-point exceptions and disable stochastic rounding
ipu_configuration = ipu.utils.set_floating_point_behaviour_options(opts=opts,
                                                                   nanoo=True,
                                                                   oflo=True, div0=True, inv=True,
                                                                   esr=False)
ipu.utils.configure_ipu_system(config=ipu_configuration)
```
When you configure floating-point exceptions, you should also explicitly enable or disable stochastic rounding by setting `esr` to `True` or `False` respectively. A more complete description of when these exceptions are raised can be found in the [Poplar documentation](https://docs.graphcore.ai/projects/poplar-api/en/latest/poplar_api.html#_CPPv4N6poplar22FloatingPointBehaviourE).

When the floating-point unit raises an exception, a large amount of information about the internal state of the IPU at the moment the exception occurred is printed before the program exits. The important piece of information to look for is the contents of the floating-point status register `$FP_STS`, which has three fields `OFLO`, `DIV0` and `INV` corresponding to overflows, divisions by 0 and invalid operations respectively. A value of `0x1` in a field indicates that that particular exception was raised, while a value of `0x0` indicates that that exception was not raised.


## Other considerations

The IPU performs convolutions internally by performing many multiply-accumulate operations. For computations in FP16, you can configure what data type is used for intermediate calculations in convolutions using `ipu.utils.set_convolution_options`. This function takes an IPU configuration `opts` and a dictionary of options `convolution_options` and returns the IPU configuration with the convolution options applied.

If the value for the `partialsType` key is set to `half` in the options dictionary, intermediate calculations in convolutions will be performed in FP16. This can improve throughput for some models without affecting accuracy. The function `ipu.utils.set_matmul_options` works similarly for matrix multiplications.

For example, from `stochastic_rounding.py`:

```python
if args.use_float16_partials:
    
    ipu_configuration = ipu.utils.set_matmul_options(
        opts=ipu_configuration, 
        matmul_options={'partialsType': 'half'})
    
    ipu_configuration = ipu.utils.set_convolution_options(
        opts=ipu_configuration,
        convolution_options={'partialsType': 'half'})
```

The full list of options for matrix multiplications is documented [here](https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs_api.html#_CPPv4N6poplin6matMulERN6poplar5GraphERKN6poplar6TensorERKN6poplar6TensorERN6poplar7program8SequenceERKN6poplar4TypeERKN6poplar12DebugContextERKN6poplar11OptionFlagsEPN6matmul13PlanningCacheE) and the full list of options for convolutions is documented [here](https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs_api.html#_CPPv4N6poplin13createWeightsERN6poplar5GraphERK10ConvParamsRKN6poplar12DebugContextERKN6poplar11OptionFlagsEP13PlanningCache). However, configuring options other than the partials type is beyond the scope of this tutorial.

The smallest positive number representable in half-precision is approximately `6.0e-08`, which means that any number smaller than half this number (approximately `3.0e-08`) will underflow. It is worth confirming that, for example, the learning rate and any optimiser parameters do not underflow. For example, the default value of `epsilon` in some implementations of the Adam optimiser is `1e-8`, which will underflow in half-precision and potentially cause numerical issues. 

Some operations are especially prone to overflow in FP16. Examples of these include exponentials, squaring, cubing, taking sums of many values, and taking sums of squares. Any such calculations should ideally be performed in FP32. Some optimisers, such as RMSProp and Adam, use large sums of squares as part of their computations and are therefore especially prone to instability in FP16.

There are some operations which are common in neural networks, such as group/layer normalisation and softmax, which contain sub-steps which are especially prone to numerical errors. However, these sub-steps are actually implemented in FP32 internally in the Graphcore port of TensorFlow, so you can freely use these operations in FP16 at the TensorFlow level.

It is strongly recommended that you use the versions of operations specialised for the IPU where available, especially those containing potentially unstable sub-steps. See the sections of the API reference on [specialised Keras layers](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/api.html#keras-layers) and [operators](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/api.html#operators) for information on what operations have versions specialised for the IPU.

Sometimes, overflows can occur in the calculation of the variance in a normalisation layer. To avoid this, the mean of the values to be normalised can be subtracted from each value before the variance is calculated. To do this in all normalisation layers, set `use_stable_statistics` to `True` in `ipu.utils.set_norm_options`. For example:

```python
ipu_configuration = ipu.utils.set_norm_options(opts=ipu_configuration,
                                               use_stable_statistics=True)
```

There is empirical evidence that stochastic gradient descent with momentum is more effective than plain SGD when training in FP16 with stochastic rounding. If you are unsure what optimizer you should use when training in FP16 with stochastic rounding, you should try using stochastic gradient descent with momentum.

You should always make sure the inputs to your network are normalised. Non-normalised inputs have been known to cause numerical issues.


# Code examples

Two code examples are provided to demonstrate the programming techniques discussed in this tutorial: `stochastic_rounding.py` and `float32_master_weights.py`. These demonstrate different approaches to training a model in FP16. Both code examples train a convolutional model to classify images from a standard dataset, reporting the average loss for each epoch. Both examples are based on `example_2.py` from Part 2 of the TensorFlow 1 introductory tutorial.

The datasets are downloaded using the TensorFlow API. See the TensorFlow API documentation for the details of the license of the [Fashion-MNIST](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/fashion_mnist/load_data) dataset. The CIFAR-10 dataset was introduced in [this technical report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), which was published in 2009 by Alex Krizhevsky.

## Running the examples

To run the code examples, you will need access to IPU hardware and the latest Poplar SDK. You will need to run the examples in a virtual environment with the Graphcore port of TensorFlow 1 installed. For software installation and setup details, please see the Getting Started guide for your hardware setup, available [here](https://docs.graphcore.ai/en/latest/hardware.html#getting-started). 


## Stochastic rounding example

In `stochastic_rounding.py`, we train a ResNet model of a given depth on the CIFAR-10 image classification dataset using stochastic rounding. The program can be run at the command line with the command
```
python stochastic_rounding.py precision depth 
```
where:

- `precision` is one of `float16` or `float32`, specifying the precision to be used.
- `depth` is a valid depth for a ResNet model training on CIFAR-10, that is, a number of the form 6N+2 for a whole number N. For example, 8, 14, 20, and 26 are all valid depths (with N = 1, 2, 3, 4 respectively). The model only trains on one IPU, so a model with too many layers may run out of memory.

The program also takes a number of optional command line arguments, documented at the end of this section.

For example, to train ResNet-14 in FP16, you would use this command:
```
python stochastic_rounding.py float16 14
```

The source code for this example is available [here](stochastic_rounding.py). The ResNet CNN architecture was introduced by Kaiming He et al. in a 2015 research paper, titled [_Deep Residual Learning for Image Recognition_](https://arxiv.org/abs/1512.03385).


## FP32 master weights example

In `float32_master_weights.py`, we train a simple convolutional model on the Fashion-MNIST dataset. The parameters are stored and updated in FP32, regardless of what precision is used for the compute. This is done using the "custom getter" method described above.

The program can be run at the command line with the command
```
python float32_master_weights.py precision
```
where:

- `precision` is one of `mixed` or `float32`, specifying the precision to be used. With `mixed` as the precision, the forward and backward pass are done in FP16, while the parameters will be stored and updated in FP32. With `float32` as the precision, all compute is done in FP32.

The program also takes a number of optional command line arguments, documented at the end of this section.

For example, to run the model as in Method 2, you would use this command:
```
python float32_master_weights.py mixed
```

The source code for this example is available [here](float32_master_weights.py).


## Optional command line arguments

Both code examples also accept the following optional command line arguments:

- `--batch-size`: Integer specifying the batch size. Defaults to 32.

- `--epochs`: Integer specifying the number of epochs to profile. Defaults to 5.

- `--loss-scaling-factor`: Float specifying the loss-scaling factor to use. Defaults to 2^8 = 256.

- `--learning-rate`: Float specifying the learning rate for the optimiser to use. Defaults to 0.01.

- `--use-float16-partials`: Use FP16 instead of FP32 for values of partial sums when calculating matrix multiplications and convolutions.

For example, to use `stochastic_rounding.py` to train ResNet-14 in FP16 for 10 epochs with a batch size of 64, you would use this command:

```
python stochastic_rounding.py float16 14 --batch-size 64 --epochs 10
```

## Other examples

Further examples of how FP16 can be used on the IPU are available in our [examples repository](https://github.com/graphcore/examples) on GitHub, including our TensorFlow 1 convolutional network applications for [training](../../../applications/tensorflow/cnns/training) and [inference](../../../applications/tensorflow/cnns/inference).


