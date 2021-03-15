# Optimising for the IPU: Computational Graph Recompilation and Executable Switching in TensorFlow

### Introduction

When code is executed on an IPU, a multi-operation computation graph is compiled
to run efficiently on the device.

This compilation ensures that the code running on the IPU is optimal: as many
tiles as possible are used, as little device memory as possible is used and the
number of execution cycles is short. Note that, in contrast to some other
platforms, the graph to be compiled isn't just one matmul operation but many
consecutive operations and so almost every graph execution is different and will
need to be compiled and optimised.

The compilation process performs many optimizations, and so it can take some
time. It is therefore important to know *when* the compilation of the graph will
happen and avoid it occurring at inconvenient times, or too often. This is
especially relevant when running benchmarks since it can add significant
overhead.

As a result, it is important to avoid recompilations as far as possible.
This technical note provides some strategies that can
help you with this.

### Consideration 0: Avoiding recompilation


To avoid recompiling the same code every time a TensorFlow process is started,
you can turn on caching of the executable. Each generated file is identified by a
64-bit hash value.

Caching is enabled by setting the option ``--executable_cache_path`` to a
directory where the compiled files will be stored. For example:

```
    export TF_POPLAR_FLAGS="--executable_cache_path=/mnt/data/USERNAME/ipu_cache/"
```
See [Caching of compiled executables](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/compiling.html)
in the TensorFlow user guide for more information.

However, there are several other cases that can still cause recompilation even
with the cache being active. To detect recompilation, set the logging as shown:

```
    export POPLAR_LOG_LEVEL=INFO
```

Then look for "Starting phase: graphConstruction". A related issue is to
also look for repeated "Loading executables" messages. For example, this log
message:

```
    Ending phase: Loading executable (duration 607 ms; diff RSS: 0.586426 MB)
```

These messages might occur in large numbers at the beginning but should not
occur after a warm-up phase. As you can see in the example log message above,
they can take a significant amount of time if they occur too frequently during
a run (more than 500ms each time).

Note that at the beginning for
initialisation, and at the end for final results there might be some executable
messages in the log. Those should not cause any problems. **You should avoid
executable loading and compilation at any cost when running benchmarks.**


### Consideration 1: Sessions


Don't use different sessions on the same device.

This advice is also relevant
when you are working with threading. Either use a single session or distribute
the sessions to different devices.



### Consideration 2: Computational graphs


If different processes run workloads on the same IPU (apart from initialisation
at the beginning) make sure that they run the same graph. Otherwise executables
will get loaded repeatedly to the IPU. This will be visible in
your logging, together with the respective time measurements.

If you have different
workloads, try to either put them together into one graph or distribute the
graphs onto different IPUs.

Relying solely on the ``with ipu_scope("/device:IPU:0"):`` statement has a high
chance of creating different computational graphs. A better approach is to
combine the whole computation into one graph and apply the
[ipu.ipu_compiler.compile](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/api.html#tensorflow.python.ipu.ipu_compiler.compile)
method as described in the
[model training documentation.](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/perf_training.html#training-loops-data-sets-and-feed-queues)
Alternatively, higher level APIs can be used such as
[estimators](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#module-tensorflow.python.ipu.ipu_estimator)
or, in TensorFlow 2, the combination of ``tf.function`` and the
[IPUStrategy](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#tensorflow.python.ipu.ipu_strategy.IPUStrategy).

For further details on how to wrap computational graphs, see the
[Graphcore TensorFlow documentation.](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/index.html)

### Consideration 3: Batch size


Calculate the correct batch size and keep it constant.

Each change in batch
size causes a new graph construction and executable. The reason is that the
computational graph is static and before compilation the batch size is
undefined. The execution instructions loaded on the device depend directly on
this as we are going to loop multiple times over data and and need to know the
number of repetitions in advance. Furthermore, with a different batch size, a
different distribution of the processing onto the tiles will be required in
order to benefit from the synergies of larger batch sizes and to obtain high
efficiency.

### Consideration 4: Weights


Keep weights and graphs separate if the weights are subject to change.

Graph freezing causes recompilation and slows
down the processing if the weights change. The reasons for this are the same as
discussed above for Consideration 2.

### Consideration 5: Constants


In addition to the weights you might also have other parameters. These
parameters should be handled as ``tf.constant`` or ``tf.placeholder`` (in
TensorFlow 1) if possible, otherwise changing them will always result in
different graphs. Note that this is not possible with all parameters - for
some, like batch size, a change will result in a different computational graph
and will require recompilation. On the other hand, parameters such as limits in
while loops can be handled as constants.

The advantage with this method is that
the graph gets compiled more generically and then the respective variables get
loaded into the executable without a recompilation. However, if you change the
parameters within a program run, you will still see different executables being
loaded.

Similar to weights in frozen graphs for inference, there are some cases
where computation is more efficient if variables are not treated as
``tf.constant`` but as normal constant parameters (e.g., float or int)
directly woven into the computational graph.
This requires some experimentation to explore case by case.

### Consideration 6: Deep dive


If none of these approaches apply to your problem or your program is too
complex to spot the source, a last resort in TensorFlow is to compare XLA dump
text files (``*.txt``) by setting the
[xla_dump](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/debug_options_flags.cc#L444)
variable to the respective folder.

Make sure that you get XLA dumps of the different executions that should have
the same executable but cause recompilation. Check that they don't get
overwritten and filter out the largest ones for the comparison - `diffmerge`
can help you visualise any differences.
Note that the XLA settings have to be made before the first TensorFlow object is
created and cannot be changed afterwards.
Some graphs will only be generated if a compilation is triggered.

Usually there will be a clear
difference in the patterns, such as a variable that has different values
between the two variants.

If there is no clear difference then the wrong files
may have been chosen for comparison.

If you have frozen weights as constants in the graph and those are the only thing
differing between the executions, this approach might not help because only low
dimensional weights get displayed. Also, larger arrays of constants might cause
issues. Other variables are usually well supported.

### Code examples


We provide two code examples.
This example code addresses the different considerations.
The [first example](recompilation.py)
is written for TensorFlow 2 as well as TensorFlow 1.

The [second example](TF2_recompilation.py)
is specifically tailored to the TensorFlow 2 API
but follows the same principles as the first example.

In the TensorFlow 2 API, ``tf.constant`` is used instead of ``tf.placeholder``.
Also, when working with strategies instead of sessions,
having multiple strategy scopes active in parallel is not supported by
TensorFlow 2 and Consideration 1 does not apply.