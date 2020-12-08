# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""Tutorial code to play around with graph recompilation and executable loading

Parameters to play around with are CACHING, NOMULTISESSION, PLACEHOLDER,
and SAMEBATCH. Some comments in the document refer to the underlying tutorial
in the documentation portal.

The code will print out what the expected behaviour should look like.
"""

import os
import numpy as np

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope


# Consideration 0: Environment setup
CACHING = True  # Cache compiled graph. The folder is tmp_tutorial.
# Consideration 1: Sessions
# Suppressed by strategy scoping
# Consideration 2, 4, 5: Graphs, Weights, Constants
# Use a placeholder that is handed over to the graph instead of a hard coded
# hyperparameter that might change between executions.
PLACEHOLDER = True
# Consideration 3: Batch size
SAMEBATCH = True  # Change the batch size between executions.

# Consideration 0: Environment setup
if "TF_POPLAR_FLAGS" in os.environ and not CACHING:
    os.environ["TF_POPLAR_FLAGS"] = ""
else:
    os.environ["TF_POPLAR_FLAGS"] = "--executable_cache_path=tmp_tutorial"
if "POPLAR_LOG_LEVEL" not in os.environ or \
        os.environ["POPLAR_LOG_LEVEL"] != "INFO":
    print("Setting POPLAR_LOG_LEVEL to INFO for graph compilation information.")
    os.environ["POPLAR_LOG_LEVEL"] = "INFO"

# Consideration 6
os.environ["XLA_FLAGS"] = "--xla_dump_to=tmp_xla_{} ".format(
    np.random.randint(2, 101))
os.environ["XLA_FLAGS"] += " --xla_dump_hlo_pass_re=forward-allocation "
os.environ["XLA_FLAGS"] += " --xla_hlo_graph_sharding_color "
os.environ["XLA_FLAGS"] += " --xla_dump_hlo_as_text "

# Configure arguments for targeting the IPU
cfg = ipu.utils.create_ipu_config()
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

with tf.device("cpu"):
    pa = tf.constant([[1., 1.]], dtype=tf.float32, name="a")
    pb = tf.constant([[0., 1.]], dtype=tf.float32, name="b")
    pc = tf.constant([[1., 5.]], dtype=tf.float32, name="c")

if PLACEHOLDER:
    mult = tf.constant(0.5, dtype=tf.float32, name="multiplier")
else:
    mult = np.random.uniform(0, 1)


@tf.function(experimental_compile=True)
def basic_graph(pa, pb, pc, mult=mult):
    # Do basic addition with tensors
    o1 = pa + pb
    o2 = pa + pc
    simple_graph_output = mult * (o1 + o2)
    return simple_graph_output


strategy = ipu.ipu_strategy.IPUStrategy()

print("\nWarm up & Caching Test: ")
print("No compilation after first execution expected but executable load. \n")
with strategy.scope():
    # Run the graph through the strategy feeding it an arbitrary dictionary
    if not PLACEHOLDER:
        mult = 10
    else:
        mult = tf.constant(10, dtype=tf.float32, name="multiplier")

    result0 = strategy.experimental_run_v2(
        basic_graph,
        [pa, pb, pc, mult])

# Consideration 2, 4, 5: Graphs, Weights, Constants
m = np.random.uniform(0, 1)
if PLACEHOLDER:
    mult = tf.constant(m, dtype=tf.float32, name="multiplier")
else:
    mult = m

with strategy.scope():
    print("\nPlaceholder test. ")
    print("No recompilation or executable switch should occur.\n")
    # Run the graph through the session feeding it an arbitrary dictionary
    result1 = strategy.experimental_run_v2(
        basic_graph,
        [pa, pb, pc, mult])

    # Consideration 3: Batch size
    if SAMEBATCH:
        bs = 1
    else:
        bs = np.random.randint(2, 101)
    print("\nBatch Size Test with batch size %d." % bs)
    print("No recompilation or executable switch should occur.")
    print("Batch size should be the original value of 1.\n")

    pa = tf.constant([[1., 1.]] * bs, dtype=tf.float32, name="a")
    pb = tf.constant([[0., 1.]] * bs, dtype=tf.float32, name="b")
    pc = tf.constant([[1., 5.]] * bs, dtype=tf.float32, name="c")

    result3 = strategy.experimental_run_v2(
        basic_graph,
        [pa, pb, pc, mult])

    print("\nFirst two results should be different (different multiplier).\n")
    print("Caching/warm up test:\t", result0)
    print()
    print("Placeholder test:    \t", result1)
    print()
    if bs > 1:
        print("Batch size test:     \t", result3[:2], "...")
    else:
        print("Batch size test:     \t", result3)
