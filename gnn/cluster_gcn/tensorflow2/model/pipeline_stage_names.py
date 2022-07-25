# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from keras.layers.core import SlicingOpLambda, TFOpLambda

from model.adjacency_processing import AdjacencyProcessing
from model.model import GcnLayer

# List of layers classes that will be assigned to the same pipeline stage
# as the previous layer (or zero if the layer being assigned is the first
# one). This list contains layers from tf.ops, which are added when
# creating a model, as such the user is not expected and cannot assign
# these layers manually.
PIPELINE_ALLOCATE_PREVIOUS = (TFOpLambda, SlicingOpLambda)

# Dictionary of user friendly names matching layer identifiers, which
# can be the layer name or its class. The layers will be assigned
# following the list of lists of entries that match some of the keys
# of this dictionary.
PIPELINE_NAMES = {
    "adj_proc": [AdjacencyProcessing],
    "hid": [GcnLayer],
}
