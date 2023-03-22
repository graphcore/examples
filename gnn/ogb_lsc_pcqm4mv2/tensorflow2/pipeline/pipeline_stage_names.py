# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from keras.layers.core import Lambda, SlicingOpLambda, TFOpLambda

from model.encoders.base_encoder import BaseEncoder, ConcatFeatures
from model.gnn.layers import DecoderLayer, InteractionNetworkLayer
from model.hybrid.layers import GPSLayer
from model.encoders.base_encoder import ConcatFeatures

# List of layers classes that will be assigned to the same pipeline stage
# as the previous layer.
PIPELINE_ALLOCATE_PREVIOUS = (TFOpLambda, SlicingOpLambda, Lambda, ConcatFeatures)

# Dictionary of user friendly names matching layer identifiers.
PIPELINE_NAMES = {"enc": [BaseEncoder], "hid": [InteractionNetworkLayer], "dec": [DecoderLayer], "gps": [GPSLayer]}
