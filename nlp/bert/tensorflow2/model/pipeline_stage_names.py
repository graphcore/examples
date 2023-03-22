# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from keras.layers.core import SlicingOpLambda, TFOpLambda
from transformers.models.bert.modeling_tf_bert import (
    TFBertEmbeddings,
    TFBertLayer,
    TFBertMLMHead,
    TFBertNSPHead,
    TFBertPooler,
)

from model.ipu_embeddings_layer import IpuTFBertEmbeddings
from model.ipu_pretraining_model import GatherSubsetOutput

# List of layers classes that will be assigned to the same pipeline stage as the previous layer (or zero if the layer
# being assigned is the first one). This list contains layers build from tf.ops, typically when transforming a Keras
# subclass model into a functional one. Hence, the user is not expected and cannot assign these layers manually.
PIPELINE_ALLOCATE_PREVIOUS = (TFOpLambda, SlicingOpLambda)

# Dictionary of user friendly names matching layer identifiers, which can be the layer name or its class. The layers
# will be assigned following the list of lists of entries that match some of the keys of this dictionary.
PIPELINE_NAMES = {
    "emb": [TFBertEmbeddings, IpuTFBertEmbeddings],
    "hid": [TFBertLayer],
    "pool": [TFBertPooler],
    "enc_out": [GatherSubsetOutput],
    "heads": [TFBertNSPHead, TFBertMLMHead],
    "qa_head": ["qa_outputs", "start_positions", "end_positions"],
    "glue_head": ["labels"],
}
