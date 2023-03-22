# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import tensorflow as tf
from tensorflow.python import ipu
from keras.layers import Dense, LayerNormalization
from keras.layers.core import TFOpLambda

from utilities.pipeline_stage_assignment import PipelineStagesAssigner


def get_pipeline_assignments(pipeline_stages):
    pipeline_allocate_previous = (TFOpLambda,)
    pipeline_names = {"hid": [Dense], "layer_norm": [LayerNormalization]}

    input_layer = tf.keras.Input(shape=1)
    kernel_initializer = tf.keras.initializers.Constant(1)

    x = input_layer
    x = x * x
    for _ in range(5):
        x = Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(x)
    x = x * x
    x = LayerNormalization()(x)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        model = tf.keras.Model(input_layer, x)

        pipeline_assigner = PipelineStagesAssigner(pipeline_allocate_previous, pipeline_names)
        assignments = model.get_pipeline_stage_assignment()
        assignments = pipeline_assigner.assign_pipeline_stages(assignments, pipeline_stages)

        model.set_pipeline_stage_assignment(assignments)
        model.print_pipeline_stage_assignment_summary()
    return assignments


@pytest.mark.parametrize(
    "pipeline_stages, expected_stages",
    [
        ([["hid", "hid", "hid"], ["hid", "hid", "layer_norm"]], [0, 0, 0, 0, 1, 1, 1, 1]),
        ([["hid", "hid", "hid", "hid", "hid", "layer_norm"]], [0, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_pipeline_stage_assigner(pipeline_stages, expected_stages):
    assignments = get_pipeline_assignments(pipeline_stages)
    assert [a.pipeline_stage for a in assignments] == expected_stages


@pytest.mark.parametrize(
    "pipeline_stages",
    [[["hid", "hid", "hid"], ["hid", "layer_norm"]], [["hid", "hid", "hid"], ["hid", "hid", "hid", "layer_norm"]]],
)
def test_pipeline_stage_assigner_requested_too_few_stages(pipeline_stages):
    with pytest.raises(Exception):
        get_pipeline_assignments(pipeline_stages)
