# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.gradient_accumulation import GradientAccumulationReductionMethod

from keras_extensions.optimization import get_optimizer


def get_model_with_grad_accum(grad_accum):
    input_layer = tf.keras.Input(shape=1)
    kernel_initializer = tf.keras.initializers.Constant(1)

    x = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(input_layer)
    y = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(x)
    model = tf.keras.Model(input_layer, y)

    model.set_gradient_accumulation_options(
        gradient_accumulation_steps_per_replica=grad_accum,
        gradient_accumulation_reduction_method=GradientAccumulationReductionMethod.RUNNING_MEAN,
    )
    return model


def get_pipelined_model(grad_accum):
    input_layer = tf.keras.Input(shape=1)
    kernel_initializer = tf.keras.initializers.Constant(1)

    with tf.keras.ipu.PipelineStage(0):
        x = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(input_layer)
    with tf.keras.ipu.PipelineStage(1):
        y = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(x)
    model = tf.keras.Model(input_layer, y)

    model.print_pipeline_stage_assignment_summary()
    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=grad_accum,
        gradient_accumulation_reduction_method=GradientAccumulationReductionMethod.RUNNING_MEAN,
    )
    return model


def ipu_prog(
    num_elements,
    num_replicas,
    gradient_accumulation,
    loss_scaling,
    pipeline,
):
    micro_batch_size = int(num_elements / gradient_accumulation / num_replicas)

    ds = tf.data.Dataset.from_tensor_slices(([1.0] * num_elements, [2.0] * num_elements))
    ds = ds.batch(micro_batch_size, drop_remainder=True)

    num_micro_batches = len(ds)

    cfg = ipu.config.IPUConfig()
    pipeline_stages = 2 if pipeline else 1
    cfg.auto_select_ipus = num_replicas * pipeline_stages
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():

        if pipeline:
            model = get_pipelined_model(gradient_accumulation)
        else:
            model = get_model_with_grad_accum(gradient_accumulation)

        model.build(input_shape=(micro_batch_size, 1))

        optimizer = get_optimizer(
            "AdamW",
            num_replicas,
            1.0,
            loss_scaling=loss_scaling,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-6,
            weight_decay_rate=0.0,
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MSE,
            metrics=[tf.keras.losses.MSE],
            steps_per_execution=num_micro_batches,
        )
        model.fit(ds, steps_per_epoch=num_micro_batches)

        return model.get_weights()[0][0][0]


class TestGradientNormalization:
    @pytest.mark.parametrize("replicas", [1, 2])
    @pytest.mark.parametrize("grad_accum", [4, 8])
    @pytest.mark.parametrize("loss_scaling", [1, 2])
    @pytest.mark.parametrize("pipeline", [True, False])
    def test_scaling_in_optimizer(self, replicas, grad_accum, loss_scaling, pipeline):
        weight = ipu_prog(
            num_elements=16,
            num_replicas=replicas,
            gradient_accumulation=grad_accum,
            loss_scaling=loss_scaling,
            pipeline=pipeline,
        )
        np.testing.assert_allclose(weight, 1.9999995)


class TestLossScaling:
    @pytest.mark.parametrize("loss_scaling", [None, 1, 32768])
    def test_loss_scaling(self, loss_scaling):
        weight = ipu_prog(
            num_elements=8, num_replicas=1, gradient_accumulation=2, loss_scaling=loss_scaling, pipeline=False
        )
        np.testing.assert_allclose(weight, 1.9999995)

    def test_loss_scaling_can_overflow_grads(self):
        # Try to overflow the gradients
        weight = ipu_prog(num_elements=8, num_replicas=1, gradient_accumulation=2, loss_scaling=4e38, pipeline=False)
        assert np.isnan(weight)

    def test_loss_scaling_can_underflow_grads(self):
        # Try to underflow the gradients
        weight = ipu_prog(num_elements=8, num_replicas=1, gradient_accumulation=2, loss_scaling=2e-38, pipeline=False)
        np.testing.assert_allclose(weight, 1.0)
