# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.python import ipu
from CentralCropIpu import CentralCropIpu


def crop(inputs, output_size, nb_ipus_per_replica):
    """Perform a central crop"""
    if nb_ipus_per_replica > 1:
        # Used in pipelined model on multiple IPUs to reduce the activation size to transfer between IPUs
        cropped = CentralCropIpu(output_size)(inputs)
    else:
        factor = output_size / inputs.shape[1]
        cropped = tf.image.central_crop(inputs, factor)
    return cropped


def double_conv_layer(filter_size, x, dtype, name):
    x = Conv2D(filter_size, 3, activation="relu", kernel_initializer="he_normal", dtype=dtype, name=name + "_conv_0")(x)
    x = Conv2D(filter_size, 3, activation="relu", kernel_initializer="he_normal", dtype=dtype, name=name + "_conv_1")(x)
    return x


def up_conv(filter_size, x, name):
    x = Conv2DTranspose(
        filters=filter_size,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu,
        name=name + "_transpose",
    )(x)
    return x


def set_pipeline_stages(model):
    assignments = model.get_pipeline_stage_assignment()
    stage_id = 0
    # Iterate over default pipeline stage assignments and set their pipeline stages.
    for assignment in assignments:
        assignment.pipeline_stage = stage_id
        # Split the model on the `encoder_block_1_maxpooling` layer.
        if assignment.layer.name.startswith("encoder_block_1_maxpooling"):
            stage_id = 1
        # Split the model on the `decoder_block_0_transpose` layer.
        elif assignment.layer.name.startswith("decoder_block_0_transpose"):
            stage_id = 2
        # Split the model on the `decoder_block_2_transpose` layer.
        elif assignment.layer.name.startswith("decoder_block_2_transpose"):
            stage_id = 3

        # Crop first then send through pipeline
        elif assignment.layer.name in ["central_crop_ipu", "central_crop_ipu_1"]:
            assignment.pipeline_stage = 0
        elif assignment.layer.name == "central_crop_ipu_2":
            assignment.pipeline_stage = 1
    # Set the assignments to the model.
    model.set_pipeline_stage_assignment(assignments)


def get_pipeline_stage_options(available_memory_proportion, nb_stages):
    len_amp = len(available_memory_proportion)
    if len_amp == 1:
        return None
    elif len_amp != nb_stages:
        raise ValueError(
            f"The number of available memory proportion values ({len_amp}) needs to be the same as the number of pipeline stages ({nb_stages})"
        )
    else:
        options = []
        for amp in available_memory_proportion:
            options.append(
                ipu.pipelining_ops.PipelineStageOptions(
                    {"availableMemoryProportion": str(amp)}, {"availableMemoryProportion": str(amp)}
                )
            )
        return options


def get_pipeline_scheduler(pipeline_scheduler):
    if pipeline_scheduler == "interleaved":
        return ipu.pipelining_ops.PipelineSchedule.Interleaved
    else:
        return ipu.pipelining_ops.PipelineSchedule.Grouped


def set_pipeline_options(model, args):
    set_pipeline_stages(model)
    # We assume one stage per IPU in this example
    options = get_pipeline_stage_options(args.available_memory_proportion, nb_stages=args.nb_ipus_per_replica)
    pipeline_scheduler = get_pipeline_scheduler(args.pipeline_scheduler)
    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=args.gradient_accumulation_count,
        recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode.Auto,
        pipeline_schedule=pipeline_scheduler,
        forward_propagation_stages_poplar_options=options,
        backward_propagation_stages_poplar_options=options,
        offload_weight_update_variables=False,
    )
