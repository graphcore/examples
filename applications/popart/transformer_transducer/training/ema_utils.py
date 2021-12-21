# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import popart
import onnx

import logging_util

# set up logging
logger = logging_util.get_basic_logger('EMA_UTILS')

network_names = ["transcription_network", "prediction_network", "joint_network"]
EMA_PREFIX = "exp_mov_avg_"


def create_exp_mov_avg_weights(builder, model_weight_names, ema_factor):
    """ creates variable tensors for exponential moving averages of model weights """

    ema_weight_names = dict()

    for nname in network_names:
        ema_weight_names[nname] = []
        for uname, weight_tensor in model_weight_names[nname]:
            # uname is a user-given name
            # weight_tensor is the tensor name given by PopART
            # The ExpMovAvg op will create new weights tensors with ID "exp_mov_avg_" + weight_tensor
            # These new weight tensors will hold the data for exponential moving averages
            exp_avg_weight_tensor = builder.customOp(
                opName="ExpMovAvg",
                opVersion=1,
                domain="com.acme",
                inputs=[weight_tensor],
                attributes={"ema_factor": ema_factor},
                numOutputs=1,
            )[0]
            ema_weight_names[nname].append((EMA_PREFIX + uname, EMA_PREFIX + weight_tensor))
            logger.info("Created EMA tensor {} for {}".format(exp_avg_weight_tensor, EMA_PREFIX + weight_tensor))

    return ema_weight_names


def set_ema_weights_offchip(session_options, ema_weight_names):
    """ Sets the tensor locations of EMA weights to be off-chip """

    tensor_location_override_dict = dict()
    for nname in network_names:
        for _, ema_wname in ema_weight_names[nname]:
            tensor_location_override_dict[ema_wname] = popart.TensorLocation(popart.TensorStorage.OffChip)
            logger.info("Setting tensor-location for {} to be OffChip".format(ema_wname))
    session_options.tensorLocationSettingsOverride = tensor_location_override_dict

    return


def transfer_ema_weights(source_model_onnx_fp, dest_model_onnx_fp):
    """ copies the exp averaged weight values to original weight tensors """

    model_proto = onnx.load(source_model_onnx_fp)

    ema_weights = dict()

    # first get all ema weights
    for weight in model_proto.graph.initializer:
        if weight.name.startswith(EMA_PREFIX):
            original_wname = weight.name.replace(EMA_PREFIX, '')
            ema_weights[original_wname] = weight

    # now copy ema weights to original weight tensors
    for weight in model_proto.graph.initializer:
        if weight.name in ema_weights:
            assert(len(weight.int32_data) == len(ema_weights[weight.name].int32_data))
            for idx, val in enumerate(ema_weights[weight.name].int32_data):
                weight.int32_data[idx] = val

    # overwrite original .onnx file with ema weights transferred to original weight tensors
    onnx.save(model_proto, dest_model_onnx_fp)

    return
