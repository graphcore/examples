# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import ctypes
import argparse
import numpy as np
import popart


def create_model_and_dataflow_for_training(builder, conf, input, ema_factor):
    """ builds the conformer model, loss function and dataflow for training """

    weights1 = builder.addInitializedInputTensor(
        np.ones([conf.input_dim, conf.output_dim], np.float32), "weights1")
    weights2 = builder.addInitializedInputTensor(
        np.ones([conf.input_dim, conf.output_dim], np.float32), "weights2")

    exp_avg_weights1 = builder.addInitializedInputTensor(
        np.ones([conf.input_dim, conf.output_dim], np.float32), "ema_weights1")
    exp_avg_weights2 = builder.addInitializedInputTensor(
        np.ones([conf.input_dim, conf.output_dim], np.float32), "ema_weights2")

    # The ExpMovAvg op will create new weights tensors with ID prefix "exp_mov_avg_"..
    # These new weight tensors will hold the data for exponential moving
    # averages
    builder.customOp(
        opName="ExpMovAvg",
        opVersion=1,
        domain="com.acme",
        inputs=[weights1, exp_avg_weights1, ema_factor],
        attributes={},
        numOutputs=1,
    )

    builder.customOp(
        opName="ExpMovAvg",
        opVersion=1,
        domain="com.acme",
        inputs=[weights2, exp_avg_weights2, ema_factor],
        attributes={},
        numOutputs=1,
    )

    model_output = builder.aiOnnx.add([builder.aiOnnx.matmul(
        [input, weights1]), builder.aiOnnx.matmul([input, weights2])])

    l1_loss = builder.aiGraphcore.l1loss([model_output], 2.0)

    anchor_types_dict = {
        l1_loss: popart.AnchorReturnType("ALL"),
    }

    proto = builder.getModelProto()
    dataflow = popart.DataFlow(conf.batches_per_step, anchor_types_dict)

    return proto, model_output, l1_loss, dataflow, [
        weights1, weights2], [
        exp_avg_weights1, exp_avg_weights2]


def add_conf_args():
    """ define the argument parser object """
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch Size")
    parser.add_argument('--input_dim', type=int, default=512,
                        help="Input Dimension")
    parser.add_argument('--output_dim', type=int, default=512,
                        help="Output Dimension")
    return parser


if __name__ == '__main__':
    parser = add_conf_args()
    conf = parser.parse_args()
    conf.batches_per_step = 1
    conf.replication_factor = 1
    conf.gradient_accumulation_factor = 16

    so_path = "build/exp_avg_custom_op.so"
    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)
    libc = ctypes.cdll.LoadLibrary(so_path)

    # building model and dataflow
    builder = popart.Builder()
    model_input = builder.addInputTensor(popart.TensorInfo("FLOAT",
                                                           [conf.batch_size,
                                                            conf.input_dim]),
                                         "test_input")

    ema_input = builder.addInputTensor(
        popart.TensorInfo("FLOAT", [1]), "test_ema")

    proto, model_output, l1_loss, dataflow, weights, exp_weights = create_model_and_dataflow_for_training(
        builder, conf, model_input, ema_input)

    optimizer_options = {"defaultLearningRate": (1.0, True),
                         "defaultMomentum": (0.0, True),
                         "defaultWeightDecay": (0.0, True),
                         "lossScaling": (1.0, True),
                         }

    optimizer = popart.SGD(optimizer_options)
    for i in exp_weights:
        optimizer.insertSpecific(i, {"learningRate": (0.0001, True)})

    session_options = popart.SessionOptions()
    if conf.replication_factor > 1:
        session_options.enableReplicatedGraphs = True
        session_options.replicatedGraphCount = conf.replication_factor
    if conf.gradient_accumulation_factor > 1:
        session_options.enableGradientAccumulation = True
        session_options.accumulationFactor = conf.gradient_accumulation_factor

    session_options.optimizerStateTensorLocationSettings.location.storage = popart.TensorStorage.OffChip
    session_options.optimizerStateTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On

    tensor_location_override_dict = dict()
    for wname in weights:
        tensor_location_override_dict["exp_mov_avg_" +
                                      wname] = popart.TensorLocation(popart.TensorStorage.OffChip)
    session_options.tensorLocationSettingsOverride = tensor_location_override_dict

    device = popart.DeviceManager().acquireAvailableDevice(conf.replication_factor)

    # create training session
    session = popart.TrainingSession(fnModel=proto,
                                     loss=l1_loss,
                                     deviceInfo=device,
                                     optimizer=optimizer,
                                     dataFlow=dataflow,
                                     userOptions=session_options)

    session.prepareDevice()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO(
        {
            model_input: np.random.random(
                (conf.replication_factor *
                 conf.batch_size *
                 conf.batches_per_step,
                 conf.gradient_accumulation_factor,
                 conf.input_dim)).astype(
                np.float32),
            ema_input: np.ones(
                (conf.replication_factor *
                 conf.batches_per_step,
                 conf.gradient_accumulation_factor,
                 1)).astype(
                np.float32) *
            0.9},
        anchors)
    session.weightsFromHost()
    session.run(stepio)
