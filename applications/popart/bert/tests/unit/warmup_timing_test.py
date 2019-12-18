# Copyright 2019 Graphcore Ltd.
import sys
import os
import ctypes
import torch
import math
import random
import numpy as np
import popart
import time
import onnx
import pytest
from typing import Iterable, Tuple, Any, Union, Mapping, Callable, Optional, NamedTuple
from bert_model import BertConfig, Bert
from bert import bert_training_session
from tests.utils import make_tuple

so_path = os.path.join(os.getcwd(), "custom_ops.so")
ctypes.cdll.LoadLibrary(so_path)


def step(session, anchors, data, update_optimizer_lr=None):
    if update_optimizer_lr is not None:
        optimizer = popart.SGD(update_optimizer_lr)
        session.updateOptimizer(optimizer)

    stepio = popart.PyStepIO(data, anchors)
    session.run(stepio)


def timed_run_steps(session, anchors, data, update_optimizer, num_steps=10000):

    step_times = np.empty((num_steps,), dtype=np.float)

    for i in range(num_steps):
        step_start = time.time()
        step(session, anchors, data, update_optimizer)
        step_end = time.time()
        step_times[i] = step_end - step_start

    return step_times


def create_session(proto: onnx.ModelProto,
                   data: Mapping[str, np.ndarray],
                   outputs: Optional[Union[str, Iterable[str]]],
                   optimizer: popart.SGD,
                   loss: Optional[Union[popart.Loss, Iterable[popart.Loss]]] = None,
                   ipus: Optional[int] = None):
    outputs = make_tuple(outputs)
    if loss is not None:
        loss = make_tuple(loss)
    # Setting up the Session
    data_flow = popart.DataFlow(
        1, {output: popart.AnchorReturnType("ALL")
            for output in outputs})

    options = popart.SessionOptions()
    options.enableGroupedMatmuls = False
    # With an Inference session we are actually testing the fwd pass of training.
    options.constantWeights = False
    options.enableStochasticRounding = False

    if ipus is not None:
        options.enableVirtualGraphs = True
    else:
        ipus = 1

    request_ipus = pow(2, math.ceil(math.log2(ipus)))
    device = popart.DeviceManager().acquireAvailableDevice(request_ipus)

    if device is None:
        raise Exception("Failed to acquire IPU.")

    session = popart.TrainingSession(fnModel=proto,
                                     deviceInfo=device,
                                     dataFeed=data_flow,
                                     userOptions=options,
                                     losses=loss,
                                     optimizer=optimizer)

    session.prepareDevice()
    session.weightsFromHost()
    session.optimizerFromHost()
    session.setRandomSeed(1984)

    anchors = session.initAnchorArrays()

    return session, anchors, device


@pytest.mark.skip("Move to Requirements Tests")
def test_warmup(custom_ops, num_steps=100000):
    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1})
    config = BertConfig(vocab_length=9728,
                        num_layers=1,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        custom_ops=['gather', 'attention'])
    popart_model = Bert(config, builder=builder)

    sequence_info = popart.TensorInfo(
        "UINT32", [config.batch_size * config.sequence_length])
    indices = builder.addInputTensor(sequence_info)
    positions = builder.addInputTensor(sequence_info)
    data = {
        indices: np.random.randint(0, config.vocab_length, (config.batch_size * config.sequence_length)).astype(np.uint32),
        positions: np.random.randint(
            0, config.sequence_length, (config.batch_size * config.sequence_length)).astype(np.uint32)
    }

    output = popart_model.build_graph(indices, positions)[0]

    losses = [popart.L1Loss(output, "l1LossVal", 0.1)]

    for loss in losses:
        loss.virtualGraph(popart_model.ipu)

    proto = popart_model.builder.getModelProto()
    optimizer = popart.SGD(0.00001)

    ipus = math.ceil(config.num_layers / config.layers_per_ipu) \
        + popart_model.layer_offset

    # Analagous to run_py, but only the setup stages
    print("Creating session and compiling graph")
    session, anchors, device = create_session(
        proto, data, output, optimizer, losses, ipus=ipus)

    print("Running with opimiser updates")
    times_with_optimiser = timed_run_steps(
        session, anchors, data, 0.1, num_steps=num_steps)
    print("Running without opimiser updates")
    times_no_optimiser = timed_run_steps(
        session, anchors, data, None, num_steps=num_steps)

    device.detach()

    # Convert seconds to milliseconds.
    opt_np = 1000*times_with_optimiser
    noopt_np = 1000*times_no_optimiser

    print(f"W/  Optimiser Update")
    print(f"\tMean: {opt_np.mean():.5f}")
    print(f"\tSum:  {opt_np.sum():.5f}")
    print(f"\tRng: {opt_np.min():.5f} -> {opt_np.max():.5f}")

    print(f"W/o  Optimiser Update")
    print(f"\tMean: {noopt_np.mean():.5f}")
    print(f"\tSum:  {noopt_np.sum():.5f}")
    print(f"\tRng: {noopt_np.min():.5f} -> {noopt_np.max():.5f}")

    mean_diff = opt_np.mean() - noopt_np.mean()
    percentage_difference = 100*mean_diff / noopt_np.mean()
    print(
        f"Mean difference, {mean_diff:.5f}ms (~{percentage_difference:.1f}%)")

    assert(percentage_difference < 5)
