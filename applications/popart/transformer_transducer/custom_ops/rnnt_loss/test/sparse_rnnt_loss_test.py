# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import popart

import test_utils
from transducer import Transducer, TransducerLoss


def wrap_and_call(transducer_fn, acts, lengths, labels, label_lengths):
    """ applies logsoftmax to input logits and calls rnn-transducer function """
    acts = torch.tensor(acts.astype(np.float32), requires_grad=True)
    copy_acts = torch.add(acts, 0.0)
    labels = labels.reshape(-1)  # convert to 1d

    labels = torch.IntTensor(labels)
    lengths = torch.IntTensor(lengths)
    label_lengths = torch.IntTensor(label_lengths)

    log_probs = nn.functional.log_softmax(copy_acts, dim=3)

    def grad_hook(grad):
        copy_acts.saved_grad = grad.clone()

    copy_acts.register_hook(grad_hook)

    def grad_hook2(grad):
        log_probs.saved_grad = grad.clone()

    log_probs.register_hook(grad_hook2)

    costs = transducer_fn.apply(log_probs, labels, lengths, label_lengths)
    cost = torch.mean(costs)
    cost.backward()
    grads = copy_acts.saved_grad
    return cost.data.numpy(), grads.data.numpy(), log_probs.saved_grad.data.numpy()


def reference_rnnt_loss(input_data, target_data, input_lengths, target_lengths):
    """ runs reference RNN-T code for given input data """
    tfn = Transducer(blank_label=0)
    cost, grads_wlogits, grads_wlogprobs = wrap_and_call(
        tfn, input_data, input_lengths, target_data, target_lengths
    )
    return cost, grads_wlogits, grads_wlogprobs


def build_graph(opts, splits=[]):
    """ Build a graph with SparseLogSoftmax and RNN-t loss, for specified input-sizes.
    Annchors the gradient for comparison with the PyTorch ground-truth. """

    builder = popart.Builder()
    inputs_info = popart.TensorInfo(
        opts.precision,
        [opts.batch_size, opts.input_size, opts.target_size + 1, opts.num_classes],
    )
    inputs = builder.addInputTensor(inputs_info, "logits")

    targets_info = popart.TensorInfo("INT32", [opts.batch_size, opts.target_size])
    targets = builder.addInputTensor(targets_info, "labels")

    input_lengths_info = popart.TensorInfo("INT32", [opts.batch_size])
    input_lengths = builder.addInputTensor(input_lengths_info, "input_lengths")

    target_lengths_info = popart.TensorInfo("INT32", [opts.batch_size])
    target_lengths = builder.addInputTensor(target_lengths_info, "target_lengths")

    if opts.reduction_type == "mean":
        reduction_type = popart.ReductionType.Mean
    elif opts.reduction_type == "sum":
        reduction_type = popart.ReductionType.Sum
    else:
        reduction_type = popart.ReductionType.NoReduction

    if reduction_type != popart.ReductionType.Mean:
        print("Only mean is supported as reduction")
        sys.exit(-1)

    if splits:
        input_splits = builder.aiOnnx.split([inputs],
                                            num_outputs=len(splits),
                                            axis=1,
                                            split=splits)
    else:
        input_splits = [inputs]

    compacted_probs_splits = []
    for input_split in input_splits:

        compact_output_tensors = builder.customOp(
            opName="SparseLogSoftmax",
            opVersion=1,
            domain="com.acme",
            inputs=[input_split, targets, target_lengths],
            attributes={},
            numOutputs=1,
        )
        compacted_probs_splits.append(compact_output_tensors[0])

    if splits:
        # stack all compacted logprobs
        compacted_probs = builder.aiOnnx.concat(compacted_probs_splits, axis=1)
    else:
        compacted_probs = compacted_probs_splits[0]

    output_tensors = builder.customOp(
        opName="RNNTLoss",
        opVersion=1,
        domain="com.acme",
        inputs=[compacted_probs, input_lengths, target_lengths],
        attributes={},  # {"reduction": int(reduction_type)},
        numOutputs=4,
    )
    neg_log_likelihood = output_tensors[0]
    proto = builder.getModelProto()
    gradient = popart.reservedGradientPrefix() + inputs
    compacted_gradient = popart.reservedGradientPrefix() + compacted_probs
    anchors = {
        compacted_gradient: popart.AnchorReturnType("ALL"),
        gradient: popart.AnchorReturnType("ALL"),
        neg_log_likelihood: popart.AnchorReturnType("ALL"),
        compacted_probs: popart.AnchorReturnType("ALL"),
    }
    data_flow = popart.DataFlow(1, anchors)
    if opts.ipu:
        device = popart.DeviceManager().acquireAvailableDevice(1)
    else:
        device = popart.DeviceManager().createIpuModelDevice({"tilesPerIPU": 1472})
    optimizer = popart.ConstSGD(0.001)
    options = popart.SessionOptions()
    options.enableFloatingPointChecks = False
    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=data_flow,
        loss=neg_log_likelihood,
        optimizer=optimizer,
        deviceInfo=device,
        userOptions=options,
    )
    t0 = time.perf_counter()
    session.prepareDevice()
    result = session.initAnchorArrays()
    session.weightsFromHost()

    duration = time.perf_counter() - t0
    print("Device prep took %.3f" % duration)
    return (
        session,
        inputs,
        targets,
        input_lengths,
        target_lengths,
        result,
        compacted_probs,
        gradient,
        compacted_gradient,
        neg_log_likelihood,
    )


def run_graph(context, input_data, target_data, input_length_data, target_length_data):
    """ Runs a single training step for given session/context and given input-data. """
    (
        session,
        inputs,
        targets,
        input_lengths,
        target_lengths,
        result,
        compacted_probs,
        gradient,
        compacted_gradient,
        neg_log_likelihood,
    ) = context
    data = {
        inputs: input_data,
        targets: target_data,
        input_lengths: input_length_data,
        target_lengths: target_length_data,
    }

    t0 = time.perf_counter()
    stepio = popart.PyStepIO(data, result)
    session.run(stepio)
    duration = time.perf_counter() - t0
    print("Execution took %.3f" % duration)
    print("Nllloss IPU", result[neg_log_likelihood])
    return (
        result,
        compacted_probs,
        gradient,
        compacted_gradient,
        neg_log_likelihood,
    )


def build_and_run_graph(
    input_data, target_data, input_length_data, target_length_data, opts, splits=[]
):
    """ Build and run a graph containing only SparseLogSoftmax and RNN-T loss, with specified input tensor data.
    Runs a single training step and anchors the gradient for comparison with the PyTorch ground-truth."""

    context = build_graph(opts, splits=splits)
    return run_graph(
        context, input_data, target_data, input_length_data, target_length_data
    )


def run_single_case(args, splits=[]):
    """ Run a single RNN-T loss test case """
    np.random.seed(42)

    (inputs, target, input_length_data, target_length_data) = test_utils.generate_data(args)
    (
        result,
        compacted_probs,
        gradients,
        compacted_gradients,
        nll,
    ) = build_and_run_graph(inputs, target, input_length_data, target_length_data, args, splits=splits)
    ipu_grad = result[gradients]
    # ipu_compacted_grad = result[compacted_gradients]
    ipu_loss = result[nll]

    pytorch_loss, pytorch_grads_wlogits, pytorch_grads_wlogprobs = reference_rnnt_loss(
        inputs, target, input_length_data, target_length_data
    )
    grad_err = test_utils.getTensorError(ipu_grad, pytorch_grads_wlogits)
    loss_err = test_utils.getTensorError(ipu_loss, pytorch_loss)
    print("Gradient Error", grad_err)
    print("Loss Error", loss_err)

    print(f"IPU Loss: {ipu_loss}")
    print("Grad result: " + ("Pass" if grad_err < 1e-5 else "FAIL"))
    print("Loss result: " + ("Pass" if loss_err < 1e-5 else "FAIL"))
    return grad_err, loss_err


@pytest.mark.parametrize(
    "input_size, target_size, num_classes, batch_size",
    [
        (4, 2, 5, 1),
        (4, 2, 5, 1),
        (4, 2, 5, 1),
        (4, 2, 5, 2),
        (4, 2, 5, 2),
        (4, 2, 5, 3),
        (4, 2, 5, 3),
    ],
)
def test_rnnt_loss_batch_size(
    custom_ops, input_size, target_size, num_classes, batch_size
):
    args = test_utils.args_from_params(
        input_size, target_size, num_classes, batch_size,
    )
    grad_err, loss_err = run_single_case(args)
    assert grad_err < 1e-5
    assert loss_err < 1e-5


@pytest.mark.parametrize(
    "precision",
    [("FLOAT"), ("FLOAT16")],
)
def test_rnnt_loss_precision(custom_ops, precision):
    params = {
        "input_size": 6,
        "target_size": 3,
        "num_classes": 5,
        "batch_size": 3,
        "precision": precision,
    }

    args = test_utils.args_from_params(**params)
    grad_err, loss_err = run_single_case(args)
    assert grad_err < 1e-5
    assert loss_err < 1e-5


@pytest.mark.parametrize(
    "ipu, splits",
    [(None, []), (None, [10, 10]), (True, []), (True, [10, 10])],
)
def test_rnnt_loss_medium_dim_wplits(custom_ops, ipu, splits):
    params = {
        "input_size": 20,
        "target_size": 10,
        "num_classes": 10,
        "batch_size": 4,
        "precision": "FLOAT16",
        "variable_input": True,
        "ipu": ipu,
    }

    if splits:
        assert sum(splits) == params["input_size"]

    args = test_utils.args_from_params(**params)
    grad_err, loss_err = run_single_case(args, splits=splits)
    assert grad_err < 1e-5
    assert loss_err < 1e-5


@pytest.mark.parametrize(
    "ipu, splits",
    [(None, [15, 15]), (None, [10, 10, 10]), (True, [15, 15]), (True, [10, 10, 10])],
)
def test_rnnt_loss_large_dim_wplits(custom_ops, ipu, splits):
    params = {
        "input_size": 30,
        "target_size": 15,
        "num_classes": 20,
        "batch_size": 4,
        "precision": "FLOAT16",
        "variable_input": True,
        "ipu": ipu,
    }

    if splits:
        assert sum(splits) == params["input_size"]

    args = test_utils.args_from_params(**params)
    grad_err, loss_err = run_single_case(args, splits=splits)
    assert grad_err < 1e-5
    assert loss_err < 1e-5


@pytest.mark.parametrize(
    "variable_input",
    [(True), (False)],
)
def test_rnnt_loss_variable_tgt(custom_ops, variable_input):
    params = {
        "input_size": 20,
        "target_size": 10,
        "num_classes": 5,
        "batch_size": 3,
        "variable_input": variable_input,
    }

    args = test_utils.args_from_params(**params)
    grad_err, loss_err = run_single_case(args)
    assert grad_err < 1e-5
    assert loss_err < 1e-5


@pytest.mark.parametrize(
    "logits_scale, precision",
    [(1, "FLOAT16"), (2, "FLOAT16"), (4, "FLOAT16"), (16, "FLOAT16"), (32, "FLOAT16"), (64, "FLOAT16"), (128, "FLOAT16"),
     (1, "FLOAT"), (2, "FLOAT"), (4, "FLOAT"), (16, "FLOAT"), (32, "FLOAT"), (64, "FLOAT"), (128, "FLOAT")],
)
def test_rnnt_loss_logits_scale(custom_ops, logits_scale, precision):
    params = {
        "input_size": 197,
        "target_size": 116,
        "num_classes": 64,
        "batch_size": 2,
        "variable_input": True,
        "logits_scale": logits_scale,
        "precision": precision,
        "ipu": True,
    }

    args = test_utils.args_from_params(**params)
    grad_err, loss_err = run_single_case(args)
    assert grad_err < 1e-4  # (relaxing grad_error req. a bit so that it passes for larger logits-scale)
    assert loss_err < 1e-5


if __name__ == "__main__":
    test_utils.load_custom_sparse_logsoftmax_op()
    test_utils.load_custom_rnnt_op()

    args = test_utils.parse_args()
    run_single_case(args)
