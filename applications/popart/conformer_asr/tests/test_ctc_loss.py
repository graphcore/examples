# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import time
import numpy as np
import torch
import popart

import test_utils


def torch_ctc_loss(input_data, target_data, input_lengths, target_lengths, opts):
    """For a representative input, calculate the pytorch CTC loss and the input gradients
    to use as reference for IPU implementation"""

    # PyTorch CPU doesn't support FP16
    if opts.precision == "FLOAT16":
        input_data = input_data.astype(np.float32)

    # PyTorch CTCLoss expect inputs with shape (seq_length, batch_size, num_symbols)
    input_data = np.transpose(input_data, (1, 0, 2)).reshape(
        (opts.input_size, opts.batch_size, opts.num_classes)
    )

    # PyTorch doesn't support uint32
    target_data = target_data.astype(np.int32).reshape(
        (opts.batch_size, opts.target_size)
    )
    _, batch_size, _ = input_data.shape
    assert target_data.shape[0] == batch_size

    ctc_loss = torch.nn.CTCLoss(blank=0, reduction=opts.reduction_type)

    t0 = time.perf_counter()
    torch_input_data = torch.from_numpy(input_data).requires_grad_()

    loss = ctc_loss(
        torch_input_data,
        torch.from_numpy(target_data),
        torch.from_numpy(input_lengths.astype(np.int32)),
        torch.from_numpy(target_lengths.astype(np.int32)),
    )
    duration = time.perf_counter() - t0
    print("Execution took %.3f" % duration)
    print("Pytorch loss", loss)

    loss.backward(retain_graph=True)
    return loss.data.numpy(), torch_input_data.grad


def build_popart_graph(opts):
    """Build a popart graph containing only the CTC loss, with specified input sizes.
    Anchors the gradient for comparison with the PyTorch reference."""

    builder = popart.Builder()
    inputs_info = popart.TensorInfo(
        opts.precision, [opts.batch_size, opts.input_size, opts.num_classes]
    )
    inputs = builder.addInputTensor(inputs_info, "log_probs")
    log_probs = builder.aiOnnx.logsoftmax([inputs], axis=2)

    targets_info = popart.TensorInfo("UINT32", [opts.batch_size, opts.target_size])
    targets = builder.addInputTensor(targets_info, "targets")

    input_lengths_info = popart.TensorInfo("UINT32", [opts.batch_size])
    input_lengths = builder.addInputTensor(input_lengths_info, "input_length")

    target_lengths_info = popart.TensorInfo("UINT32", [opts.batch_size])
    target_lengths = builder.addInputTensor(target_lengths_info, "target_length")

    if opts.reduction_type == "mean":
        reduction_type = popart.ReductionType.Mean
    elif opts.reduction_type == "sum":
        reduction_type = popart.ReductionType.Sum
    else:
        reduction_type = popart.ReductionType.NoReduction

    partial32 = 1 if opts.partial32 else 0

    output_tensors = builder.customOp(
        opName="CtcLoss",
        opVersion=1,
        domain="com.acme",
        inputs=[log_probs, targets, input_lengths, target_lengths],
        attributes={
            "blank": 0,
            "reduction": int(reduction_type),
            "partial32": partial32,
        },
        numOutputs=4,
    )
    neg_log_likelihood = output_tensors[0]
    proto = builder.getModelProto()
    gradient = popart.reservedGradientPrefix() + log_probs
    anchors = {
        gradient: popart.AnchorReturnType("ALL"),
        neg_log_likelihood: popart.AnchorReturnType("ALL"),
        log_probs: popart.AnchorReturnType("ALL"),
    }
    data_flow = popart.DataFlow(1, anchors)
    if opts.ipu:
        device = popart.DeviceManager().acquireAvailableDevice(1)
    else:
        device = popart.DeviceManager().createIpuModelDevice({"tilesPerIPU": 1216})
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
        log_probs,
        gradient,
        neg_log_likelihood,
    )


def run_popart_graph(context, input_data, target_data, input_length_data, target_length_data):
    """ Runs the popart graph for the given inputs """
    (
        session,
        inputs,
        targets,
        input_lengths,
        target_lengths,
        result,
        log_probs,
        gradient,
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
    return (result, log_probs, gradient, neg_log_likelihood)


def build_and_run_popart_graph(input_data, target_data, input_length_data, target_length_data, opts):
    """ Build and run a popart graph containing only the CTC loss, with specified input tensor data.
    Runs a single training step and anchors the gradient for comparison with the PyTorch reference. """

    context = build_popart_graph(opts)

    return run_popart_graph(
        context, input_data, target_data, input_length_data, target_length_data
    )


def run_single_case(args):
    """ Run a single CTC loss test case """
    np.random.seed(42)

    (inputs, target, input_length_data, target_length_data) = test_utils.generate_data(args)
    (result, log_probs, gradients, nll) = build_and_run_popart_graph(
        inputs, target, input_length_data, target_length_data, args
    )
    ipu_grad = result[gradients]
    ipu_loss = result[nll]
    print(ipu_grad)

    pytorch_loss, pytorch_res = torch_ctc_loss(
        result[log_probs], target, input_length_data, target_length_data, args
    )

    ipu_grad = np.transpose(ipu_grad, (1, 0, 2)).reshape(pytorch_res.shape)

    grad_err = test_utils.getTensorError(ipu_grad, pytorch_res)
    loss_err = test_utils.getTensorError(ipu_loss, pytorch_loss)
    print("Gradient Error", grad_err)
    print("Loss Error", loss_err)

    print(f"IPU Loss: {ipu_loss}")
    print("Grad result: " + ("Pass" if grad_err < 1e-4 else "FAIL"))
    print("Loss result: " + ("Pass" if loss_err < 1e-5 else "FAIL"))
    return grad_err, loss_err


# Note that we do not test NoReduction at BS > 1 as Popart requires scalar losses
@pytest.mark.ipus(1)
@pytest.mark.parametrize(
    "input_size, target_size, num_classes, batch_size, reduction_type, ipu",
    [
        (4, 2, 5, 1, "mean", False),
        (4, 2, 5, 1, "mean", True),
        (4, 2, 5, 1, "sum", False),
        (4, 2, 5, 1, "sum", True),
        (4, 2, 5, 1, "none", False),
        (4, 2, 5, 1, "none", True),
        (4, 2, 5, 2, "mean", False),
        (4, 2, 5, 2, "mean", True),
        (4, 2, 5, 2, "sum", False),
        (4, 2, 5, 2, "sum", True),
        (4, 2, 5, 3, "mean", False),
        (4, 2, 5, 3, "mean", True),
        (4, 2, 5, 3, "sum", False),
        (4, 2, 5, 3, "sum", True),
    ],
)
def test_ctc_loss_batch_size(custom_ops, input_size, target_size, num_classes, batch_size, reduction_type, ipu):
    """ test across different batch-sizes """
    args = test_utils.args_from_params(
        input_size, target_size, num_classes, batch_size, reduction_type
    )
    grad_err, loss_err = run_single_case(args)
    assert grad_err < 1e-4
    assert loss_err < 1e-5


@pytest.mark.ipus(1)
@pytest.mark.parametrize(
    "reduction_type, precision, ipu",
    [("mean", "FLOAT", False), ("mean", "FLOAT16", False), ("sum", "FLOAT", False), ("sum", "FLOAT16", False),
     ("mean", "FLOAT", True), ("mean", "FLOAT16", True), ("sum", "FLOAT", True), ("sum", "FLOAT16", True)],
)
def test_ctc_loss_precision(custom_ops, reduction_type, precision, ipu):
    """ test across different reduction types and precisions """
    params = {
        "input_size": 6,
        "target_size": 3,
        "num_classes": 5,
        "batch_size": 3,
        "reduction_type": reduction_type,
        "precision": precision,
        "ipu": ipu,
    }

    args = test_utils.args_from_params(**params)
    grad_err, loss_err = run_single_case(args)
    assert grad_err < 1e-4
    assert loss_err < 1e-5


@pytest.mark.ipus(1)
@pytest.mark.parametrize(
    "batch_size, precision, ipu",
    [(1, "FLOAT16", False), (1, "FLOAT", False), (2, "FLOAT16", False), (2, "FLOAT", False),
     (4, "FLOAT16", False), (4, "FLOAT", False),
     (1, "FLOAT16", True), (1, "FLOAT", True), (2, "FLOAT16", True), (2, "FLOAT", True),
     (4, "FLOAT16", True), (4, "FLOAT", True)
     ],
)
def test_ctc_loss_asr_dim(custom_ops, batch_size, precision, ipu):
    """ test for typical ASR model dimensions"""
    num_classes = 36
    input_size = 375
    target_size = 200
    reduction_type = "mean"

    args = test_utils.args_from_params(
        input_size, target_size, num_classes, batch_size, reduction_type, precision, ipu
    )
    grad_err, loss_err = run_single_case(args)
    assert grad_err < 1e-4
    assert loss_err < 1e-5


@pytest.mark.ipus(1)
@pytest.mark.parametrize(
    "reduction_type, variable_input, ipu",
    [("mean", True, False), ("mean", False, False), ("sum", True, False), ("mean", False, False),
     ("mean", True, True), ("mean", False, True), ("sum", True, True), ("mean", False, True)],
)
def test_ctc_loss_variable_tgt(custom_ops, reduction_type, variable_input, ipu):
    """ test for both constant and variable sequence lengths """
    params = {
        "input_size": 20,
        "target_size": 10,
        "num_classes": 5,
        "batch_size": 3,
        "reduction_type": reduction_type,
        "variable_input": variable_input,
        "ipu": ipu,
    }

    args = test_utils.args_from_params(**params)
    grad_err, loss_err = run_single_case(args)
    assert grad_err < 1e-4
    assert loss_err < 1e-5


if __name__ == "__main__":

    test_utils.load_custom_ops_lib()
    args = test_utils.parse_args()
    run_single_case(args)
