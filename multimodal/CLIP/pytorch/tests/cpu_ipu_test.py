# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import datetime
import warnings

import import_helper
import numpy as np
import popart
import poptorch
import pytest
import torch
from args import parse_args
from model import CLIP
from log import Logger


def test_ipu_cpu_match():
    """
    Test that the CLIP model ran on IPU approximately matches that same model ran on the CPU.
    """
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Config
    config = """
    --config unit_test
    """.split()
    config = parse_args(config)
    log = Logger("./" + datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + ".log", level="info")

    # Fix the random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # PopTorch options
    opts = poptorch.Options()
    opts.replicationFactor(config.replication_factor)
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.device_iterations)
    opts.Training.gradientAccumulation(config.gradient_accumulation)
    opts.outputMode(poptorch.OutputMode.Final)
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings().useOnChipStorage(True).useReplicatedTensorSharding(False)
    )
    opts.randomSeed(config.random_seed)

    opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))

    model_cpu = CLIP(config).train()
    model_ipu = CLIP(config).parallelize(log).train()

    model_ipu.load_state_dict(model_cpu.state_dict())

    # Check that copy was successful
    assert model_ipu is not model_cpu
    assert all([(a == b).all() for a, b in zip(model_cpu.parameters(), model_ipu.parameters())]) is True

    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=config.learning_rate)
    optimizer_ipu = poptorch.optim.AdamW(model_ipu.parameters(), lr=config.learning_rate, loss_scaling=1.0)

    poptorch_model = poptorch.trainingModel(model_ipu, opts, optimizer=optimizer_ipu)

    images = torch.randn(10, 3, 224, 224)
    input_ids = torch.randint(0, config.vocab_size, (10, config.context_length))
    images_ipu = images.repeat((config.gradient_accumulation, 1, 1, 1))
    input_ids_ipu = input_ids.repeat((config.gradient_accumulation, 1))

    # Training loop
    for step in range(10):
        # Step IPU model
        iput_output = poptorch_model(images_ipu, input_ids_ipu)
        ipu_loss = iput_output

        # Step cpu model
        optimizer_cpu.zero_grad()
        for b in range(config.gradient_accumulation):
            cpu_output = model_cpu(images, input_ids)
            cpu_loss = cpu_output
            cpu_loss.div(config.gradient_accumulation).backward()

        optimizer_cpu.step()

        with torch.no_grad():
            print(f"CPU Loss: {cpu_loss}, IPU Loss: {ipu_loss.numpy()}")
            # Check the losses are approximately equal
            assert np.allclose(cpu_loss.numpy(), ipu_loss.numpy(), atol=1e-6)


if __name__ == "__main__":
    test_ipu_cpu_match()
