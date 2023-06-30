# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from asyncio.log import logger
import numpy as np
import poptorch
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))

from modeling import loss as module_loss
from modeling import model as module_arch
from configs.parse_config import ConfigParser


def test_ipu_cpu_match():
    """
    Test that the Frozen-in-time model ran on IPU approximately matches that same
    model ran on the CPU.
    Usage: python test/cpu_ipu_test.py
    """
    import warnings

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Config
    config = ConfigParser(config="configs/unit_test.json")

    # poptorch options
    opts = poptorch.Options()
    options = config["IPU_options"]

    # fix the random seeds
    torch.manual_seed(options.get("random_seed", 0))
    np.random.seed(options.get("random_seed", 0))
    opts.randomSeed(options.get("random_seed", 0))

    opts.replicationFactor(options.get("replication_factor", 1))
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(options.get("batches_per_step", 1))
    gradientAccumulation = options["training"].get("gradientAccumulation", 3)
    opts.Training.gradientAccumulation(gradientAccumulation)
    opts.setAvailableMemoryProportion(config["IPU_options"]["available_memory_proportion"])
    opts.outputMode(poptorch.OutputMode.Final)
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings().useOnChipStorage(True).useReplicatedTensorSharding(False)
    )

    opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))

    model_cpu = module_arch.PipelinedWithLoss(
        config.initialize("arch", module_arch), config.initialize(name="loss", module=module_loss), logger
    ).train()
    model_ipu = (
        module_arch.PipelinedWithLoss(
            config.initialize("arch", module_arch), config.initialize(name="loss", module=module_loss), logger
        )
        .parallelize(config["IPU_options"]["pipelined_layers"])
        .train()
    )

    model_ipu.load_state_dict(model_cpu.state_dict())

    # check that copu was successful
    assert model_ipu is not model_cpu
    assert all([(a == b).all() for a, b in zip(model_cpu.parameters(), model_ipu.parameters())]) is True

    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=config["optimizer"]["args"]["lr"])
    optimizer_ipu = poptorch.optim.AdamW(model_ipu.parameters(), lr=config["optimizer"]["args"]["lr"], loss_scaling=1.0)

    poptorch_model = poptorch.trainingModel(model_ipu, opts, optimizer=optimizer_ipu)

    input_ids = torch.randint(
        0, 30522, (config["data_loader"]["training"]["batch_size"], config["arch"]["args"]["text_params"]["max_length"])
    )
    attention_mask = torch.randint(
        0, 1, (config["data_loader"]["training"]["batch_size"], config["arch"]["args"]["text_params"]["max_length"])
    )
    video = torch.rand(
        (
            config["data_loader"]["training"]["batch_size"],
            config["data_loader"]["training"]["video_params"]["num_frames"],
            3,
            224,
            224,
        ),
        dtype=torch.float32,
    )
    input_ids_ipu = input_ids.repeat((gradientAccumulation, 1))
    attention_mask_ipu = attention_mask.repeat((gradientAccumulation, 1))
    video_ipu = video.repeat((gradientAccumulation, 1, 1, 1, 1))

    # Training loop
    for step in range(10):
        # Step IPU model
        iput_output = poptorch_model(input_ids_ipu, attention_mask_ipu, video_ipu)
        ipu_loss = iput_output
        # Step cpu model
        optimizer_cpu.zero_grad()
        for b in range(gradientAccumulation):
            cpu_loss = model_cpu(input_ids, attention_mask, video)
            cpu_loss.backward()
        optimizer_cpu.step()

        with torch.no_grad():
            # Check the losses are approximately equal
            assert np.allclose(cpu_loss.numpy(), ipu_loss.numpy(), atol=1e-3)
