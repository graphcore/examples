# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import popart
import pytest
import numpy as np
from args import parse_args
from models import DALLE
from models import VQGanVAE


def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


def test_ipu_cpu_match():
    """
    Test that the DALL-E model ran on IPU approximately matches that same
    model ran on the CPU.
    """
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Config
    config = """
    --config unit_test
    """.split()
    config = parse_args(config)

    # options
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    # Poptorch options
    opts = poptorch.Options()
    opts.replicationFactor(1)

    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.device_iterations)
    opts.Training.gradientAccumulation(config.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
    opts.outputMode(poptorch.OutputMode.Final)
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        .useOnChipStorage(True)
        .useReplicatedTensorSharding(False))
    opts.randomSeed(config.random_seed)

    # Precision options
    opts.Precision.enableStochasticRounding(False)

    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(1)
    }
    opts.setAvailableMemoryProportion(mem_prop)


    # Models
    dalle_params = dict(
        num_text_tokens=8192,
        text_seq_len=config.text_seq_len,
        dim=config.hidden_size,
        depth=config.num_hidden_layers,
        heads=config.num_attention_heads,
        dim_head=config.dim_head,
        loss_img_weight=config.loss_img_weight,
        attn_types=tuple(config.attn_types.split(',')),
        ff_dropout=config.ff_dropout,
        attn_dropout=config.attn_dropout,
    )

    vae = VQGanVAE(config.vqgan_model_path, config.vqgan_config_path)
    model_cpu = DALLE(vae=vae, **dalle_params).train()
    model_ipu = DALLE(vae=vae, **dalle_params).train()
    model_ipu.load_state_dict(model_cpu.state_dict())

    # Check that copy was successful
    assert model_ipu is not model_cpu
    assert all([(a == b).all() for a, b in zip(
        model_cpu.parameters(), model_ipu.parameters())]) is True

    optimizer_cpu = torch.optim.AdamW(get_trainable_params(model_cpu), lr=3e-4)
    optimizer_ipu = poptorch.optim.AdamW(get_trainable_params(model_ipu), lr=3e-4, loss_scaling=1.0)
    poptorch_model = poptorch.trainingModel(model_ipu, opts, optimizer=optimizer_ipu)

    # Input
    IMAGE_SIZE = vae.image_size
    batch_size = config.batch_size
    text_cpu = torch.tensor([[593, 2276, 1465, 5466, 1644,  638, 1219, 6682,  594,  557, 2162, 1554,
                              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                              0,    0,    0,    0,    0,    0,    0,    0]])
    image_cpu = torch.tensor([[601,  697,  417,  306,  553,  666,   84,  707,  956,  253,  758,  614,
                               820, 1007,  808,  611,  903,  636,  897,  683,  479,  436,  633,  624,
                               248,  609,  580,  126,   66,  783,  628,  519,  328,  924,  921,  579,
                               819,  977,  921,  365,  493,  789,  580,  915,  301,  219,  186,  126,
                               873,  382,  789,  683,  653,  682,  921,  682,  773,  945,  494,  929,
                               892,  473,   36,   35,  299,  256,  444,  632,  444,  215,  598,  184,
                               787,  250,  593,  772,  656,  154,  830,  579,  218,  964,  426,   36,
                               27,  913,    4,  990,  707,  988,  370,  830,  586,  243,  313,   94,
                               96,  671,  720,  719,  642,  171,  498,  313,  184,  432,  500,  244,
                               913,  988,  700,  928,  382, 1018,  705,  187,  681,  243,  996,   83,
                               32,  202,  148,  173,  210,  546,  287,  820,  168,  957,  215,  914,
                               134,  185,  611, 1002,  575,  432,  873,  749,  215,  253,  906,  810,
                               812,  812,  219,  219,  375, 1022,  713,  389,  186,  965,   56,  710,
                               210, 1006,  788,  618,   63,  154,  669,  325,  407,  306,  979,  849,
                               354,  382,  426,  894,  661,  633,    4,  810,  748,  992,  192,  494,
                               382,  261,  382,  914,  494,  681,  701,  656,  748,  692,  737,  972,
                               328,  992,  172,  985,  154,  918,   63,   20,  808,  868,  634,  462,
                               498,   20,  830,  854,  600,   54,  794,  572,  143,   20,  812,  618,
                               589,  590,  873,  737,  918,  171,  830,   66,  972,  572,  244,  382,
                               36,  572,  685,  922,  222,  329,  258,  595,  642,  171,  830,  460,
                               964,  590,  714,  520,  714,   37,   55,  323,  316,  258,   37,  906,
                               893,  334,  503,  226]])
    text_ipu = text_cpu.repeat((config.gradient_accumulation, 1))
    image_ipu = image_cpu.repeat((config.gradient_accumulation, 1))
    # Training Loop
    for step in range(10):
        # Step IPU model
        ipu_output = poptorch_model(text_ipu, image_ipu)
        ipu_loss = ipu_output

        # Step CPU Model
        optimizer_cpu.zero_grad()
        for b in range(config.gradient_accumulation):
            cpu_output = model_cpu(text_cpu, image_cpu)
            cpu_loss = cpu_output
            cpu_loss.div(config.gradient_accumulation).backward()
        optimizer_cpu.step()

        with torch.no_grad():
            print(f"CPU Loss: {cpu_loss}, IPU Loss: {ipu_loss.numpy()[0]}")
            # Check the losses are approximately equal
            assert np.allclose(cpu_loss.numpy(), ipu_loss.numpy()[0], atol=1e-1)
