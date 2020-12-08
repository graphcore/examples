# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import NamedTuple
import pytest
import numpy as np
import popart
from bert_optimizer import ScheduledOptimizerFactory, Schedule
from bert import Iteration

STEPWISE_LR_SCHEDULE = {0: 1e-10,
                        500: 1e-9,
                        1000: 1e-8,
                        1200: 1e-8/2,
                        1800: 1e-8/16}

EPOCHWISE_LR_SCHEDULE = {1: 1e-10,
                         4: 1e-9,
                         8: 1e-8,
                         12: 1e-8/2,
                         16: 1e-8/4}

STEPWISE_LS_SCHEDULE = {100: 4,
                        500: 8,
                        1000: 20,
                        1200: 16,
                        1800: 20}


class MockIteration:
    def __init__(self, epochs, steps_per_epoch):
        self.learning_rate = -1
        self.count = 0
        self.epoch = 0
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * epochs


class TestConfig(NamedTuple):
    __test__ = False

    task: str = "PRETRAINING"
    projection_lr_scale: float = 1.0

    learning_rate: float = 1e-8
    loss_scaling: float = 1.0

    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    velocity_scaling: float = 1.0

    # For the scheduled optimizer factory
    lr_schedule_by_epoch: dict = None
    lr_schedule_by_step: dict = None
    ls_schedule_by_epoch: dict = None
    ls_schedule_by_step: dict = None

    # For the linear optimizer factory
    enable_warmup: bool = False
    warmup_steps: int = 100
    warmup_init_lr: float = 0
    steps_per_warmup_update: int = 1
    enable_lr_decay: bool = False
    steps_per_decay_update: int = 1

    # For the per-tensor learning rate
    execution_mode: str = "PIPELINE"
    pipeline_lr_scaling: bool = False
    pipeline_lr_scaling_offset: float = 0.1
    pipeline_momentum_scaling: bool = False

    # Needed for iteration/optimizer integration test
    continue_training_from_epoch: int = 0
    epochs: int = 1
    epochs_per_save: int = 1
    steps_per_log: int = 1
    batch_size: int = 1
    gradient_accumulation_factor: int = 1
    replication_factor: int = 1

    inference_lm_perplexity: bool = False
    inference: bool = False

    squad_lr_scale: int = 1


def run_scheduled_optimizer_factory_case(config, iteration, epoch_truth=None, step_truth=None, option_name="defaultLearningRate"):
    """Runs a single case of the schedule optimizer factory tests. Simulates running through
    every step of every epoch required by the config, and updates the optimizer factory as
    defined by the schedule. Then checks the optimizer parameters to ensure they are correct."""
    if epoch_truth is None:
        epoch_truth = {}

    if step_truth is None:
        step_truth = {}

    factory = ScheduledOptimizerFactory(config, iteration)
    for iteration.epoch in range(iteration.epochs):
        for _ in range(iteration.steps_per_epoch):
            if factory.should_update(iteration):
                factory.update(iteration)

            if iteration.count in step_truth:
                lr = factory.option_values[option_name]
                assert lr == step_truth[iteration.count]
            iteration.count += 1

        if iteration.epoch in epoch_truth:
            lr = factory.option_values[option_name]
            assert lr == epoch_truth[iteration.epoch]


@pytest.mark.category1
@pytest.mark.parametrize(
    "config, epoch_truth, step_truth",
    [
        (
            TestConfig(learning_rate=1e-8),
            {0: 1e-8, 5: 1e-8, 10: 1e-8, 19: 1e-8},
            {0: 1e-8, 25: 1e-8, 50: 1e-8, 99: 1e-8}
        )
    ])
def test_scheduled_optimizer_factory_constant_lr(config, epoch_truth, step_truth):
    """ Check that the learning rate doesn't change if no schedule is provided"""
    iteration = MockIteration(20, 100)
    run_scheduled_optimizer_factory_case(config,
                                         iteration,
                                         epoch_truth=epoch_truth,
                                         step_truth=step_truth)


@pytest.mark.category1
@pytest.mark.parametrize(
    "mode, option_name, sched, manual_truth",
    [
        (
            "lr_schedule_by_epoch",
            "defaultLearningRate",
            {0: 1e-10, 5: 1e-9, 10: 1e-8, 12: 1e-8/2, 18: 1e-8/16},
            {1: 1e-10, 3: 1e-10, 4: 1e-10, 8: 1e-9, 20: 1e-8/16}
        ),
        (
            "lr_schedule_by_epoch",
            "defaultLearningRate",
            {1: 1e-10, 5: 1e-9, 10: 1e-8, 12: 1e-8/2, 18: 1e-8/16},
            {0: TestConfig().learning_rate, 3: 1e-10, 4: 1e-10, 8: 1e-9, 20: 1e-8/16}
        ),
        (
            "ls_schedule_by_epoch",
            "lossScaling",
            {1: 4, 4: 8, 8: 20, 12: 16, 16: 12},
            {0: TestConfig().loss_scaling, 3: 4, 5: 8, 12: 16, 19: 12}
        )
    ])
def test_scheduled_optimizer_factory_epochwise(mode, option_name, sched, manual_truth):
    def generate_epoch_truth(epoch_sched, additional):
        return {**epoch_sched, **additional}

    def generate_step_truth(epoch_truth, steps_per_epoch):
        return {e * steps_per_epoch: epoch_truth[e] for e in epoch_truth}

    iteration = MockIteration(20, 100)
    config = TestConfig(**{mode: sched})

    epoch_truth = generate_epoch_truth(sched, manual_truth)
    step_truth = generate_step_truth(epoch_truth, iteration.steps_per_epoch)
    run_scheduled_optimizer_factory_case(config, iteration, epoch_truth=epoch_truth, step_truth=step_truth, option_name=option_name)


@pytest.mark.category1
@pytest.mark.parametrize(
    "mode, option_name, sched, manual_truth",
    [
        (
            "lr_schedule_by_step",
            "defaultLearningRate",
            {0: 1e-10, 500: 1e-9, 1000: 1e-8, 1200: 1e-8/2, 1800: 1e-8/16},
            {10: 1e-10, 300: 1e-10, 600: 1e-9, 1100: 1e-8, 2000: 1e-8/16}
        ),
        (
            "lr_schedule_by_step",
            "defaultLearningRate",
            {250: 1e-10, 500: 1e-9, 1000: 1e-8, 1200: 1e-8/2, 1800: 1e-8/16},
            {0: TestConfig().learning_rate, 300: 1e-10, 600: 1e-9, 1100: 1e-8, 2000: 1e-8/16}
        ),
        (
            "ls_schedule_by_step",
            "lossScaling",
            {100: 4, 500: 8, 1000: 20, 1200: 16, 1800: 12},
            {0: TestConfig().loss_scaling, 300: 4, 600: 8, 1100: 20, 2000: 12}
        )
    ])
def test_scheduled_optimizer_factory_stepwise(mode, option_name, sched, manual_truth):

    def generate_step_truth(step_sched, additional):
        return {**step_sched, **additional}

    def generate_epoch_truth(step_truth, steps_per_epoch):
        return {math.floor(c / steps_per_epoch): step_truth[c] for c in step_truth}

    iteration = MockIteration(20, 100)
    config = TestConfig(**{mode: sched})

    step_truth = generate_step_truth(sched, manual_truth)
    epoch_truth = generate_epoch_truth(step_truth, iteration.steps_per_epoch)
    run_scheduled_optimizer_factory_case(config, iteration, epoch_truth=epoch_truth, step_truth=step_truth, option_name=option_name)


@pytest.mark.category1
@pytest.mark.parametrize(
    "manual_truth, option_name",
    [
        ({0: 3, 300: 4, 600: 8, 1100: 20, 2000: 12}, "lossScaling"),
        ({10: STEPWISE_LR_SCHEDULE[0], 300: STEPWISE_LR_SCHEDULE[0], 600: STEPWISE_LR_SCHEDULE[500], 1100: STEPWISE_LR_SCHEDULE[1000], 2000: STEPWISE_LR_SCHEDULE[1800]}, "defaultLearningRate")
    ])
def test_scheduled_optimizer_factory_multiple_schedules(manual_truth, option_name):
    """
    LR and LS: Step schedule, with an inferred init 0 value
    This test checks there's no interaction between the different schedules above."""
    iteration = MockIteration(20, 100)

    config = TestConfig(loss_scaling=3, learning_rate=STEPWISE_LR_SCHEDULE[0], ls_schedule_by_step=STEPWISE_LS_SCHEDULE, lr_schedule_by_step=STEPWISE_LR_SCHEDULE)

    if option_name == "lossScaling":
        sched = STEPWISE_LS_SCHEDULE
    else:
        sched = STEPWISE_LR_SCHEDULE

    # Check the loss scaling is correctly handled
    step_truth = {**sched, **manual_truth}
    epoch_truth = dict([(math.floor(c / iteration.steps_per_epoch), step_truth[c]) for c in step_truth])
    run_scheduled_optimizer_factory_case(config, iteration, epoch_truth=epoch_truth, step_truth=step_truth, option_name=option_name)


@pytest.mark.category1
@pytest.mark.parametrize(
    "config",
    [
        TestConfig(),
        TestConfig(lr_schedule_by_step=STEPWISE_LR_SCHEDULE),
        TestConfig(ls_schedule_by_epoch=EPOCHWISE_LR_SCHEDULE),
        TestConfig(lr_schedule_by_step=STEPWISE_LR_SCHEDULE,
                   ls_schedule_by_epoch=EPOCHWISE_LR_SCHEDULE)
    ])
def test_scheduled_optimiser_params_const_flag(config):
    """Check that scheduled parameters are correctly set to non-const, with others remaining const"""
    iteration = MockIteration(20, 100)
    factory = ScheduledOptimizerFactory(config, iteration)

    expected_non_const = []
    if config.lr_schedule_by_epoch is not None or config.lr_schedule_by_step is not None:
        expected_non_const.append("defaultLearningRate")
    if config.ls_schedule_by_epoch is not None or config.ls_schedule_by_step is not None:
        expected_non_const.append("lossScaling")

    optimizer_options = factory.optimizer_options

    for key, value in optimizer_options.items():
        assert not value[1] if key in expected_non_const else value[1]


@pytest.mark.category1
@pytest.mark.parametrize(
    "schedule, expected",
    [
        (
            {0: 1e-10, 5: 1e-9, 10: 1e-8, 12: 5e-9, 18: 2.5e-9},
            {0: 1e-10, 5: 1e-9, 10: 1e-8, 12: 5e-9, 18: 2.5e-9}
        ),
        (
            {"0": 1e-10, "5": 1e-9, "10": 1e-8, "12": 5e-9, "18": 2.5e-9},
            {0: 1e-10, 5: 1e-9, 10: 1e-8, 12: 5e-9, 18: 2.5e-9}
        ),
        (
            {0: "1e-10", 5: "1e-9", 10: "1e-8", 12: "5e-9", 18: "2.5e-9"},
            {0: 1e-10, 5: 1e-9, 10: 1e-8, 12: 5e-9, 18: 2.5e-9}
        ),
        (
            {"0": "1e-10", "5": "1e-9", "10": "1e-8", "12": "5e-9", "18": "2.5e-9"},
            {0: 1e-10, 5: 1e-9, 10: 1e-8, 12: 5e-9, 18: 2.5e-9}
        )
    ])
def test_schedule_key_parsing(schedule, expected):
    """Tests the parser can handle variations of string, float and int representations"""

    epoch_schedule = Schedule.from_args("defaultLearningRate", schedule, None, 1e-3)
    step_schedule = Schedule.from_args("defaultLearningRate", None, schedule, 1e-3)

    assert epoch_schedule.schedule == expected
    assert step_schedule.schedule == expected


@pytest.mark.category1
@pytest.mark.parametrize(
    "start_epoch, steps_per_epoch, num_epochs, lr_schedule, expected",
    [
        (0, 250, 10, STEPWISE_LR_SCHEDULE, STEPWISE_LR_SCHEDULE[0]),
        (1, 250, 10, STEPWISE_LR_SCHEDULE, STEPWISE_LR_SCHEDULE[0]),
        (3, 250, 10, STEPWISE_LR_SCHEDULE, STEPWISE_LR_SCHEDULE[500]),
        (5, 250, 10, STEPWISE_LR_SCHEDULE, STEPWISE_LR_SCHEDULE[1200])
    ])
def test_schedule_with_continue_from_epoch(start_epoch, steps_per_epoch, num_epochs, lr_schedule, expected):
    """Make sure the optimiser restarts the schedule from the correct point when resuming training
    from a given epoch"""

    config = TestConfig(**{
        "continue_training_from_epoch": start_epoch,
        "epochs": num_epochs,
        "lr_schedule_by_step": lr_schedule
    })

    iteration = Iteration(
        config,
        batches_per_step=10,
        steps_per_epoch=steps_per_epoch,
        writer=None,
        recording_steps=1)

    factory = ScheduledOptimizerFactory(config, iteration)
    lr = factory.option_values["defaultLearningRate"]
    assert lr == expected


@pytest.mark.category1
@pytest.mark.ipus(1)
@pytest.mark.parametrize(
    "steps_per_epoch, lr_schedule_by_step, layer_inputs, offset, pipeline",
    [
        (2, {0: 0.5, 1: 0.1}, [100, 200, 300], 0.1, True),
        (5, {0: 0.0008, 3: 0.00001}, [5, 3, 7], 0.25, True),
        (5, {0: 0.0008, 3: 0.00001}, [5, 3, 7], 0.1, False)
    ])
def test_per_tensor_lr(steps_per_epoch, lr_schedule_by_step, layer_inputs, offset, pipeline):
    """Ensure the learning rate is correctly applied to different tensors according to the per-tensor
    scale factor and offset"""

    def expected_step_weights(iteration, config, layer_input, lr_scale):
        """Calculate the expected weight values after successive steps"""
        step_schedule = config.lr_schedule_by_step

        test_case = [layer_input]
        min_schedule_step = min(step_schedule.keys())
        step_lr = step_schedule[min_schedule_step]

        for i in range(0, iteration.total_steps):
            if i in step_schedule:
                step_lr = step_schedule[i]

            scaled_lr = step_lr * lr_scale if config.execution_mode == "PIPELINE" else step_lr

            test_case.append(test_case[-1] - scaled_lr)
        return test_case

    def generate_test_case(offset, step_schedule, layer_inputs, pipeline=True):
        """Generate the config and ground-truth for a given set of parameters"""
        # Assuming 3 layers, work out the per-pipeline-stage LR scaling given the lower bound
        true_scaling_values = [
            offset,
            offset + (1 - offset) / 2,
            1
        ]

        config = TestConfig(
            lr_schedule_by_step=step_schedule,
            execution_mode="PIPELINE" if pipeline else "DEFAULT",
            pipeline_lr_scaling=pipeline,
            pipeline_lr_scaling_offset=offset)

        test_case = {stage: expected_step_weights(iteration,
                                                  config,
                                                  layer_inputs[stage],
                                                  true_scaling_values[stage])
                     for stage in range(3)}

        return test_case, config, true_scaling_values

    def assert_scaled_lr(factory, true_scaling_values):
        """Compare the per-stage learning rate scales against the true values"""
        for stage in factory.pipeline_stage_lr_scaling:
            assert(true_scaling_values[stage] ==
                   factory.pipeline_stage_lr_scaling[stage])

    def test(config, iteration, true_scaling, test_case):
        builder = popart.Builder()

        w0name = "weight_0"
        w1name = "weight_1"
        w2name = "weight_2"

        input0Shape = [1, 1, 1]
        input0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", input0Shape), "input0")

        w0data = np.array([test_case[0][0]], dtype=np.float32)
        w0R = np.empty([1, ], dtype=np.float32)
        w0Id = builder.addInitializedInputTensor(w0data, w0name)

        w1data = np.array([test_case[1][0]], dtype=np.float32)
        w1R = np.empty([1, ], dtype=np.float32)
        w1Id = builder.addInitializedInputTensor(w1data, w1name)

        w2data = np.array([test_case[2][0]], dtype=np.float32)
        w2R = np.empty([1, ], dtype=np.float32)
        w2Id = builder.addInitializedInputTensor(w2data, w2name)

        add0 = builder.aiOnnx.add([w0Id, input0])
        add1 = builder.aiOnnx.add([w1Id, add0])
        add2 = builder.aiOnnx.add([w2Id, add1])
        loss = builder.aiGraphcore.l1loss([add2], 1.0, debugPrefix="l1LossVal")
        builder.addOutputTensor(add2)

        proto = builder.getModelProto()
        dataFlow = popart.DataFlow(1, {})
        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}
        opts.enableGroupedMatmuls = False
        pat = popart.Patterns(popart.PatternsLevel.Default)
        dm = popart.DeviceManager()
        dm.setOnDemandAttachTimeout(int(1e4))
        device = dm.acquireAvailableDevice(
            1,
            connectionType=popart.DeviceConnectionType.OnDemand,
            selectionCriterion=popart.DeviceSelectionCriterion.Random)
        if device is None:
            raise OSError("Failed to acquire IPU.")

        # The stage->tensor map would come from the Bert model in reality
        # (see model.tensors)
        mock_tensor_map = {
            0: [w0Id],
            1: [w1Id],
            2: [w2Id]
        }

        factory = ScheduledOptimizerFactory(
            config, iteration, "SGD", tensors=mock_tensor_map)
        assert_scaled_lr(factory, true_scaling)

        optimizer_step0 = factory.create()

        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=dataFlow,
            userOptions=opts,
            loss=loss,
            optimizer=optimizer_step0,
            patterns=pat,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        input_data = np.array([3.1415], dtype=np.float32)
        stepio = popart.PyStepIO({input0: input_data}, anchors)

        for step in range(iteration.total_steps):
            session.run(stepio)
            session.weightsToHost()
            weightsRead = popart.PyWeightsIO({w0Id: w0R, w1Id: w1R, w2Id: w2R})
            session.readWeights(weightsRead)

            assert (np.isclose(test_case[0][step+1], w0R))
            assert (np.isclose(test_case[1][step+1], w1R))
            assert (np.isclose(test_case[2][step+1], w2R))

            iteration.count += 1

            if factory.should_update(iteration):
                optimizer_step1 = factory.update_and_create(iteration)
                assert_scaled_lr(factory, true_scaling)

                session.updateOptimizerFromHost(optimizer_step1)

    iteration = MockIteration(1, steps_per_epoch)
    test_case, config, true_scaling = generate_test_case(offset,
                                                         lr_schedule_by_step,
                                                         layer_inputs,
                                                         pipeline=pipeline)
    test(config, iteration, true_scaling, test_case)
