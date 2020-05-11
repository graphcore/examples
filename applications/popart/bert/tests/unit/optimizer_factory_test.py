# Copyright 2019 Graphcore Ltd.
import pytest
import sys
import math
import popart
import numpy as np
from typing import NamedTuple
from bert_optimizer import ScheduledOptimizerFactory, Schedule
from bert import Iteration


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


@pytest.mark.category1
def test_scheduled_optimizer_factory():

    def test_case(config, iteration, epoch_truth={}, step_truth={}, option_name="defaultLearningRate"):
        factory = ScheduledOptimizerFactory(config, iteration)
        for iteration.epoch in range(iteration.epochs):
            for step in range(iteration.steps_per_epoch):
                if factory.should_update(iteration):
                    factory.update(iteration)

                if iteration.count in step_truth:
                    lr = factory.option_values[option_name]
                    assert(lr == step_truth[iteration.count])
                iteration.count += 1

            if iteration.epoch in epoch_truth:
                lr = factory.option_values[option_name]
                assert(lr == epoch_truth[iteration.epoch])

    # =============================  Constant learning rate  ==============================
    iteration = MockIteration(20, 100)
    config = TestConfig(learning_rate=1e-8)

    epoch_truth = {0: 1e-8, 5: 1e-8, 10: 1e-8, 19: 1e-8}
    step_truth = {0: 1e-8, 25: 1e-8, 50: 1e-8, 99: 1e-8}
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth)

    #  ===================  Epoch schedule, with a given init 0 value  ====================
    iteration = MockIteration(20, 100)
    lr_schedule_by_epoch = {0: 1e-10, 5: 1e-9,
                            10: 1e-8, 12: 1e-8/2, 18: 1e-8/16}
    config = TestConfig(lr_schedule_by_epoch=lr_schedule_by_epoch)
    epoch_truth = {**lr_schedule_by_epoch, **
                   {1: 1e-10, 3: 1e-10, 4: 1e-10, 8: 1e-9, 20: 1e-8/16}}
    step_truth = dict([(e * iteration.steps_per_epoch, epoch_truth[e])
                       for e in epoch_truth])
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth)

    #  ================  Epoch schedule, with an inferred init 0 value  ===================
    iteration = MockIteration(20, 100)
    lr_schedule_by_epoch = {1: 1e-10,
                            5: 1e-9,
                            10: 1e-8,
                            12: 1e-8/2,
                            18: 1e-8/16}
    config = TestConfig(learning_rate=1e-8, lr_schedule_by_epoch=lr_schedule_by_epoch)
    epoch_truth = {**lr_schedule_by_epoch, **{0: 1e-8,
                                              3: 1e-10,
                                              4: 1e-10,
                                              8: 1e-9,
                                              20: 1e-8/16}}
    step_truth = dict([(e * iteration.steps_per_epoch, epoch_truth[e])
                       for e in epoch_truth])
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth)

    #  ====================  Step schedule, with a given init 0 value  ====================
    iteration = MockIteration(20, 100)
    lr_schedule_by_step = {0: 1e-10,
                           500: 1e-9,
                           1000: 1e-8,
                           1200: 1e-8/2,
                           1800: 1e-8/16}
    config = TestConfig(lr_schedule_by_step=lr_schedule_by_step)
    step_truth = {**lr_schedule_by_step, **{10: 1e-10,
                                            300: 1e-10,
                                            600: 1e-9,
                                            1100: 1e-8,
                                            2000: 1e-8/16}}
    epoch_truth = dict(
        [(math.floor(c / iteration.steps_per_epoch), step_truth[c]) for c in step_truth])
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth)

    #  ====================  Step schedule, with a given init 0 value  ====================
    iteration = MockIteration(20, 100)
    lr_schedule_by_step = {0: 1e-10,
                           500: 1e-9,
                           1000: 1e-8,
                           1200: 1e-8/2,
                           1800: 1e-8/16}
    config = TestConfig(learning_rate=1e-8, lr_schedule_by_step=lr_schedule_by_step)
    step_truth = {**lr_schedule_by_step, **{0: 1e-10,
                                            300: 1e-10,
                                            600: 1e-9,
                                            1100: 1e-8,
                                            2000: 1e-8/16}}
    epoch_truth = dict(
        [(math.floor(c / iteration.steps_per_epoch), step_truth[c]) for c in step_truth])
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth)

    #  ==========  Loss Scaling: Epoch schedule, with an inferred init 0 value  ===========
    iteration = MockIteration(20, 100)
    ls_schedule_by_epoch = {1: 4,
                            4: 8,
                            8: 20,
                            12: 16,
                            16: 12}
    config = TestConfig(loss_scaling=3, ls_schedule_by_epoch=ls_schedule_by_epoch)
    epoch_truth = {**ls_schedule_by_epoch, **{0: 3,
                                              3: 4,
                                              5: 8,
                                              12: 16,
                                              19: 12}}
    step_truth = dict([(e * iteration.steps_per_epoch, epoch_truth[e])
                       for e in epoch_truth])
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth, option_name="lossScaling")

    #  ==========  Loss Scaling: Step schedule, with an inferred init 0 value  ============
    iteration = MockIteration(20, 100)
    ls_schedule_by_step = {100: 4,
                           500: 8,
                           1000: 20,
                           1200: 16,
                           1800: 12}
    config = TestConfig(loss_scaling=3, ls_schedule_by_step=ls_schedule_by_step)
    step_truth = {**ls_schedule_by_step, **{0: 3,
                                            300: 4,
                                            600: 8,
                                            1100: 20,
                                            2000: 12}}
    epoch_truth = dict(
        [(math.floor(c / iteration.steps_per_epoch), step_truth[c]) for c in step_truth])
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth, option_name="lossScaling")

    #  ============  LR and LS: Step schedule, with an inferred init 0 value  =============
    # This test checks there's no interaction between the different schedules above.
    iteration = MockIteration(20, 100)
    ls_schedule_by_step = {100: 4,
                           500: 8,
                           1000: 20,
                           1200: 16,
                           1800: 12}

    lr_schedule_by_step = {100: 1e-10,
                           500: 1e-9,
                           1000: 1e-8,
                           1200: 1e-8/2,
                           1800: 1e-8/16}

    config = TestConfig(loss_scaling=3, learning_rate = 1e-7,
                        ls_schedule_by_step=ls_schedule_by_step,
                        lr_schedule_by_step=lr_schedule_by_step)

    # Check the loss scaling is correctly handled
    step_truth = {**ls_schedule_by_step, **{0: 3,
                                            300: 4,
                                            600: 8,
                                            1100: 20,
                                            2000: 12}}
    epoch_truth = dict(
        [(math.floor(c / iteration.steps_per_epoch), step_truth[c]) for c in step_truth])
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth, option_name="lossScaling")

    # Check the learning rate is correctly handled
    iteration = MockIteration(20, 100)
    step_truth = {**lr_schedule_by_step, **{10: 1e-7,
                                            300: 1e-10,
                                            600: 1e-9,
                                            1100: 1e-8,
                                            2000: 1e-8/16}}
    epoch_truth = dict(
        [(math.floor(c / iteration.steps_per_epoch), step_truth[c]) for c in step_truth])
    test_case(config, iteration, epoch_truth=epoch_truth,
              step_truth=step_truth, option_name="defaultLearningRate")


@pytest.mark.category1
def test_scheduled_optimiser_params_const_flag():

    def test_case(config):
        iteration = MockIteration(20, 100)
        factory = ScheduledOptimizerFactory(config, iteration)

        expected_non_const = []
        if config.lr_schedule_by_epoch is not None or config.lr_schedule_by_step is not None:
            expected_non_const.append("defaultLearningRate")
        if config.ls_schedule_by_epoch is not None or config.ls_schedule_by_step is not None:
            expected_non_const.append("lossScaling")

        optimizer_options = factory.optimizer_options

        for key, value in optimizer_options.items():
            assert(not value[1] if key in expected_non_const else value[1])

    lr_schedule_by_step = {0: 1e-10,
                           500: 1e-9,
                           1000: 1e-8,
                           1200: 1e-8/2,
                           1800: 1e-8/16}

    ls_schedule_by_epoch = {1: 4,
                            4: 8,
                            8: 20,
                            12: 16,
                            16: 12}

    configs = [TestConfig(),
               TestConfig(lr_schedule_by_step=lr_schedule_by_step),
               TestConfig(ls_schedule_by_epoch=ls_schedule_by_epoch),
               TestConfig(lr_schedule_by_step=lr_schedule_by_step,
                          ls_schedule_by_epoch=ls_schedule_by_epoch)]

    for config in configs:
        test_case(config)


@pytest.mark.category1
def test_schedule_key_parsing():
    """Tests the parser can handle variations of string, float and int representations"""
    iteration = MockIteration(1, 550)

    schedule_int_keys = {0: 1e-10,
                         5: 1e-9,
                         10: 1e-8,
                         12: 5e-9,
                         18: 2.5e-9}

    schedule_str_keys = {"0": 1e-10,
                         "5": 1e-9,
                         "10": 1e-8,
                         "12": 5e-9,
                         "18": 2.5e-9}

    schedule_str_vals = {0: 1e-10,
                         5: 1e-9,
                         10: 1e-8,
                         12: 5e-9,
                         18: 2.5e-9}

    schedule_str_both = {"0": "1e-10",
                         "5": "1e-9",
                         "10": "1e-8",
                         "12": "5e-9",
                         "18": "2.5e-9"}

    test_cases = [schedule_int_keys, schedule_str_keys,
                  schedule_str_vals, schedule_str_both]

    test_schedules_epoch = [Schedule.from_args(
        "defaultLearningRate", case, None, 1e-3) for case in test_cases]
    test_schedules_steps = [Schedule.from_args(
        "defaultLearningRate", None, case, 1e-3) for case in test_cases]

    equality = [test.schedule ==
                test_schedules_epoch[0].schedule for test in test_schedules_epoch]
    assert(all(equality))
    equality = [test.schedule ==
                test_schedules_epoch[0].schedule for test in test_schedules_steps]
    assert(all(equality))


iteration_test_lr_schedule_by_step = {0: 1e-10,
                                      25: 1e-9,
                                      50: 1e-8,
                                      75: 1e-8/2,
                                      100: 1e-8/16}


@pytest.mark.category1
@pytest.mark.parametrize(
    "start_epoch, steps_per_epoch, num_epochs, lr_schedule, expected",
    [
        (0, 10, 10, iteration_test_lr_schedule_by_step, iteration_test_lr_schedule_by_step[0]),
        (1, 10, 10, iteration_test_lr_schedule_by_step, iteration_test_lr_schedule_by_step[0]),
        (3, 10, 10, iteration_test_lr_schedule_by_step, iteration_test_lr_schedule_by_step[25]),
        (5, 10, 10, iteration_test_lr_schedule_by_step, iteration_test_lr_schedule_by_step[50])])
def test_schedule_with_continue_from_epoch(start_epoch, steps_per_epoch, num_epochs, lr_schedule, expected):

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
    assert(lr == expected)


@pytest.mark.category1
def test_per_tensor_lr():

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

        builder.addOutputTensor(add2)

        proto = builder.getModelProto()
        dataFlow = popart.DataFlow(1, {})
        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}
        opts.enableGroupedMatmuls = False
        pat = popart.Patterns(popart.PatternsLevel.DEFAULT)
        device = popart.DeviceManager().acquireAvailableDevice(1)
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
            config, iteration, tensors=mock_tensor_map)
        assert_scaled_lr(factory, true_scaling)

        optimizer_step0 = factory.create()

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            losses=[popart.L1Loss(add2, "l1LossVal", 1.0)],
            optimizer=optimizer_step0,
            passes=pat,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        input_data = np.array([3.1415], dtype=np.float32)
        stepio = popart.PyStepIO({input0: input_data}, anchors)
        session.optimizerFromHost()

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

                session.updateOptimizer(optimizer_step1)
                session.optimizerFromHost()

    #  ==============================  2 Steps, decayed LR  ===============================
    iteration = MockIteration(1, 2)
    lr_schedule_by_step = {0: 0.5,
                           1: 0.1}

    layer_inputs = [100, 200, 300]
    test_case, config, true_scaling = generate_test_case(0.1,
                                                         lr_schedule_by_step,
                                                         layer_inputs)
    test(config, iteration, true_scaling, test_case)

    #  ========================== 5 Steps, slower decayed LR  =============================
    iteration = MockIteration(1, 5)
    lr_schedule_by_step = {0: 0.0008,
                           3: 0.00001}

    layer_inputs = [5, 3, 7]
    test_case, config, true_scaling = generate_test_case(0.25,
                                                         lr_schedule_by_step,
                                                         layer_inputs)
    test(config, iteration, true_scaling, test_case)

    #  =================== 5 Steps, pipeline disabled (no scaled LR)  =====================
    iteration = MockIteration(1, 5)
    lr_schedule_by_step = {0: 0.0008,
                           3: 0.00001}

    layer_inputs = [5, 3, 7]
    test_case, config, true_scaling = generate_test_case(0.1,
                                                         lr_schedule_by_step,
                                                         layer_inputs,
                                                         pipeline=False)
    test(config, iteration, true_scaling, test_case)


