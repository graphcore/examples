# Copyright 2020 Graphcore Ltd.
import numpy as np
import popart
import argparse
import time
import os
import sys
import datetime
import json
from onnx import TensorProto, numpy_helper
from collections import deque
from resnet_builder import PopartBuilderResNet
from resnet_data import load_dataset
from torch.utils.tensorboard import SummaryWriter

np.random.seed(0)  # For predictable weight initialization


def popart_writer(opts):
    writer = SummaryWriter(log_dir=opts.log_dir_name)
    return writer


def _get_popart_type(np_type):
    return {
        np.float16: 'FLOAT16',
        np.float32: 'FLOAT'
    }[np_type]


class Timer:


    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self._interval = self.end - self.start

    def interval(self):
        return self._interval


class ResNet(PopartBuilderResNet):


    def __init__(self, opts, builder=popart.Builder()):
        self.builder = builder
        self.dtype = opts.precision
        super(PopartBuilderResNet, self).__init__(opts)

        # Apply dataset specific changes.
        if opts.dataset == 'CIFAR-10':
            self.initial_k = 3
            self.initial_mp = False
            self.num_classes = 10
        elif opts.dataset == 'IMAGENET':
            self.initial_k = 7
            self.initial_mp = True
            self.num_classes = 1000
        else:
            raise ValueError("Unknown Dataset {}".format(opts.dataset))


def init_session(proto, losses, device, dataFlow,
                 options, training, optimizer=None,
                 gcpLogDir=None):

    # Create a session to compile and execute the graph
    if training:
        session_type = "training"
        session = popart.TrainingSession(fnModel=proto,
                                         losses=losses,
                                         deviceInfo=device,
                                         optimizer=optimizer,
                                         dataFeed=dataFlow,
                                         userOptions=options)

    else:
        session_type = "validation"
        session = popart.InferenceSession(fnModel=proto,
                                          losses=losses,
                                          deviceInfo=device,
                                          dataFeed=dataFlow,
                                          userOptions=options)


    print("Preparing the {} graph".format(session_type))
    with Timer() as prepareTimer:
        session.prepareDevice()

    print("{0} graph preparation complete. Duration: {1:.3f} seconds".format(session_type.capitalize(), prepareTimer.interval()))

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    return session, anchors


def create_inputs(builder, opts):

    # Data Inputs
    if opts.dataset == 'CIFAR-10':
        image = builder.addInputTensor(
            popart.TensorInfo(_get_popart_type(opts.precision),
                              [opts.samples_per_replica, 4, 32, 32]))

    elif opts.dataset == 'IMAGENET':
        image = builder.addInputTensor(
            popart.TensorInfo(_get_popart_type(opts.precision),
                              [opts.samples_per_replica, 4, 224, 224]))

    else:
        raise ValueError("Unknown Dataset {}".format(opts.dataset))

    label = builder.addInputTensor(
        popart.TensorInfo("INT32", [opts.samples_per_replica]))


    return image, label


def get_device(num_ipus, simulation=False):

    deviceManager = popart.DeviceManager()
    if simulation:
        print("Creating ipu sim")
        ipu_options = {
            "compileIPUCode": True,
            'numIPUs': num_ipus,
            "tilesPerIPU": 1216
        }
        device = deviceManager.createIpuModelDevice(ipu_options)
        if device is None:
            raise OSError("Failed to acquire IPU.")
    else:
        print("Aquiring IPU")
        device = deviceManager.acquireAvailableDevice(num_ipus)
        if device is None:
            raise OSError("Failed to acquire IPU.")
        else:
            print("Acquired IPU: {}".format(device))

    return device


def get_options(opts, training=True):

    # Create a session to compile and execute the graph
    options = popart.SessionOptions()

    options.engineOptions = {
        "debug.allowOutOfMemory": "true"
    }

    # Enable the reporting of variables in the summary report
    options.reportOptions = {'showVarStorage': 'true'}

    if opts.fp_exceptions:
        # Enable exception on floating point errors
        options.enableFloatingPointChecks = True

    if opts.prng:
        options.enableStochasticRounding = True

    # Need to disable constant weights so they can be set before
    # executing the inference session
    options.constantWeights = False

    # Enable auto-sharding
    if opts.num_ipus > 1 and opts.num_ipus > opts.replication_factor:
        options.enableVirtualGraphs = True
        options.virtualGraphMode = popart.VirtualGraphMode.Auto

    # Enable pipelining
    if opts.pipeline:
        options.enablePipelining = True

    # Enable graph to be replicated
    if opts.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = opts.replication_factor

    if opts.gradient_accumulation_factor > 1:
        if training:
            options.enableGradientAccumulation = True
            options.accumulationFactor = opts.gradient_accumulation_factor

    # Don't pull the accumulation tensors off the device
    options.disableGradAccumulationTensorStreams = True

    # Enable recomputation
    if opts.recompute:
        if opts.pipeline:
            options.autoRecomputation = popart.RecomputationType.Pipeline
        else:
            options.autoRecomputation = popart.RecomputationType.Standard

    return options


def create_model(builder, opts, image, label):

    resnet = ResNet(opts, builder)
    resnet.train = True

    logits = resnet(image)

    probs = resnet.builder.aiOnnx.softmax([logits])

    argmax = resnet.builder.aiOnnx.argmax([probs], axis=1, keepdims=0)

    loss = popart.NllLoss(probs, label, "loss",
                          reduction=popart.ReductionType.Mean)

    outputs = {
        argmax: popart.AnchorReturnType("ALL"),
        "loss": popart.AnchorReturnType("ALL")
    }

    proto = resnet.builder.getModelProto()

    return proto, loss, argmax, outputs


def calulate_learning_rate(opts, steps_per_epoch):

    base_lr = 2 ** opts.base_learning_rate
    decay_lr = opts.learning_rate_decay
    lrs = [base_lr * opts.batch_size * decay for decay in decay_lr]

    iterations = int(opts.epochs * steps_per_epoch)
    lr_drops = [int(i * iterations) for i in opts.learning_rate_schedule]
    return lrs, lr_drops


def train_process(opts):

    builder = popart.Builder()

    # Create the resnet model
    image, label = create_inputs(builder, opts)

    # Create the data set
    training_dataset = load_dataset(opts, training=True)
    validation_dataset = load_dataset(opts, training=False)

    # Calulate the learning rate for training
    steps_per_epoch = len(training_dataset)
    lrs, lr_drops = calulate_learning_rate(opts, steps_per_epoch)
    current_lr = lrs.pop(0)
    next_drop = lr_drops.pop(0)

    # Get the popart session options
    training_options = get_options(opts, training=True)

    # Set up a summary writer
    writer = popart_writer(opts)

    # Get the device to run on
    device = get_device(opts.num_ipus, opts.simulation)

    # Create the training session
    proto, loss, argmax, outputs = create_model(
        builder, opts, image, label)

    (training_session,
     training_anchors) = init_session(
        proto,
        [loss],
        device,
        dataFlow=popart.DataFlow(
            opts.batches_per_step,
            outputs),
        options=training_options,
        training=True,
        optimizer=popart.SGD({
            "defaultLearningRate": (current_lr, False),
            "defaultWeightDecay": (opts.weight_decay, True)
            }),
        gcpLogDir=opts.gc_profile_log_dir)

    if not opts.no_validation:

        # Create the validation session
        validation_options = get_options(opts, training=False)

        (validation_session,
         validation_anchors) = init_session(proto, [loss],
                                            device,
                                            dataFlow=popart.DataFlow(
                                                opts.batches_per_step,
                                                outputs),
                                            options=validation_options,
                                            training=False,
                                            gcpLogDir=opts.gc_profile_log_dir)

    # Copy weights and optimization parameters onto the device
    training_session.weightsFromHost()
    training_session.optimizerFromHost()

    batch_losses = deque(maxlen=opts.steps_per_log)
    batch_accs = deque(maxlen=opts.steps_per_log)
    batch_run_duration = deque(maxlen=opts.steps_per_log)
    total_samples = 0

    validation_losses = deque(maxlen=opts.steps_per_log)
    validation_accs = deque(maxlen=opts.steps_per_log)

    # Iterations
    for e in range(opts.epochs):

        # Set the timing start point for training
        training_start_point = time.time()

        print("Executing epoch ", e)
        # Do one epoch of training
        for step, data in enumerate(training_dataset):

            total_steps = (e*steps_per_epoch) + step
            epoch = e + (step/steps_per_epoch)

            # Follow Learning Rate Schedule
            if total_steps > next_drop:
                current_lr = lrs.pop(0)
                if len(lr_drops) > 0:
                    next_drop = lr_drops.pop(0)
                else:
                    next_drop = np.inf

                training_session.updateOptimizer(popart.SGD({
                    "defaultLearningRate": (current_lr, False),
                    "defaultWeightDecay": (opts.weight_decay, True)
                    }))
                training_session.optimizerFromHost()
                print("Learning_rate change to {}".format(current_lr))

            images = data[0]
            labels = data[1]

            stepio = popart.PyStepIO(
                {
                    image: images,
                    label: labels
                }, training_anchors)

            # Train
            with Timer() as t1:
                training_session.run(stepio)

            batch_run_duration.append(t1.interval())

            # Get the loss and 'learnt' labels
            # - Sum the losses across replication & batch size
            nll_loss_anch = training_anchors["loss"]
            arg_max_anch = training_anchors[argmax]

            batch_losses.append(nll_loss_anch)
            batch_accs.append(100*np.mean(arg_max_anch == labels))

            total_samples += (opts.batches_per_step * opts.batch_size * opts.gradient_accumulation_factor)

            if not total_steps % opts.steps_per_log or total_steps == 0:

                training_duration = time.time() - training_start_point

                print_format = ("step: {step:6d}, epoch: {epoch:6.2f}, "
                                "lr: {lr:6.2g}, loss: {loss:6.3f}, "
                                "accuracy: {train_acc:6.3f}%, "
                                "img/sec: {img_per_sec:6.2f} "
                                "step_time: {duration:6.2f} sec "
                                "ipu_execution_time: {run_duration:6.2f}")

                stats = {
                    'step': total_steps,
                    'epoch': epoch,
                    'lr': current_lr,
                    'loss': np.mean(batch_losses),
                    'train_acc': np.mean(batch_accs),
                    'img_per_sec': total_samples/training_duration,
                    'duration': training_duration,
                    'run_duration': np.mean(batch_run_duration),
                }

                print(print_format.format(**stats))

                # log the metrics
                writer.add_scalar('loss', np.mean(batch_losses), total_steps)
                writer.add_scalar('train_acc', np.mean(batch_accs), total_steps)
                writer.add_scalar('learning rate', current_lr, total_steps)

                # Reset the metrics
                batch_accs.clear()
                batch_losses.clear()
                batch_run_duration.clear()
                total_samples = 0

                # Reset the training start point
                training_start_point = time.time()

        # Evaluation
        if not opts.no_validation:

            # The onnx file that is the graph we created, with weights set by current state of training
            onnx_file_name = "ckpt.onnx"

            training_session.modelToHost(onnx_file_name)

            # Copy weights and optimization parameters onto the device
            validation_session.resetHostWeights(onnx_file_name)
            validation_session.weightsFromHost()

            validation_start_point = time.time()
            for validation_data in validation_dataset:

                validation_images = validation_data[0]
                validation_labels = validation_data[1]

                validation_stepio = popart.PyStepIO(
                    {
                        image: validation_images,
                        label: validation_labels
                    }, validation_anchors)

                validation_session.run(validation_stepio)

                # Get the loss and 'predicted' labels
                validation_nll_loss_anch = validation_anchors["loss"]
                validation_arg_max_anch = validation_anchors[argmax]

                validation_losses.append(validation_nll_loss_anch)
                validation_accs.append(
                    100 * np.mean(validation_arg_max_anch == validation_labels))

            print("Validation accuracy epoch {:6.2f}, img/sec:{:6.2f} "
                  "accuracy: {:6.3f}% loss: {:6.3f}"
                  .format(
                      epoch,
                      (
                          len(validation_dataset) *
                          opts.batch_size *
                          opts.batches_per_step /
                          (time.time() - validation_start_point)
                      ),
                      np.mean(validation_accs),
                      np.mean(validation_losses)))

            writer.add_scalar('val_acc', np.mean(validation_accs), total_steps)

            training_session.resetHostWeights(onnx_file_name)

            # Write the training weights to the device
            training_session.weightsFromHost()
            training_session.optimizerFromHost()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ResNet/ResNext training in Popart',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # -------------- DATASET ------------------
    group = parser.add_argument_group('Dataset')
    group.add_argument(
        '--dataset', type=str, choices=["IMAGENET", "CIFAR-10"],
        default="CIFAR-10",
        help="Choose which dataset to train on")
    group.add_argument(
        '--data-dir', type=str, required=True,
        help="Path to data")
    group.add_argument(
        '--num-workers', type=int, default=16,
        help="The number of Pytorch dataloader workers that are forked")

    # -------------- MODEL ------------------
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--batch-size', type=int, default=32,
        help='The number of samples used over all graph replicas to determine a weight update. This number must be a multiple of replication-factor.'
             'Note that if using gradient accumulation, the number of samples used to determine a weight update (a.k.a. the effective batch size) is this number multiplied by the gradient accumulation factor.')
    group.add_argument(
        '--size', type=str, choices=["8", "20", "18", "50", "x50"], default="20",
        help='Size of Resnet graph. An "x" prefix, such as "x50", indicates a ResNext model')
    group.add_argument(
        '--norm-type', choices=["BATCH", "GROUP", "NONE"], default="GROUP",
        help="Choose which normalization to use after each convolution")
    group.add_argument(
        '--norm-groups', type=int, default=16,
        help="Sets the number of groups when using the 'GROUP' norm-type")
    group.add_argument(
        '--shortcut-type', choices=['A', 'B', 'C'], default='B',
        help="ResNet shortcut type. Defaults to definition specified")
    group.add_argument(
        '--precision', choices=['16', '32'], default="16",
        help="Setting of float datatype 16 or 32 bits")
    group.add_argument(
        '--prng', action="store_true", default=True,
        help="Enable Stochastic Rounding")
    group.add_argument(
        '--no-prng', action="store_false", dest='prng', default=False,
        help="Disable Stochastic Rounding")
    group.add_argument(
        '--fp-exceptions', action="store_true", default=False,
        help="Enable floating point (FP) exceptions. This halts execution and prints debugging output when an FP exception is encountered.")
    # -------------- TRAINING ------------------
    group = parser.add_argument_group('Training')

    group.add_argument(
        '--base-learning-rate', type=int, default=-6,
        help="Base learning rate exponent (2**N). blr = lr /  bs")
    group.add_argument(
        '--learning-rate-decay', type=str, default="1,0.1,0.01",
        help="Learning rate decay schedule. Comma Separated ('1,0.1,0.01')")
    group.add_argument(
        '--learning-rate-schedule', type=str, default="0.5,0.75",
        help="Learning rate drop points (proportional to total number of training steps)."
             "Comma Separated ('0.5,0.75')")
    group.add_argument(
        '--epochs', type=int, default=30,
        help="Number of training epochs")
    group.add_argument(
        '--no-validation', action="store_true",
        help="Don't do any validation runs.")
    group.add_argument(
        '--valid-per-epoch', type=float, default=1,
        help="Validation steps per epoch.")
    group.add_argument(
        '--steps-per-log', type=int, default=1,
        help="Log statistics every N steps.")
    group.add_argument(
        '--weight-decay', type=float, default=0,
        help="Value for weight decay bias. Setting to 0 removes weight decay.")
    group.add_argument(
        '--num-ipus', type=int, default=1,
        help="Number of IPUs")
    group.add_argument(
        '--replication-factor', type=int, default=1,
        help="Number of times to replicate the graph to perform data parallel"
             " training. Must be a factor of the number of IPUs")
    group.add_argument(
        '--recompute', action="store_true", default=False,
        help="Enable recomputations of activations in backward pass")
    group.add_argument(
        '--pipeline', action="store_true", default=False,
        help="Pipeline the model over IPUs")
    group.add_argument(
        '--batches-per-step', type=int, default=250,
        help="How many minibatches to perform on the device before returning"
             "to the host.")
    group.add_argument(
        '--simulation', action="store_true",
        help="Run the program on the IPU Model")
    group.add_argument(
        '--gradient-accumulation-factor', type=int, default=1,
        help="Accumulate a number of gradients before updating the weights")
    # -------------- LOGGING ------------------
    group = parser.add_argument_group('Logging')
    group.add_argument(
        '--log-dir', type=str, default="logs",
        help="Name of parent directory for logs and tensorboard protobufs")
    group.add_argument(
        '--log_dir_string', type=str, default="net",
        help="Append a string to the name of the log-dir subdirectory for this run, for tracking")
    group.add_argument(
        '--get-reports', action="store_true",
        help="Store graph report in log-dir")

    args = parser.parse_args()

    args.learning_rate_decay = list(map(
        float, args.learning_rate_decay.split(',')))
    args.learning_rate_schedule = list(
        map(float, args.learning_rate_schedule.split(',')))
    args.learning_rate = (2**args.base_learning_rate) * (args.batch_size)

    if ((args.batch_size % args.replication_factor) != 0):
        raise Exception("Invalid Argument : Batch size ({}) must be a "
                        "multiple of replication factor ({})"
                        .format(args.batch_size, args.replication_factor))

    if ((args.num_ipus % args.replication_factor) != 0):
        raise Exception("Invalid Argument : Number of IPUs ({}) must be a "
                        "multiple of replication factor ({})"
                        .format(args.num_ipus, args.replication_factor))

    # The number of samples that each replica will process at once
    args.samples_per_replica = int(args.batch_size / args.replication_factor)
    args.effective_batch_size = args.batch_size * args.gradient_accumulation_factor

    # Create a directory to store tensorboard protobuf and summary log
    timestamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    log_dir_components = [timestamp, str(args.size), str(args.log_dir_string)]
    log_path = str(args.log_dir)

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    for component in log_dir_components:
        path = os.path.join(log_path, component)
        os.mkdir(path)
        log_path = path

    args.log_dir_name = log_path

    # Display Options.
    log_str = ("ResNet{size} Training.\n"
               " Dataset {dataset}\n"
               " Num IPUs {num_ipus}\n"
               " Precision {precision}\n"
               " Num Workers {num_workers}\n"
               " Stochastic Rounding {prng}\n"
               " Floating Point Exceptions {fp_exceptions}\n\n"
               " Batch Size {batch_size}.\n"
               " Replication Factor {replication_factor}.\n"
               " Gradient Accumulation Factor {gradient_accumulation_factor}.\n"
               " Effective batch size (number of samples evaluated in one weight update: {effective_batch_size}.\n"
               " Batches Per Step {batches_per_step}.\n"
               " Epochs {epochs}\n"
               " Weight Decay {weight_decay}\n"
               " Base Learning Rate 2^{base_learning_rate}\n"
               " Learning Rate {learning_rate}\n"
               " Learning Rate Schedule {learning_rate_schedule}\n")

    print(log_str.format(**vars(args)))

    with open(os.path.join(args.log_dir_name, "log_str.txt"), "w+") as f:
        log_lines = str(args).split(',')
        for line in log_lines:
            f.write(line)
        f.write('\n')
        f.write(log_str.format(**vars(args)))

    # ResNext uses batch norm. The issue below affects replicated ResNext training
    if args.replication_factor > 1 and args.norm_type == "BATCH":
        print("Using batch normalization and replication of graphs is not "
              "fully supported due to mean and variance not being reduced.")

    args.train = True
    args.precision = np.float16 if args.precision == '16' else np.float32

    # Detemine if we are current with the gc-profile tool
    args.gc_profile_log_dir = os.environ.get('GC_PROFILE_LOG_DIR', log_path)

    train_process(args)
