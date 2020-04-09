# Copyright 2019 Graphcore Ltd.
import time
import argparse
import os

import popart
import popart.torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model


def _get_torch_type(np_type):
    return {
        np.float16: 'torch.HalfTensor',
        np.float32: 'torch.FloatTensor'
    }[np_type]


def get_dataset(opts):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Train
    trainset = torchvision.datasets.CIFAR10(root=opts.data_dir,
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=opts.batch_size *
                                              opts.batches_per_step,
                                              shuffle=(not opts.no_shuffle),
                                              num_workers=0,
                                              drop_last=True)

    # Test
    testset = torchvision.datasets.CIFAR10(root=opts.data_dir,
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=opts.batch_size *
                                             opts.batches_per_step,
                                             shuffle=(not opts.no_shuffle),
                                             num_workers=0,
                                             drop_last=True)

    return trainset, testset, trainloader, testloader


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


def get_options(opts):

    # Create a session to compile and execute the graph
    options = popart.SessionOptions()

    options.engineOptions = {"debug.allowOutOfMemory": "true"}

    # Enable the reporting of variables in the summary report
    options.reportOptions = {'showVarStorage': 'true'}

    if opts.fp_exceptions:
        # Enable exception on floating point errors
        options.enableFloatingPointChecks = True

    if not opts.no_prng:
        options.enableStochasticRounding = True

    # Need to disable constant weights so they can be set before
    # executing the inference session
    options.constantWeights = False

    # Enable recomputation
    if opts.recompute:
        options.autoRecomputation = popart.RecomputationType.Standard

    # Enable auto-sharding
    if opts.num_ipus > 1 and opts.num_ipus > opts.replication_factor:
        options.enableVirtualGraphs = True
        options.virtualGraphMode = popart.VirtualGraphMode.Auto

    # Enable pipelining
    if opts.pipeline:
        options.enablePipelining = True

    if opts.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = opts.replication_factor

        # Enable merge updates
        options.mergeVarUpdate = popart.MergeVarUpdateType.AutoLoose
        options.mergeVarUpdateMemThreshold = 6000000

    return options


def train_process(opts):
    net = getattr(model, opts.model_name)(
        pretrained=False,
        progress=True,
        num_classes=10 if opts.dataset == "CIFAR-10" else 1000)

    # Models are missing a softmax layer to work with our NllLoss,
    # so we just add one on.
    net = nn.Sequential(net, nn.Softmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=opts.learning_rate,
                          momentum=opts.momentum,
                          weight_decay=opts.weight_decay)

    trainset, testset, trainloader, testloader = get_dataset(opts)

    inputs, labels = iter(trainloader).next()

    sessionOpts = get_options(opts)

    patterns = popart.Patterns()
    patterns.InPlace = opts.no_inplacing

    start = time.time()
    # Pass all the pytorch stuff to the session
    torchSession = popart.torch.TrainingSession(
        torchModel=net,
        inputs=inputs,
        targets=labels,
        optimizer=optimizer,
        losses=criterion,
        batch_size=opts.batch_size,
        batches_per_step=opts.batches_per_step,
        deviceInfo=get_device(opts.num_ipus, opts.simulation),
        userOptions=sessionOpts,
        passes=patterns)
    print("Converting pytorch model took {:.2f}s".format(time.time() - start))

    # Prepare for training.
    start = time.time()
    print("Compiling model...")
    anchors = torchSession.initAnchorArrays()

    torchSession.prepareDevice()
    torchSession.optimizerFromHost()
    torchSession.weightsFromHost()

    torchSession.setRandomSeed(0)
    print("Compiling popart model took {:.2f}s".format(time.time() - start))
    for epoch in range(opts.epochs):  # loop over the dataset multiple times
        run_training(opts, epoch, torchSession, trainloader, trainset, anchors)
        if (not opts.no_validation) and ((epoch + 1) %
                                         opts.valid_per_epoch == 0):
            run_validation(opts, epoch, torchSession, testloader, testset)

    print('Finished Training')
    # Save the popart training report
    if opts.gc_profile_log_dir is not None:
        from gcprofile import save_popart_report
        save_popart_report(torchSession)


def run_training(opts, epoch, torchSession, trainloader, trainset, anchors):
    start_time = time.time()

    running_loss = 0.0
    running_accuracy = 0
    print("#" * 20, "Train phase:", "#" * 20)
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        torchSession.run(inputs, labels)
        running_loss += np.mean(anchors["loss_0"])

        progress = (opts.bar_width * (i + 1) * opts.batch_size *
                    opts.batches_per_step) // len(trainset)
        if not opts.no_progress_bar:
            print(
                f'\repoch {epoch + 1} [{progress * "."}{(opts.bar_width - progress) * " "}]  ',
                end='')

        results = np.argmax(
            anchors['output_0'].reshape([
                opts.batches_per_step * opts.batch_size,
                10 if opts.dataset == "CIFAR-10" else 1000
            ]), 1)
        num_correct = np.sum(results == anchors['target_0'].reshape(
            [opts.batches_per_step * opts.batch_size]))
        running_accuracy += num_correct
    end_time = time.time()
    print()
    print("   Accuracy={0:.2f}%".format(running_accuracy * 100 /
                                        len(trainset)))
    print("   Loss={0:.4f}".format(running_loss / (i + 1)))
    print("Images per second: {:.0f}".format(
        len(trainset) / (end_time - start_time)))


def run_validation(opts, epoch, torchSession, testloader, testset):

    # Save the model with weights
    onnx_path = os.path.join(opts.data_dir, opts.onnx_model_name)
    torchSession.modelToHost(onnx_path)

    inferenceOpts = get_options(opts)
    inferenceOpts.constantWeights = False

    # Pytorch currently doesn't support importing from onnx:
    # https://github.com/pytorch/pytorch/issues/21683
    # And pytorch->onnx->caffe2 is broken:
    # https://github.com/onnx/onnx/issues/2463
    # So we import into popart session and infer.
    # Alternatively, use any other ONNX compatible runtime.
    builder = popart.Builder(onnx_path)
    inferenceSession = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFeed=popart.DataFlow(opts.batches_per_step,
                                 {"output_0": popart.AnchorReturnType("ALL")}),
        deviceInfo=get_device(opts.num_ipus, opts.simulation),
        userOptions=inferenceOpts)

    print("Compiling test model...")
    inferenceSession.prepareDevice()

    inferenceSession.weightsFromHost()
    inferenceAnchors = inferenceSession.initAnchorArrays()
    print("#" * 20, "Test phase:", "#" * 20)
    test_accuracy = 0
    for j, data in enumerate(testloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        stepio = popart.PyStepIO({"input_0": inputs.data.numpy()},
                                 inferenceAnchors)

        inferenceSession.run(stepio)

        progress = (opts.bar_width * (j + 1) * opts.batch_size *
                    opts.batches_per_step) // len(testset)

        if not opts.no_progress_bar:
            print(
                f'\rtest epoch {epoch + 1} [{progress * "."}{(opts.bar_width - progress) * " "}]  ',
                end='')

        results = np.argmax(
            inferenceAnchors['output_0'].reshape(
                [opts.batches_per_step * opts.batch_size, 10]), 1)
        num_correct = np.sum(results == labels.data.numpy().reshape(
            [opts.batches_per_step * opts.batch_size]))
        test_accuracy += num_correct
    inferenceSession = None
    print()
    print("Accuracy: {}%".format(test_accuracy * 100 / len(testset)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='TorchVision training in Popart',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # -------------- DATASET ------------------
    group = parser.add_argument_group('Dataset')
    group.add_argument('--dataset',
                       type=str,
                       choices=["IMAGENET", "CIFAR-10"],
                       default="CIFAR-10",
                       help="Choose which dataset to run on")
    group.add_argument('--data-dir',
                       type=str,
                       default="/tmp/data/",
                       help="Path to data")
    group.add_argument('--onnx-model-name',
                       type=str,
                       default="torchModel.onnx",
                       help="ONNX model save name")
    group.add_argument('--no-shuffle',
                       action="store_true",
                       help="Disable shuffling on the dataset.")

    # -------------- MODEL ------------------
    group = parser.add_argument_group('Model')
    group.add_argument('--model-name',
                       type=str,
                       default="resnet18",
                       help='Model name e.g. resnet18',
                       choices=["resnet18", "resnet34", "resnet50"])
    group.add_argument('--batch-size',
                       type=int,
                       default=4,
                       help='Set batch size. '
                       'This must be a multiple of the replication-factor')
    group.add_argument('--no-prng',
                       action="store_true",
                       help="Single flag to disable Stochastic Rounding")
    group.add_argument('--fp-exceptions',
                       action="store_true",
                       default=False,
                       help="Enable floating point exception")
    group.add_argument('--no-inplacing',
                       action="store_false",
                       help="Disable inplacing")

    # -------------- TRAINING ------------------
    group = parser.add_argument_group('Training')

    group.add_argument('--learning-rate',
                       type=int,
                       default=-6,
                       help="Learning rate exponent (2**N). blr = lr /  bs")
    group.add_argument('--epochs',
                       type=int,
                       default=30,
                       help="Number of training epochs")
    group.add_argument('--no-validation',
                       action="store_true",
                       help="Dont do any validation runs.")
    group.add_argument('--valid-per-epoch',
                       type=int,
                       default=10,
                       help="Validation steps per epoch.")
    group.add_argument(
        '--weight-decay',
        type=float,
        default=0,
        help="Value for weight decay bias, setting to 0 removes weight decay.")
    group.add_argument(
        '--momentum',
        type=float,
        default=0,
        help="Value for momentum, setting to 0 removes momentum.")
    group.add_argument('--num-ipus',
                       type=int,
                       default=1,
                       help="Number of IPU's")
    group.add_argument(
        '--replication-factor',
        type=int,
        default=1,
        help="Number of times to replicate the graph to perform data parallel"
        " training. Must be a factor of the number of IPUs")
    group.add_argument(
        '--recompute',
        action="store_true",
        default=False,
        help="Enable recomputations of activations in backward pass")
    group.add_argument('--pipeline',
                       action="store_true",
                       help="Pipeline the model over IPUs")
    group.add_argument(
        '--batches-per-step',
        type=int,
        default=100,
        help="How many minibatches to perform on the device before returning"
        "to the host.")
    group.add_argument('--simulation',
                       action="store_true",
                       help="Run the program on the IPU Model")
    group.add_argument('--no-progress-bar',
                       action="store_true",
                       help="Don't show the epoch progress bar")
    group.add_argument('--bar-width',
                       type=int,
                       default=20,
                       help="Progress bar width")

    args = parser.parse_args()

    args.learning_rate = (2**args.learning_rate) * (args.batch_size)

    if (args.batch_size % args.replication_factor) != 0:
        raise Exception("Invalid Argument : Batch size ({}) must be a "
                        "multiple of replication factor ({})".format(
                            args.batch_size, args.replication_factor))

    if (args.num_ipus % args.replication_factor) != 0:
        raise Exception("Invalid Argument : Number of IPUs ({}) must be a "
                        "multiple of replication factor ({})".format(
                            args.num_ipus, args.replication_factor))

    # The number of samples that the device will process currently
    args.samples_per_device = (int)(args.batch_size / args.replication_factor)

    # Display Options.
    log_str = ("{model_name} Training.\n"
               " Dataset {dataset}\n"
               " Num IPUs {num_ipus}\n"
               " Disable Stochastic Rounding {no_prng}\n"
               " Floating Point Exceptions {fp_exceptions}\n"
               " Epochs per validation {valid_per_epoch}\n"
               " Pipelining {pipeline}\n"
               "Training Graph.\n"
               " Batch Size {batch_size}.\n"
               " Batches Per Step {batches_per_step}.\n"
               " Replication Factor {replication_factor}.\n"
               " Epochs {epochs}\n"
               " Momentum {momentum}\n"
               " Weight Decay {weight_decay}\n"
               " Learning Rate {learning_rate}\n"
               "Validation Graph.\n")

    print(log_str.format(**vars(args)))

    args.train = True

    # Determine if we are current with the gc-profile tool
    args.gc_profile_log_dir = os.environ.get('GC_PROFILE_LOG_DIR', None)

    train_process(args)
