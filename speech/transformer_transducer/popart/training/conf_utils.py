# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import numpy as np
import popart
import json
import yaml
import os
import sys

import logging_util
import popdist
import popdist.popart

# set up logging
logger = logging_util.get_basic_logger(__name__)


def add_conf_args(run_mode):
    """ define the argument parser object """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-conf-file', type=str, required=True,
                        help='Path to model configuration yaml file')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to save model checkpoints during training' if run_mode == 'training' else
                             'Path to onnx model file to be used for validation')
    parser.add_argument('--wandb', action="store_true", default=False,
                        help="Enabling logging to Weights and Biases")
    parser.add_argument('--wandb_entity', type=str,
                        required='--wandb' in sys.argv, help='Weights and Biases entity')
    parser.add_argument('--wandb_run_name', type=str,
                        required='--wandb' in sys.argv, help='Weights and biases run name')
    if run_mode == 'training':
        parser.add_argument('--batch-size', type=int, default=32,
                            help="Batch-size for training")
        parser.add_argument('--device-iterations', type=int, default=1,
                            help="Number of iterations run on the device before syncing with the host.")
        parser.add_argument('--gradient-accumulation-factor', type=int, default=32,
                            help="gradient accumulation factor")
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'LAMB'],
                            default='LAMB', help='choose which optimizer to use')
        parser.add_argument('--base-lr', default=4e-3,
                            type=float, help='Base learning rate')
        parser.add_argument('--min-lr', default=1e-5,
                            type=float, help='minimum learning rate')
        parser.add_argument("--lr-exp-gamma", default=0.935, type=float,
                            help='gamma factor for exponential lr scheduler')
        parser.add_argument('--num-epochs', type=int, default=100,
                            help="Number of t raining epochs")
        parser.add_argument('--num-steps', type=int, default=None,
                            help="Force a fixed number of steps to be run rather than fixed epochs")
        parser.add_argument('--start-checkpoint-dir', type=str, required=False,
                            help='Path to model checkpoint to start training from')
        parser.add_argument('--start-epoch', type=int, default=0,
                            help="Start epoch. Start checkpoint should exist if start-epoch > 0")
        parser.add_argument("--warmup-epochs", default=6, type=int,
                            help='initial number of epochs of increasing learning rate')
        parser.add_argument("--hold-epochs", default=40, type=int,
                            help='number of epochs of constant learning rate after warmup')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='Beta 1 for LAMB optimizer')
        parser.add_argument('--beta2', default=0.999,
                            type=float, help='Beta 2 for LAMB optimizer')
        parser.add_argument('--max-weight-norm', default=10.0,
                            type=float, help='Max weight norm for LAMB optimizer')
        parser.add_argument('--enable-ema-weights', action="store_true", default=False,
                            help="whether to enable enable exponential moving averages of weights for checkpointing")
        parser.add_argument('--ema-factor', type=float, default=0.999,
                            help='Discount factor for exp averaging of model weights')
        parser.add_argument("--gradient-clipping-norm", type=float, default=None,
                            help="Set the gradient clipping norm in the Lamb optimizer. Default is no clipping")
        parser.add_argument('--weight-decay', default=1e-3,
                            type=float, help='Weight decay for the optimizer')
        parser.add_argument('--loss-scaling', default=512.0,
                            type=float, help='Loss scaling')
        parser.add_argument('--num-buckets', type=int, default=1,
                            help='If provided, samples will be grouped by audio duration, '
                                 'to this number of buckets, for each bucket, '
                                 'random samples are batched, and finally '
                                 'all batches are randomly shuffled')
        parser.add_argument('--enable-stochastic-rounding', action="store_true", default=False,
                            help="whether to enable stochastic rounding on device")
        parser.add_argument('--num-lstm-shards', type=int, default=4,
                            help="number of LSTM shards for training")
        parser.add_argument('--generated-data', action="store_true", default=False,
                            help="whether to use generated data for training benchmarking")
        parser.add_argument('--do-validation', action="store_true", default=False,
                            help="whether to run validation")
        parser.add_argument('--epoch-to-start-validation', type=int, default=0,
                            help="Epoch number from which we start validation process.")
        parser.add_argument('--do-batch-serialization-joint-net', action="store_true", default=False,
                            help="whether to do batch serialization for the Joint Network")
        parser.add_argument('--joint-net-batch-split-size', type=int, default=1,
                            help="size of split along batch dimension for JointNet batch serialization")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--replication-factor', type=int, default=16,
                        help="Replication factor for data parallel " + run_mode)
    parser.add_argument('--enable-half-partials', action="store_true", default=False,
                        help="whether to enable half partials for matmuls")
    parser.add_argument('--enable-lstm-half-partials', action="store_true", default=False,
                        help="whether to enable half partials for LSTM layers")
    parser.add_argument('--max-duration', type=float, default=16.8,
                        help='Discard samples longer than max-duration')
    parser.add_argument('--max-symbols-per-step', type=int, default=300,
                        help='Maximum number of symbols per step for validation')
    parser.add_argument('--joint-net-split-size', type=int, default=15,
                        help='The split size of joint network along the audio-frame dimension')
    parser.add_argument('--fp-exceptions', action="store_true", default=False,
                        help="Enable floating point exception")
    parser.add_argument('--use-ipu-model', action="store_true",
                        help="Run the program on the IPU Model")
    parser.add_argument("--device-id", type=int, default=None,
                        help="Select a specific IPU device.")
    parser.add_argument("--device-connection-type", type=str, default="always",
                        choices=["always", "ondemand", "offline"],
                        help="Set the popart.DeviceConnectionType.")
    parser.add_argument("--device-version", type=str, default=None,
                        help="Set the IPU version (for offline compilation).")
    parser.add_argument("--device-tiles", type=int, default=None,
                        help="Set the number of tiles (for offline compilation).")
    parser.add_argument("--device-ondemand-timeout", type=int, default=int(1e4),
                        help="Set the seconds to wait for an ondemand device to become before available before exiting.")
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help="Batch-size for validation")
    parser.add_argument('--val-device-iterations', type=int, default=1,
                        help="Number of iterations run on the device before syncing with the host for validation")
    parser.add_argument('--val-num-lstm-shards', type=int, default=1,
                        help="number of LSTM shards for validation")

    return parser


def get_conf(parser):
    """ parse the arguments and set the model configuration parameters """
    conf = parser.parse_args()

    # make paths absolute
    wd = os.path.dirname(__file__)
    conf.model_dir = os.path.join(wd, conf.model_dir)
    conf.data_dir = os.path.join(wd, conf.data_dir)
    if hasattr(conf, "start_checkpoint_dir") and conf.start_checkpoint_dir is not None:
        conf.start_checkpoint_dir = os.path.join(wd, conf.start_checkpoint_dir)

    set_model_conf(conf)

    return conf


def set_model_conf(conf, print_model_conf=True):
    """ set the model configuration parameters """

    model_conf_path = conf.model_conf_file
    logger.info("Loading model configuration from {}".format(model_conf_path))
    with open(model_conf_path, 'r') as f:
        conf.model_conf = yaml.safe_load(f)

    if print_model_conf:
        logger.info("Model configuration params:")
        logger.info(json.dumps(vars(conf),
                               sort_keys=True, indent=4))

    return conf


def get_session_options(opts):
    """ get popart session options """

    # Create a session to compile and execute the graph
    options = popart.SessionOptions()

    options.enableStochasticRounding = opts.enable_stochastic_rounding
    partials_type = "half" if opts.enable_half_partials else "float"
    options.partialsTypeMatMuls = partials_type

    options.engineOptions = {
        "debug.allowOutOfMemory": "true"
    }

    options.lstmOptions = {"numShards": str(opts.num_lstm_shards),
                           "partialsType": "half" if opts.enable_lstm_half_partials else "float",
                           "rnnStepsPerWU": "1"}

    # Enable the reporting of variables in the summary report
    options.reportOptions = {'showVarStorage': 'true'}

    if opts.fp_exceptions:
        # Enable exception on floating point errors
        options.enableFloatingPointChecks = True

    # Need to disable constant weights so they can be set before
    # executing the inference session
    options.constantWeights = False

    if opts.local_replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = opts.local_replication_factor

        # Enable merge updates
        # options.mergeVarUpdate = popart.MergeVarUpdateType.AutoLoose
        # disabling merge pattern so that graph builds for lamb/pipelining/replication/offchip
        options.mergeVarUpdate = popart.MergeVarUpdateType.Off
        options.mergeVarUpdateMemThreshold = 6000000

    if opts.training and opts.gradient_accumulation_factor > 1:
        options.enableGradientAccumulation = True
        options.accumulationFactor = opts.gradient_accumulation_factor

    options.optimizerStateTensorLocationSettings.location.storage = popart.TensorStorage.OffChip
    options.optimizerStateTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On

    options.enableOutlining = True
    options.outlineThreshold = -np.inf
    options.enableOutliningCopyCostPruning = False

    # this is required for batch-serialization to work
    options.explicitRecomputation = True

    if opts.use_popdist:
        popdist.popart.configureSessionOptions(options)

    return options


def create_session_anchors(proto, loss, device, dataFlow,
                           options, training, optimizer=None, use_popdist=False):
    """ Create the desired session and compile the graph """

    if training:
        session_type = "training"
        session_kwargs = dict(
            fnModel=proto,
            loss=loss,
            deviceInfo=device,
            optimizer=optimizer,
            dataFlow=dataFlow,
            userOptions=options
        )
    else:
        session_type = "inference"
        session_kwargs = dict(
            fnModel=proto,
            deviceInfo=device,
            dataFlow=dataFlow,
            userOptions=options
        )
    if training:
        if use_popdist:
            hvd = try_import_horovod()
            session = hvd.DistributedTrainingSession(
                **session_kwargs, enableEngineCaching=False)
        else:
            session = popart.TrainingSession(**session_kwargs)
    else:
        session = popart.InferenceSession(**session_kwargs)
    try:
        logger.info("Preparing the {} graph".format(session_type))
        session.prepareDevice()
        logger.info("{0} graph preparation complete.".format(
            session_type.capitalize(),))
    except popart.OutOfMemoryException as e:
        logger.warn("Caught OutOfMemoryException during prepareDevice")
        raise

    if training and use_popdist:
        # make sure to broadcast weights when using popdist/poprun
        hvd.broadcast_weights(session)

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    return session, anchors


def try_import_horovod():
    try:
        import horovod.popart as hvd
        hvd.init()
    except ImportError:
        raise ImportError("Could not find the PopART horovod extension. "
                          "Please install the horovod .whl provided in the Poplar SDK.")
    return hvd


def set_popdist_args(args):
    if not popdist.isPopdistEnvSet():
        logger.info("No PopRun detected. Using single instance training")
    else:
        logger.info("PopRun is detected")

        args.use_popdist = True
        num_total_replicas = popdist.getNumTotalReplicas()
        args.local_replication_factor = popdist.getNumLocalReplicas()
        args.num_instances = popdist.getNumInstances()
        assert(num_total_replicas ==
               args.local_replication_factor * args.num_instances)
        args.instance_idx = popdist.getInstanceIndex()

        if args.replication_factor != num_total_replicas:
            raise RuntimeError(f"Replication factor({args.replication_factor}) "
                               f"should match popdist replication factor ({num_total_replicas})")

        if args.samples_per_step % args.num_instances != 0:
            raise RuntimeError(f"The number of samples per step({args.samples_per_step}) "
                               f"has to be a integer multiple of the number of instances({args.num_instances})")


class RunTimeConf(object):
    """ Runtime Conf object that encapsulates various params required for running on IPU-POD systems """

    def __init__(self, conf, run_mode):
        self.data_dir = conf.data_dir
        # this is set to False by default and may be updated in set_popdist_args
        self.use_popdist = False
        self.num_instances = 1  # may be updated in set_popdist_args
        self.instance_idx = 0  # may be updated in set_popdist_args
        self.replication_factor = conf.replication_factor
        # may be updated in set_popdist_args
        self.local_replication_factor = conf.replication_factor
        self.precision = np.float16
        self.fp_exceptions = conf.fp_exceptions
        self.enable_half_partials = conf.enable_half_partials
        self.enable_lstm_half_partials = conf.enable_lstm_half_partials
        if run_mode == "training":
            self.training = True
            self.batch_size = conf.batch_size
            if self.batch_size % self.replication_factor != 0:
                raise RuntimeError(
                    f"Training batch size({self.batch_size}) has to be a integer multiple "
                    f"of the replication factor({self.replication_factor})")
            self.device_iterations = conf.device_iterations
            self.gradient_accumulation_factor = conf.gradient_accumulation_factor
            self.samples_per_device = self.batch_size // self.replication_factor
            self.samples_per_step = self.batch_size * \
                self.device_iterations * self.gradient_accumulation_factor
            self.num_epochs = conf.num_epochs
            self.num_buckets = conf.num_buckets
            self.enable_stochastic_rounding = conf.enable_stochastic_rounding
            self.num_lstm_shards = conf.num_lstm_shards
            self.joint_net_split_size = conf.joint_net_split_size
            self.enable_ema_weights = conf.enable_ema_weights
            self.ema_factor = conf.ema_factor
            self.do_batch_serialization_joint_net = conf.do_batch_serialization_joint_net
            self.joint_net_batch_split_size = conf.joint_net_batch_split_size
        elif run_mode == "validation":
            self.training = False
            self.batch_size = conf.val_batch_size
            if self.batch_size % self.replication_factor != 0:
                raise RuntimeError(
                    f"Validation batch size({self.batch_size}) has to be a integer multiple "
                    f"of the replication factor({self.replication_factor})")
            self.device_iterations = conf.val_device_iterations
            self.samples_per_device = self.batch_size // self.replication_factor
            self.samples_per_step = self.batch_size * self.device_iterations
            self.enable_stochastic_rounding = False
            self.num_lstm_shards = conf.val_num_lstm_shards
        else:
            raise RuntimeError(f"Not a valid run_mode: {run_mode}")

        # have to set popdist related variables
        set_popdist_args(self)
        return
