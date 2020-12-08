# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import numpy as np
import popart
import json
import os
import copy
import logging_util

import text_utils

# set up logging
logger = logging_util.get_basic_logger(__name__)


def add_conf_args(run_mode):
    """ define the argument parser object """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Set batch size for training.')
    parser.add_argument('--batch_size_for_inference', type=int, default=12,
                        help='Set batch size for inference.')
    parser.add_argument('--dataset', type=str, choices=['VCTK'],
                        default='VCTK',
                        help='Choose which dataset to process')
    parser.add_argument('--no_pre_load_data', action="store_true", default=False,
                        help="do not pre-load the full data-set into memory")
    if run_mode == 'training':
        parser.add_argument('--model_dir', type=str, required=True,
                            help='Path to save model checkpoints during training')
        parser.add_argument('--data_dir', type=str, required=True,
                            help='Path to data')
    elif run_mode in ['autoregressive_synthesis', 'prep_autoregressive_graph']:
        parser.add_argument('--inference_model_dir', type=str, required=True,
                            help='Path to directory where inference model is saved')
    if run_mode in ['prep_autoregressive_graph', 'non_autoregressive_synthesis']:
        parser.add_argument('--trained_model_file', type=str, required=True,
                            help='Path to onnx file for trained model')
    if 'synthesis' in run_mode:  # autoregressive or non-autoregressive
        parser.add_argument('--sentence', type=str, required=True,
                            help='Text to synthesize speech')
        parser.add_argument('--results_path', type=str, required=True,
                            help='Path to save results files')
    parser.add_argument('--batches_per_step', type=int, default=50,
                        help="How many mini-batches to perform on the device before returning to the host.")
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help="Number of training epochs")
    parser.add_argument('--init_lr', type=float, default=0.05,
                        help="Initial learning rate")
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help="How many epochs to complete before checkpointing")
    parser.add_argument('--validation_interval', type=int, default=10,
                        help="How many epochs to complete before running validation")
    parser.add_argument('--not_multi_thread_dataloader', action="store_true", default=False,
                        help="Disable multi threaded data loading")
    parser.add_argument('--num_threads', type=int, default=32,
                        help="The number of threads to be used to load data")
    parser.add_argument('--replication_factor', type=int, default=1,
                        help="Number of times to replicate the graph to perform data parallel "
                             "training or inference. Must be a factor of the number of IPUs")
    parser.add_argument('--simulation', action="store_true",
                        help="Run the program on the IPU Model")
    parser.add_argument('--select_ipu', type=str, default="AUTO",
                        help="Select IPU: either AUTO or a valid IPU ID")
    parser.add_argument('--num_ipus', type=int, default=1,
                        help="Number of IPUs")
    parser.add_argument('--recompute', action="store_true", default=False,
                        help="Enable recomputations of activations in backward pass")
    parser.add_argument('--prng', action="store_true", default=True,
                        help="Enable Stochastic Rounding")
    parser.add_argument('--fp_exceptions', action="store_true", default=False,
                        help="Enable floating point exception")
    parser.add_argument('--profile', action="store_true", default=False,
                        help="Perform gc-profile. Profile is generated only when compilation fails")
    parser.add_argument('--no_validation', action="store_true",
                        help="Do not do any validation runs.")
    parser.add_argument('--proportion_train_set', type=float, default=0.80,
                        help="Proportion of training set [0.0-1.0]")
    parser.add_argument('--generated_data', action="store_true", default=False,
                        help="Enable random data generation for benchmarking")

    return parser


def get_conf(parser):
    """ parse the arguments and set the model configuration parameters """
    conf = parser.parse_args()
    # For the deep-voice model, numerical stability issues were observed with FP16
    # (hence we don't support FP16)
    conf.precision = np.float32

    if conf.select_ipu != 'AUTO':
        conf.select_ipu = int(conf.select_ipu)

    # The number of samples that each device will process (for training)
    conf.samples_per_device = int(conf.batch_size / conf.replication_factor)
    # The number of samples that each device will process (for inference)
    conf.samples_per_device_for_inference = int(conf.batch_size_for_inference / conf.replication_factor)

    set_model_conf(conf)

    return conf


def set_model_conf(conf, print_model_conf=True):
    """ set the model configuration parameters """

    if conf.dataset == 'VCTK':
        conf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "vctk_model_conf.json")
        logger.info("Loading model configuration from {}".format(conf_path))
        with open(conf_path) as f:
            model_conf = json.load(f)

    conf.num_symbols = len(text_utils.symbols)
    for k in model_conf.keys():
        setattr(conf, k, model_conf[k])

    conf.max_spectrogram_length = int((conf.max_duration_secs * conf.sample_rate) /
                                      (conf.hop_length * conf.n_frames_per_pred))

    if print_model_conf:
        logger.info("Model configuration params:")
        logger.info(json.dumps(serialize_model_conf(conf),
                               sort_keys=True, indent=4))

    return conf


def serialize_model_conf(conf):
    """ convert configuration object into json serializable object """
    conf_dict = copy.copy(vars(conf))
    conf_dict['precision'] = 32 if conf_dict['precision'] == np.float32 else np.float16

    return conf_dict


def get_device(conf):
    """ Acquire IPU device """
    device_manager = popart.DeviceManager()
    if conf.simulation:
        logger.info("Creating ipu sim")
        ipu_options = {
            "compileIPUCode": True,
            'numIPUs': conf.num_ipus,
            "tilesPerIPU": 1216
        }
        device = device_manager.createIpuModelDevice(ipu_options)
        if device is None:
            raise OSError("Failed to acquire IPU.")
    else:
        logger.info("Acquiring IPU")
        if conf.select_ipu == 'AUTO':
            device = device_manager.acquireAvailableDevice(conf.num_ipus)
        else:
            device = device_manager.acquireDeviceById(conf.select_ipu)
        if device is None:
            raise OSError("Failed to acquire IPU.")
        else:
            logger.info("Acquired IPU: {}".format(device))

    return device


def get_session_options(opts):
    """ get popart session options """

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

    # Enable recomputation
    if opts.recompute:
        options.autoRecomputation = popart.RecomputationType.Standard

    # Enable auto-sharding
    if opts.num_ipus > 1 and opts.num_ipus > opts.replication_factor:
        options.enableVirtualGraphs = True
        options.virtualGraphMode = popart.VirtualGraphMode.Auto

    if opts.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = opts.replication_factor

        # Enable merge updates
        options.mergeVarUpdate = popart.MergeVarUpdateType.AutoLoose
        options.mergeVarUpdateMemThreshold = 6000000

    return options


def create_session_anchors(proto, loss, device, dataFlow,
                           options, training, optimizer=None, profile=False):
    """ Create the desired session and compile the graph """

    if training:
        session_type = "training"
        session = popart.TrainingSession(fnModel=proto,
                                         loss=loss,
                                         deviceInfo=device,
                                         optimizer=optimizer,
                                         dataFlow=dataFlow,
                                         userOptions=options)
    else:
        session_type = "validation"
        session = popart.InferenceSession(fnModel=proto,
                                          deviceInfo=device,
                                          dataFlow=dataFlow,
                                          userOptions=options)

    try:
        logger.info("Preparing the {} graph".format(session_type))
        session.prepareDevice()
        logger.info("{0} graph preparation complete.".format(session_type.capitalize(),))

    except popart.OutOfMemoryException as e:
        logger.warn("Caught Exception while Preparing Device")
        # Dump the profiled result before raising exception and exit
        if profile:
            from gcprofile import save_popart_report
            save_popart_report(session, exception=e)
        raise

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    return session, anchors
