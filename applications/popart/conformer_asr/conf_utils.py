# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import numpy as np
import popart
import json
import copy

import logging_util
import text_utils

# set up logging
logger = logging_util.get_basic_logger(__name__)


def add_conf_args(run_mode):
    """ define the argument parser object """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-conf-file', type=str, required=True,
                        help='Path to model configuration file json')
    if run_mode == 'training':
        parser.add_argument('--model-dir', type=str, required=True,
                            help='Path to save model checkpoints during training')
        parser.add_argument('--data-dir', type=str, required=True,
                            help='Path to data')
        parser.add_argument('--dataset', type=str, choices=['train-clean-100', 'train-clean-360', 'train-other-500'],
                            default='train-clean-100', help='choose which data subset to use for training')
    elif run_mode == 'inference':
        parser.add_argument('--model-file', type=str, required=True,
                            help='Path to onnx model file to be used for inference')
        parser.add_argument('--data-dir', type=str, required=True,
                            help='Path to data')
        parser.add_argument('--results-dir', type=str, required=True,
                            help='Path where inference results are saved')
        parser.add_argument('--dataset', type=str, choices=['test-clean', 'test-other'],
                            default='test-clean', help='choose which data subset to use for testing')
    parser.add_argument('--num-epochs', type=int, default=5000,
                        help="Number of training epochs")
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help="How many epochs to complete before checkpointing")
    parser.add_argument('--no-pre-load-data', action="store_true", default=False,
                        help="do not pre-load the full data-set into memory")
    parser.add_argument('--not-multi-thread-dataloader', action="store_true", default=False,
                        help="Disable multi threaded data loading")
    parser.add_argument('--num-threads', type=int, default=32,
                        help="The number of threads to be used to load data")
    parser.add_argument('--simulation', action="store_true",
                        help="Run the program on the IPU Model")
    parser.add_argument('--select-ipu', type=str, default="AUTO",
                        help="Select IPU: either AUTO or a valid IPU ID")
    parser.add_argument('--fp-exceptions', action="store_true", default=False,
                        help="Enable floating point exception")

    return parser


def get_conf(parser):
    """ parse the arguments and set the model configuration parameters """
    conf = parser.parse_args()

    if conf.select_ipu != 'AUTO':
        conf.select_ipu = int(conf.select_ipu)

    set_model_conf(conf)

    # The number of samples that each MultiIPU device will process
    conf.samples_per_device = int(conf.batch_size / conf.replication_factor)

    return conf


def set_model_conf(conf, print_model_conf=True):
    """ set the model configuration parameters """

    model_conf_path = conf.model_conf_file
    logger.info("Loading model configuration from {}".format(model_conf_path))
    with open(model_conf_path) as f:
        model_conf = json.load(f)

    # number of symbols (this includes the CTC blank symbol)
    conf.num_symbols = len(text_utils.symbols)
    for k in model_conf.keys():
        setattr(conf, k, model_conf[k])

    conf.num_ipus = conf.num_pipeline_stages * conf.replication_factor
    assert(conf.num_pipeline_stages == len(conf.encoder_layers_per_stage))

    # represent precision as numpy data type
    conf.precision = np.float32 if conf.precision == 32 else np.float16

    if print_model_conf:
        logger.info("Model configuration params:")
        logger.info(json.dumps(serialize_model_conf(conf),
                               sort_keys=True, indent=4))

    return conf


def serialize_model_conf(conf):
    """ convert configuration object into json serializable object """
    conf_dict = copy.copy(vars(conf))
    conf_dict['precision'] = 32 if conf_dict['precision'] == np.float32 else 16

    return conf_dict


def get_device(conf):
    """ Acquire IPU device """
    device_manager = popart.DeviceManager()
    if conf.simulation:
        logger.info("Creating ipu sim")
        ipu_options = {
            "compileIPUCode": True,
            'numIPUs': conf.num_ipus,
            "tilesPerIPU": 1472
        }
        device = device_manager.createIpuModelDevice(ipu_options)
        if device is None:
            raise OSError("Failed to create IpuModelDevice.")
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

    if opts.num_pipeline_stages > 1:
        options.enablePipelining = True
        options.virtualGraphMode = popart.VirtualGraphMode.Manual
        options.autoRecomputation = popart.RecomputationType.Pipeline

    options.engineOptions = {
        "debug.allowOutOfMemory": "true"
    }

    # Enable the reporting of variables in the summary report
    options.reportOptions = {'showVarStorage': 'true'}

    if opts.fp_exceptions:
        # Enable exception on floating point errors
        options.enableFloatingPointChecks = True

    # Need to disable constant weights so they can be set before
    # executing the inference session
    options.constantWeights = False

    if opts.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = opts.replication_factor

        # Enable merge updates
        options.mergeVarUpdate = popart.MergeVarUpdateType.AutoLoose
        options.mergeVarUpdateMemThreshold = 6000000

    return options


def create_session_anchors(proto, loss, device, dataFlow,
                           options, training, optimizer=None):
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
        session_type = "inference"
        session = popart.InferenceSession(fnModel=proto,
                                          deviceInfo=device,
                                          dataFlow=dataFlow,
                                          userOptions=options)

    try:
        logger.info("Preparing the {} graph".format(session_type))
        session.prepareDevice()
        logger.info("{0} graph preparation complete.".format(session_type.capitalize(),))
    except popart.OutOfMemoryException as e:
        logger.warn("Caught OutOfMemoryException during prepareDevice")
        raise

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    return session, anchors
