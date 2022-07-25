# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import os
import sys
import json
from tqdm import tqdm
from collections import deque
import logging_util

import conf_utils
import librispeech_data
import conformer_builder
from examples_utils import load_lib

# set up logging
logger = logging_util.get_basic_logger('CONFORMER_TRAIN')


def _get_popart_type(np_type):
    return {
        np.float16: 'FLOAT16',
        np.float32: 'FLOAT'
    }[np_type]


def load_ctc_loss_lib():
    logger.info("Building (if necessary) and loading ctc_loss custom op.")
    cd = os.path.dirname(os.path.abspath(__file__))
    os.environ['CTC_LOSS_CODELET_PATH'] = cd + '/custom_operators/ctc_loss/codelet.cpp'
    load_lib(cd + '/custom_operators/ctc_loss/ctc_loss.cpp')


def create_inputs_for_training(builder, conf):
    """ defines the input tensors for the conformer model """

    inputs = dict()

    inputs["text_input"] = builder.addInputTensor(popart.TensorInfo("UINT32",
                                                                    [conf.samples_per_device,
                                                                     conf.max_text_sequence_length]),
                                                  "text_input")
    inputs["mel_spec_input"] = builder.addInputTensor(popart.TensorInfo(_get_popart_type(conf.precision),
                                                                        [conf.samples_per_device,
                                                                         conf.mel_bands,
                                                                         conf.max_spectrogram_length]),
                                                      "mel_spec_input")
    inputs["input_length"] = builder.addInputTensor(popart.TensorInfo("UINT32", [conf.samples_per_device]),
                                                    "input_length")

    inputs["target_length"] = builder.addInputTensor(popart.TensorInfo("UINT32", [conf.samples_per_device]),
                                                     "target_length")

    return inputs


def create_model_and_dataflow_for_training(builder, conf, inputs):
    """ builds the conformer model, loss function and dataflow for training """

    conformer_encoder = conformer_builder.ConformerEncoder(builder,
                                                           input_dim=conf.mel_bands,
                                                           sequence_length=conf.max_spectrogram_length,
                                                           encoder_dim=conf.encoder_dim,
                                                           attention_heads=conf.attention_heads,
                                                           encoder_layers_per_stage=conf.encoder_layers_per_stage,
                                                           dropout_rate=conf.dropout_rate,
                                                           use_conv_module=conf.use_conv_module,
                                                           cnn_module_kernel=conf.kernel_size,
                                                           subsampling_factor=conf.subsampling_factor,
                                                           dtype=conf.precision)

    conformer_decoder = conformer_builder.ConformerDecoder(builder,
                                                           encoder_dim=conf.encoder_dim,
                                                           num_symbols=conf.num_symbols,
                                                           dtype=conf.precision)

    encoder_output = conformer_encoder(inputs["mel_spec_input"])

    # CTC layer is placed on last pipelining stage
    with builder.virtualGraph(conf.num_pipeline_stages - 1):

        decoder_output = conformer_decoder(encoder_output)

        ctc_outputs = builder.customOp(opName="CtcLoss",
                                       opVersion=1,
                                       domain="com.acme",
                                       inputs=[decoder_output,
                                               inputs["text_input"],
                                               inputs["input_length"],
                                               inputs["target_length"]],
                                       attributes={"blank": 0, "reduction": int(popart.ReductionType.Mean)},
                                       numOutputs=4)

        ctc_neg_log_likelihood = ctc_outputs[0]

    anchor_types_dict = {
        ctc_neg_log_likelihood: popart.AnchorReturnType("ALL"),
    }

    proto = builder.getModelProto()
    dataflow = popart.DataFlow(conf.device_iterations, anchor_types_dict)

    return proto, ctc_neg_log_likelihood, dataflow


def update_and_reset_loss_data(ctc_loss_data, model_dir):
    """ writes latest loss value to file and resets list """

    out_filename = os.path.join(model_dir, 'ctc_losses.txt')
    with open(out_filename, 'a') as f:
        f.write(str(np.mean(ctc_loss_data)) + '\n')
    logger.info('Current CTC loss: ' + str(np.mean(ctc_loss_data)))
    ctc_loss_data.clear()
    return


if __name__ == '__main__':

    logger.info("Conformer Training in Popart")

    parser = conf_utils.add_conf_args(run_mode='training')
    conf = conf_utils.get_conf(parser)
    session_options = conf_utils.get_session_options(conf)
    device = conf_utils.get_device(conf)

    # setting numpy seed
    np.random.seed(1222)

    load_ctc_loss_lib()

    if not os.path.exists(conf.model_dir):
        logger.info("Creating model directory {}".format(conf.model_dir))
        os.makedirs(conf.model_dir)
    conf_path = os.path.join(conf.model_dir, "model_conf.json")
    logger.info("Saving model configuration params to {}".format(conf_path))
    with open(conf_path, 'w') as f:
        json.dump(conf_utils.serialize_model_conf(conf), f,
                  sort_keys=True, indent=4)

    # building model and dataflow
    builder = popart.Builder()
    conformer_model_inputs = create_inputs_for_training(builder, conf)

    proto, ctc_neg_log_likelihood, dataflow = create_model_and_dataflow_for_training(builder,
                                                                                     conf,
                                                                                     conformer_model_inputs)

    # create optimizer
    if conf.optimizer == 'SGD':
        optimizer_dict = {"defaultLearningRate": (conf.init_lr, False),
                          "defaultWeightDecay": (0, True)}
        logger.info("Creating SGD optimizer: {}".format(json.dumps(optimizer_dict)))
        optimizer = popart.SGD(optimizer_dict)
    elif conf.optimizer == 'Adam':
        optimizer_dict = {
            "defaultLearningRate": (conf.init_lr, True),
            "defaultBeta1": (conf.beta1, True),
            "defaultBeta2": (conf.beta2, True),
            "defaultWeightDecay": (0.0, True),
            "defaultEps": (conf.adam_eps, True),
            "lossScaling": (1.0, True),
        }
        logger.info("Creating Adam optimizer: {}".format(json.dumps(optimizer_dict)))
        optimizer = popart.Adam(optimizer_dict)
    else:
        logger.info("Not a valid optimizer option: {}".format(conf.optimizer))
        sys.exit(-1)

    # create training session
    logger.info("Creating the training session")
    training_session, anchors = \
        conf_utils.create_session_anchors(proto,
                                          ctc_neg_log_likelihood,
                                          device,
                                          dataflow,
                                          session_options,
                                          training=True,
                                          optimizer=optimizer)
    logger.info("Sending weights from Host")
    training_session.weightsFromHost()
    training_session.setRandomSeed(1222)

    logger.info("Preparing LibriSpeech dataset")
    dataset = librispeech_data.LibriSpeechDataset(conf)
    logger.info("Number of clips in {} for training: {}".format(conf.dataset, len(dataset)))

    if not conf.no_pre_load_data:
        logger.info("Loading full training dataset into memory (this may take a few minutes)")
        all_step_data = dataset.load_all_step_data()

    ctc_loss_data = deque(maxlen=dataset.num_steps)

    for epoch in range(conf.num_epochs):

        if not conf.no_pre_load_data:
            tqdm_iter = tqdm(all_step_data, disable=not sys.stdout.isatty())
        else:
            dataset_iterator = dataset.get_step_data_iterator()
            tqdm_iter = tqdm(dataset_iterator, disable=not sys.stdout.isatty())

        for mel_spec_data, text_data, ctc_input_length_data, ctc_target_length_data in tqdm_iter:

            stepio = popart.PyStepIO(
                {
                    conformer_model_inputs["text_input"]: text_data,
                    conformer_model_inputs["mel_spec_input"]: mel_spec_data,
                    conformer_model_inputs["input_length"]: ctc_input_length_data,
                    conformer_model_inputs["target_length"]: ctc_target_length_data,
                }, anchors)

            training_session.run(stepio)

            ctc_loss_data.append(np.copy(anchors[ctc_neg_log_likelihood]))

            tqdm_iter.set_description("CTC loss: " + str(np.mean(ctc_loss_data)), refresh=sys.stdout.isatty())

        logger.info("Completed Epoch # %d / %d" % (epoch + 1, conf.num_epochs))
        update_and_reset_loss_data(ctc_loss_data, conf.model_dir)
        if (epoch + 1) % conf.checkpoint_interval == 0:
            # Saving current model to checkpoint
            ckpt_filename = os.path.join(conf.model_dir, 'checkpoint_{}.onnx'.format(epoch + 1))
            logger.info('Saving model to {}'.format(ckpt_filename))
            training_session.modelToHost(ckpt_filename)
