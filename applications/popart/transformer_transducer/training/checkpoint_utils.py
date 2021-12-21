# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import time
import numpy as np
import popart

import ema_utils
import logging_util

# set up logging
logger = logging_util.get_basic_logger('CHECKPOINT_UTILS')


def remove_file_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)


def get_training_ckpt_path(checkpt_dir):
    training_ckpt_path = os.path.join(checkpt_dir, 'rnnt_checkpoint_training.onnx')
    return training_ckpt_path


def get_validation_ckpt_path(checkpt_dir):
    validation_ckpt_path = os.path.join(checkpt_dir, 'rnnt_checkpoint_validation.onnx')
    return validation_ckpt_path


def get_decoder_weights_validation_path(checkpt_dir):
    decoder_weights_validation_path = os.path.join(checkpt_dir, 'decoder_weights_validation.npy')
    return decoder_weights_validation_path


def get_ckpt_ready_path(checkpt_dir):
    ckpt_ready_path = os.path.join(checkpt_dir, 'ready')
    return ckpt_ready_path


def get_loss_data_fp(model_dir):
    rnnt_loss_data_fp = os.path.join(model_dir, 'rnnt_losses.txt')
    return rnnt_loss_data_fp


def get_learning_rate_data_fp(model_dir):
    learning_rate_data_fp = os.path.join(model_dir, 'learning_rates.txt')
    return learning_rate_data_fp


def wait_for_file(file_path):
    while not os.path.exists(file_path):
        time.sleep(1)
    return


def remove_checkpoint_ready_file(conf, epoch):
    # needed to proactively remove stale checkpoint ready file for the next epoch,
    # so another instance won't try to use it
    next_ckpt_dir = os.path.join(conf.model_dir, 'checkpoint_{}'.format(epoch))
    next_ckpt_ready_path = get_ckpt_ready_path(next_ckpt_dir)
    remove_file_if_exists(next_ckpt_ready_path)
    return


def prepare_for_checkpointing(conf):
    """ prepares files before starting training """
    rnnt_loss_data_fp = get_loss_data_fp(conf.model_dir)
    logger.info('Creating RNN-t loss data file at {}'.format(rnnt_loss_data_fp))
    open(rnnt_loss_data_fp, 'w').close()
    learning_rate_data_fp = get_learning_rate_data_fp(conf.model_dir)
    open(learning_rate_data_fp, 'w').close()
    remove_checkpoint_ready_file(conf, conf.start_epoch + 1)
    return


def create_model_checkpt(builder, checkpt_dir, training_session, weight_names, ema_weight_names, precision, enable_ema_weights):
    """ save .onnx file for given session and numpy files with weights for greedy decoder """

    if not os.path.exists(checkpt_dir):
        logger.info("Creating model checkpoint directory {}".format(checkpt_dir))
        os.makedirs(checkpt_dir)

    # Checkpoint is not ready
    ckpt_ready_path = get_ckpt_ready_path(checkpt_dir)
    remove_file_if_exists(ckpt_ready_path)

    training_ckpt_path = get_training_ckpt_path(checkpt_dir)
    remove_file_if_exists(training_ckpt_path)
    validation_ckpt_path = get_validation_ckpt_path(checkpt_dir)
    remove_file_if_exists(validation_ckpt_path)
    training_session.modelToHost(training_ckpt_path)
    if enable_ema_weights:
        # transfer ema weights to original weight tensors for evaluation
        logger.info("Creating .onnx model with EMA weights")
        ema_utils.transfer_ema_weights(training_ckpt_path, validation_ckpt_path)
    else:
        # create a hard link to training onnx from validation onnx
        os.link(training_ckpt_path, validation_ckpt_path)

    decoder_weights_training_path = os.path.join(checkpt_dir, 'decoder_weights_training.npy')
    remove_file_if_exists(decoder_weights_training_path)
    decoder_weights_validation_path = get_decoder_weights_validation_path(checkpt_dir)
    remove_file_if_exists(decoder_weights_validation_path)

    save_weights_for_decoder(builder, training_session, weight_names, precision, decoder_weights_training_path)
    if enable_ema_weights:
        save_weights_for_decoder(builder, training_session, ema_weight_names, precision, decoder_weights_validation_path)
    else:
        # create a hard link to training .npy from validation .npy file
        os.link(decoder_weights_training_path, decoder_weights_validation_path)

    # Checkpoint is ready
    open(ckpt_ready_path, 'a').close()

    return


def save_weights_for_decoder(builder, training_session, weight_names_to_checkpt, precision, decoder_weights_fp):
    """ save weights for greedy decoding to numpy files
    Note: weight_names_to_checkpt can be either actual model weights or exp-mov-averaged weights """

    # Initializing decoder parameter dictionary
    decoder_weights_dict = dict()
    for uname, wname in weight_names_to_checkpt["prediction_network"] + weight_names_to_checkpt["joint_network"]:
        original_wname = wname.replace(ema_utils.EMA_PREFIX, '')
        param_shape = builder.getTensorShape(original_wname)
        decoder_weights_dict[wname] = np.empty(param_shape, precision)

    logger.info("Saving decoder weights to {}".format(decoder_weights_fp))
    weightsIo = popart.PyWeightsIO(decoder_weights_dict)
    training_session.readWeights(weightsIo)

    for uname, wname in weight_names_to_checkpt["prediction_network"] + weight_names_to_checkpt["joint_network"]:
        # remove exp_mov_avg_ prefix from uname and add to dict so that validation script works
        original_uname = uname.replace(ema_utils.EMA_PREFIX, '')
        decoder_weights_dict[original_uname] = decoder_weights_dict[wname]
        decoder_weights_dict.pop(wname, None)
    np.save(decoder_weights_fp, decoder_weights_dict)

    # NOTE - these weights can be loaded as:
    # decoder_weights_dict =  np.load(decoder_weights_fp, allow_pickle=True)[()]

    return


def write_training_progress_results(conf, rnnt_loss, learning_rate):
    """ writes rnnt-loss and learning-rate data to files """
    rnnt_loss_data_fp = get_loss_data_fp(conf.model_dir)
    learning_rate_data_fp = get_learning_rate_data_fp(conf.model_dir)
    with open(rnnt_loss_data_fp, "a") as f:
        f.write(str(rnnt_loss) + '\n')
    with open(learning_rate_data_fp, "a") as f:
        f.write(str(learning_rate) + '\n')
