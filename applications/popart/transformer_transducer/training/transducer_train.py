# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import os
from collections import deque
import time
import glob

from ipu_sampler import IpuBucketingSampler
from common.data.dali.data_loader import DaliDataLoader
from common.data.text import Tokenizer
from rnnt_reference import config

import logging_util
import conf_utils
import custom_op_utils
import checkpoint_utils
import transducer_blocks
import transducer_builder
from transducer_optimizer import TransducerOptimizerFactory
import ema_utils
import device
from feat_proc_cpp_async import AsyncDataProcessor
import transducer_validation
from transducer_decoder import TransducerGreedyDecoder
import test_transducer

# set up logging
logger = logging_util.get_basic_logger('TRANSDUCER_TRAIN')


def _get_popart_type(np_type):
    return {
        np.float16: 'FLOAT16',
        np.float32: 'FLOAT',
        np.int32: 'INT32'
    }[np_type]


def create_inputs_for_training(builder, model_conf, conf):
    """ defines the input tensors for the Transformer Transducer model """

    inputs = dict()

    in_feats = model_conf["transformer_transducer"]["in_feats"]  # num-mel-bands X frame-stacking-factor

    inputs["text_input"] = builder.addInputTensor(popart.TensorInfo("INT32",
                                                                    [conf.samples_per_device,
                                                                     conf.max_token_sequence_len]),
                                                  "text_input")
    inputs["mel_spec_input"] = builder.addInputTensor(popart.TensorInfo(_get_popart_type(conf.precision),
                                                                        [conf.samples_per_device,
                                                                         in_feats,
                                                                         conf.max_spec_len_after_stacking]),
                                                      "mel_spec_input")
    inputs["input_length"] = builder.addInputTensor(popart.TensorInfo("INT32", [conf.samples_per_device]),
                                                    "input_length")

    inputs["target_length"] = builder.addInputTensor(popart.TensorInfo("INT32", [conf.samples_per_device]),
                                                     "target_length")

    return inputs


def create_model_and_dataflow_for_training(builder, model_conf, conf, inputs):
    """ builds the Transformer Transducer model, loss function and dataflow for training """

    in_feats = model_conf["transformer_transducer"]["in_feats"]  # num-mel-bands X frame-stacking-factor
    subsampling_factor = model_conf["transformer_transducer"]["subsampling_factor"]
    num_encoder_layers = model_conf["transformer_transducer"]["num_encoder_layers"]
    encoder_dim = model_conf["transformer_transducer"]["encoder_dim"]
    num_attention_heads = model_conf["transformer_transducer"]["num_attention_heads"]
    enc_dropout = model_conf["transformer_transducer"]["enc_dropout"]
    kernel_size = model_conf["transformer_transducer"]["kernel_size"]


    transcription_network = transducer_builder.TranscriptionNetwork(builder,
                                                                    in_feats,
                                                                    subsampling_factor,
                                                                    num_encoder_layers,
                                                                    encoder_dim,
                                                                    num_attention_heads,
                                                                    enc_dropout,
                                                                    kernel_size=kernel_size,
                                                                    dtype=conf.precision)

    pred_n_hid = model_conf["transformer_transducer"]["pred_n_hid"]
    pred_rnn_layers = model_conf["transformer_transducer"]["pred_rnn_layers"]
    pred_dropout = model_conf["transformer_transducer"]["pred_dropout"]
    forget_gate_bias = model_conf["transformer_transducer"]["forget_gate_bias"]
    weights_init_scale = model_conf["transformer_transducer"]["weights_init_scale"]

    prediction_network = transducer_builder.PredictionNetwork(builder,
                                                              conf.num_symbols - 1,
                                                              pred_n_hid,
                                                              pred_rnn_layers,
                                                              pred_dropout,
                                                              forget_gate_bias,
                                                              weights_init_scale,
                                                              dtype=conf.precision)

    transcription_out, transcription_lens = transcription_network(inputs["mel_spec_input"], inputs["input_length"])
    logger.info("Shape of Transcription-Network Output: {}".format(builder.getTensorShape(transcription_out)))

    prediction_out = prediction_network(inputs["text_input"])
    logger.info("Shape of Prediction-Network Output: {}".format(builder.getTensorShape(prediction_out)))

    joint_n_hid = model_conf["transformer_transducer"]["joint_n_hid"]
    joint_dropout = model_conf["transformer_transducer"]["joint_dropout"]
    transcription_out_len = builder.getTensorShape(transcription_out)[1]
    joint_network_w_rnnt_loss = transducer_builder.JointNetwork_wRNNTLoss(builder,
                                                                          transcription_out_len,
                                                                          encoder_dim,
                                                                          pred_n_hid,
                                                                          joint_n_hid,
                                                                          conf.num_symbols,
                                                                          joint_dropout,
                                                                          dtype=conf.precision,
                                                                          transcription_out_split_size=conf.joint_net_split_size,
                                                                          do_batch_serialization=conf.do_batch_serialization_joint_net,
                                                                          samples_per_device=conf.samples_per_device,
                                                                          batch_split_size=conf.joint_net_batch_split_size,
                                                                          shift_labels_by_one=True)

    neg_log_likelihood = joint_network_w_rnnt_loss(transcription_out, transcription_lens, prediction_out,
                                                   inputs["text_input"], inputs["target_length"])
    # logger.info("Shape of Joint-Network Output: {}".format(builder.getTensorShape(joint_out)))

    logger.info("Parameter count of the transcription network: {}".format(transcription_network.param_count))
    logger.info("Parameter count of the prediction network: {}".format(prediction_network.param_count))
    logger.info("Parameter count of the joint network: {}".format(joint_network_w_rnnt_loss.param_count))
    logger.info("Parameter count of the whole network: {}".format(transducer_blocks.Block.global_param_count))

    weight_names = {
        "transcription_network": transcription_network.tensor_list,
        "prediction_network": prediction_network.tensor_list,
        "joint_network": joint_network_w_rnnt_loss.tensor_list
    }

    if conf.enable_ema_weights:
        # define exponential moving average weights
        ema_weight_names = ema_utils.create_exp_mov_avg_weights(builder, weight_names, conf.ema_factor)
    else:
        ema_weight_names = None

    anchor_types_dict = {
        neg_log_likelihood: popart.AnchorReturnType("ALL"),
    }

    proto = builder.getModelProto()
    dataflow = popart.DataFlow(conf.batches_per_step, anchor_types_dict)

    return proto, neg_log_likelihood, dataflow, weight_names, ema_weight_names


def setup_training_data_pipeline(conf, transducer_config):
    """ sets up and returns the data-loader for training """
    logger.info('Setting up datasets for training (instance {})...'.format(conf.instance_idx))

    train_manifests = [os.path.join(conf.data_dir, train_manifest)
                       for train_manifest in ['librispeech-train-clean-100-wav.json',
                                              'librispeech-train-clean-360-wav.json',
                                              'librispeech-train-other-500-wav.json']]


    train_dataset_kw, train_features_kw, train_splicing_kw, train_specaugm_kw = config.input(transducer_config, 'train')
    conf.train_splicing_kw = train_splicing_kw
    conf.train_specaugm_kw = train_specaugm_kw

    # set right absolute path for sentpiece_model
    transducer_config["tokenizer"]["sentpiece_model"] = os.path.join(conf.data_dir, '..',
                                                                     transducer_config["tokenizer"]["sentpiece_model"])
    tokenizer_kw = config.tokenizer(transducer_config)
    tokenizer = Tokenizer(**tokenizer_kw)

    sampler = IpuBucketingSampler(
        conf.num_buckets,
        conf.samples_per_step,
        conf.num_epochs,
        np.random.default_rng(seed=310),
        num_instances=conf.num_instances,
        instance_offset=conf.instance_idx
    )

    assert(conf.samples_per_step % conf.num_instances == 0)
    samples_per_step_per_instance = conf.samples_per_step // conf.num_instances
    logger.debug("DaliDataLoader SamplesPerStepPerInstance = {} (instance {})".format(samples_per_step_per_instance,
                                                                                      conf.instance_idx))
    train_loader = DaliDataLoader(gpu_id=None,
                                  dataset_path=conf.data_dir,
                                  config_data=train_dataset_kw,
                                  config_features=train_features_kw,
                                  json_names=train_manifests,
                                  batch_size=samples_per_step_per_instance,
                                  # dataloader should return data for one step for each instance
                                  sampler=sampler,
                                  grad_accumulation_steps=1,
                                  pipeline_type='train',
                                  device_type="cpu",
                                  tokenizer=tokenizer)
    conf.max_spec_len_after_stacking = round(train_loader.max_spec_len_before_stacking /
                                             train_splicing_kw["frame_subsampling"])
    conf.max_token_sequence_len = train_loader.max_token_sequence_len
    conf.num_symbols = tokenizer.num_labels + 1

    return train_loader


if __name__ == '__main__':

    logger.info("RNN-T Training in Popart")

    parser = conf_utils.add_conf_args(run_mode='training')
    conf = conf_utils.get_conf(parser)

    training_runtime_conf = conf_utils.RunTimeConf(conf, run_mode='training')
    instance_idx = training_runtime_conf.instance_idx

    np.random.seed(instance_idx)

    transducer_config = config.load(conf.model_conf_file)
    config.apply_duration_flags(transducer_config, conf.max_duration)

    if os.path.exists(conf.model_dir):
        checkpoint_dirs = glob.glob(os.path.join(conf.model_dir, 'checkpoint_*'))
        if len(checkpoint_dirs) > 0:
            logger.warn("Checkpoints located at model checkpoint directory {} will be over-written!".format(conf.model_dir))
    else:
        logger.info("Creating model checkpoint directory {}".format(conf.model_dir))
        os.makedirs(conf.model_dir)

    if conf.generated_data:
        train_loader = test_transducer.setup_generated_data_pipeline(training_runtime_conf, transducer_config)
    else:
        train_loader = setup_training_data_pipeline(training_runtime_conf, transducer_config)

    if conf.do_validation:
        val_runtime_conf = conf_utils.RunTimeConf(conf, run_mode='validation')
        val_loader, val_feat_proc, val_tokenizer = transducer_validation.setup_validation_data_pipeline(val_runtime_conf,
                                                                                                        transducer_config)
        pytorch_rnnt_model = transducer_validation.create_pytorch_rnnt_model(transducer_config,
                                                                             val_tokenizer.num_labels + 1)
        greedy_decoder = TransducerGreedyDecoder(blank_idx=0,
                                                 max_symbols_per_step=conf.max_symbols_per_step,
                                                 shift_labels_by_one=True)

    training_session_options = conf_utils.get_session_options(training_runtime_conf)
    device = device.acquire_device(conf, training_runtime_conf.local_replication_factor, training_runtime_conf)

    logger.debug("Loading SparseLogSoftMax op")
    custom_op_utils.load_custom_sparse_logsoftmax_op()
    logger.debug("Loading RNN-T loss op")
    custom_op_utils.load_custom_rnnt_op()
    logger.debug("Loading Exp-Mov-Avg custom op/pattern")
    custom_op_utils.load_exp_avg_custom_op()

    # building model and dataflow
    builder = popart.Builder()
    training_inputs = create_inputs_for_training(builder, conf.model_conf, training_runtime_conf)

    proto, rnnt_loss, dataflow, weight_names, ema_weight_names = \
        create_model_and_dataflow_for_training(builder, conf.model_conf, training_runtime_conf, training_inputs)

    if conf.enable_ema_weights:
        ema_utils.set_ema_weights_offchip(training_session_options, ema_weight_names)

    steps_per_epoch = len(train_loader)
    step, epoch = (conf.start_epoch * steps_per_epoch, conf.start_epoch)

    optimizer_factory = TransducerOptimizerFactory(conf.optimizer, conf.base_lr, conf.min_lr, conf.lr_exp_gamma,
                                                   steps_per_epoch, conf.warmup_epochs, conf.hold_epochs,
                                                   conf.beta1, conf.beta2, conf.weight_decay,
                                                   opt_eps=1e-9, loss_scaling=conf.loss_scaling,
                                                   gradient_clipping_norm=conf.gradient_clipping_norm,
                                                   max_weight_norm=conf.max_weight_norm)

    transducer_optimizer = optimizer_factory.update_and_create(step, epoch)

    # create training session
    logger.info("Creating the training session")
    training_session, training_anchors = \
        conf_utils.create_session_anchors(proto,
                                          rnnt_loss,
                                          device,
                                          dataflow,
                                          training_session_options,
                                          training=True,
                                          optimizer=transducer_optimizer,
                                          use_popdist=training_runtime_conf.use_popdist)

    if conf.do_validation:
        inference_session, inference_anchors, inference_inputs, inference_transcription_out, inference_transcription_out_lens = \
            transducer_validation.create_inference_transcription_session(device, conf.model_conf, val_runtime_conf)

    if conf.start_checkpoint_dir:
        onnx_fp = checkpoint_utils.get_training_ckpt_path(conf.start_checkpoint_dir)
        logger.info("Loading weights from starting checkpoint: {}".format(onnx_fp))
        training_session.resetHostWeights(onnx_fp)
    elif conf.start_epoch > 0:
        raise RuntimeError(f"If start epoch > 0, the start checkpoint directory must be provided")

    logger.info("Graph Prepared Successfully! Sending weights from Host")
    training_session.weightsFromHost()

    # Saving initialized model to checkpoint
    if instance_idx == 0:
        checkpoint_utils.prepare_for_checkpointing(conf)
        ckpt_dir = os.path.join(conf.model_dir, 'checkpoint_initial')
        logger.info('Saving initialized model to {}'.format(ckpt_dir))
        checkpoint_utils.create_model_checkpt(builder, ckpt_dir, training_session, weight_names, ema_weight_names,
                                              training_runtime_conf.precision, conf.enable_ema_weights)

    rnnt_loss_data = deque(maxlen=steps_per_epoch)

    data_iterator = train_loader.data_iterator()
    logger.info("Creating Asynchronous Data Processor")
    async_data_processor = AsyncDataProcessor(conf = training_runtime_conf)
    # We want to use different seeds for different instances,
    # so that random masks sequences in feature augmentation are different.
    async_data_processor.setRandomSeed(instance_idx)
    async_data_processor.set_iterator(data_iterator)

    for epoch in range(conf.start_epoch, conf.num_epochs):

        logger.info("Epoch # {}".format(epoch + 1))

        async_data_processor.submit_data()

        while True:
            step_start_time = time.time()
            start_time = step_start_time

            feat_proc_result = async_data_processor.get()
            assert(feat_proc_result and len(feat_proc_result) == 4)
            feats, feat_lens, txt, txt_lens = feat_proc_result

            logger.debug("Feature acquisition time: {:.6}".format(time.time() - start_time))

            start_time = time.time()

            async_data_processor.submit_data()

            logger.debug("Data retrieval time: {:.6}".format(time.time() - start_time))

            start_time = time.time()

            stepio = popart.PyStepIO(
                {
                    training_inputs["text_input"]: txt,
                    training_inputs["mel_spec_input"]: feats,
                    training_inputs["input_length"]: feat_lens,
                    training_inputs["target_length"]: txt_lens,
                }, training_anchors)

            training_session.run(stepio)

            logger.debug("IPU time: {:.6}".format(time.time() - start_time))

            rnnt_loss_data.append(np.copy(training_anchors[rnnt_loss]))
            ave_loss = np.mean(rnnt_loss_data)
            if instance_idx == 0:
                current_step_loss = np.mean(training_anchors[rnnt_loss])
                logger.info("Epoch {}, Step {}/{}, Current RNNT loss: {}, Average RNNT loss: {}".
                            format(epoch + 1,
                                   step % steps_per_epoch + 1, steps_per_epoch,
                                   str(current_step_loss),
                                   str(ave_loss)))
                checkpoint_utils.write_training_progress_results(conf, ave_loss, optimizer_factory.current_lr)

            step += 1
            transducer_optimizer = optimizer_factory.update_and_create(step, epoch)
            training_session.updateOptimizerFromHost(transducer_optimizer)

            step_time = time.time() - step_start_time
            logger.debug("Step time: {:.6}".format(step_time))
            logger.info("Throughput: {}".format(training_runtime_conf.samples_per_step / step_time))

            if step % steps_per_epoch == 0:
                break

        # Saving initialized model to checkpoint
        ckpt_dir = os.path.join(conf.model_dir, 'checkpoint_{}'.format(epoch + 1))
        if instance_idx == 0:
            logger.info('Saving model after epoch {} to {}'.format(epoch + 1, ckpt_dir))
            checkpoint_utils.create_model_checkpt(builder, ckpt_dir, training_session, weight_names, ema_weight_names,
                                                  training_runtime_conf.precision, conf.enable_ema_weights)

            # Proactively remove stale checkpoint ready file for the next epoch,
            # so another instance won't use it
            checkpoint_utils.remove_checkpoint_ready_file(conf, epoch + 2)

        if conf.do_validation and epoch + 1 >= conf.epoch_to_start_validation:
            start_time = time.time()

            training_ckpt_path = checkpoint_utils.get_training_ckpt_path(ckpt_dir)
            validation_ckpt_path = checkpoint_utils.get_validation_ckpt_path(ckpt_dir)
            decoder_weights_validation_path = checkpoint_utils.get_decoder_weights_validation_path(ckpt_dir)

            ckpt_ready_path = checkpoint_utils.get_ckpt_ready_path(ckpt_dir)
            # We need this, because in poprun scenario, current instance can finish it's epoch earlier than instance 0
            checkpoint_utils.wait_for_file(ckpt_ready_path)

            transducer_validation.update_pytorch_rnnt_model(pytorch_rnnt_model, decoder_weights_validation_path)

            # Run validation network
            wer, scores, num_words = transducer_validation.evaluate(val_runtime_conf, validation_ckpt_path,
                                                                    val_loader,
                                                                    val_feat_proc,
                                                                    inference_session,
                                                                    inference_anchors,
                                                                    inference_inputs,
                                                                    inference_transcription_out,
                                                                    inference_transcription_out_lens,
                                                                    pytorch_rnnt_model,
                                                                    greedy_decoder,
                                                                    val_tokenizer.detokenize)

            logger.debug("Validation time: {:.6}".format(time.time() - start_time))

            if training_runtime_conf.num_instances > 1:
                wer = transducer_validation.dist_wer(scores, num_words)
            if instance_idx == 0:
                logger.info("Global WER = {}".format(wer))

            # Weights on a same device are now replaced with weights from inference session
            # We need to restore them
            training_session.resetHostWeights(training_ckpt_path)
            training_session.weightsFromHost()

    async_data_processor.stop()
