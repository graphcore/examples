# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import os
import numpy as np
import torch
import multiprocessing
from functools import partial
import time

from ipu_sampler import IpuSimpleSampler
from common.data.dali.data_loader import DaliDataLoader
from common.data.text import Tokenizer
from common.data import features

import common.helpers as helpers
import common.metrics as metrics

import transducer_builder
import conf_utils
import logging_util
import mpi_utils

from rnnt_reference import config
from rnnt_reference.model import RNNT
from transducer_decoder import TransducerGreedyDecoder
import device as device_module

# set up logging
logger = logging_util.get_basic_logger('TRANSDUCER_VALIDATION')

np.set_printoptions(threshold=128)
np.set_printoptions(linewidth=1024)


def _get_popart_type(np_type):
    return {
        np.float16: 'FLOAT16',
        np.float32: 'FLOAT'
    }[np_type]


def create_inputs_for_inference(builder, model_conf, conf):
    """ defines the input tensors of the transcription network for inference """

    inputs = dict()

    # num-mel-bands X frame-stacking-factor
    in_feats = model_conf["rnnt"]["in_feats"]

    inputs["mel_spec_input"] = builder.addInputTensor(popart.TensorInfo(_get_popart_type(conf.precision),
                                                                        [conf.samples_per_device,
                                                                         in_feats,
                                                                         conf.max_spec_len_after_stacking]),
                                                      "mel_spec_input")

    inputs["input_length"] = builder.addInputTensor(popart.TensorInfo("INT32", [conf.samples_per_device]),
                                                    "input_length")

    return inputs


def create_model_and_dataflow_for_inference(builder, model_conf, conf, inputs):
    """ builds the transcription network and dataflow for inference """

    # num-mel-bands X frame-stacking-factor
    in_feats = model_conf["transformer_transducer"]["in_feats"]
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

    inference_transcription_out, inference_transcription_out_lens = transcription_network(inputs["mel_spec_input"],
                                                                                          inputs["input_length"])
    logger.info("Shape of Transcription-Network Output: {}".format(
        builder.getTensorShape(inference_transcription_out)))

    pred_n_hid = model_conf["transformer_transducer"]["pred_n_hid"]

    joint_n_hid = model_conf["transformer_transducer"]["joint_n_hid"]
    joint_dropout = model_conf["transformer_transducer"]["joint_dropout"]
    inference_transcription_out_len = builder.getTensorShape(
        inference_transcription_out)[1]
    joint_network = transducer_builder.JointNetwork(builder,
                                                    inference_transcription_out_len,
                                                    encoder_dim,
                                                    pred_n_hid,
                                                    joint_n_hid,
                                                    conf.num_symbols,
                                                    joint_dropout,
                                                    dtype=conf.precision)

    with builder.virtualGraph(0):
        inference_transcription_out = joint_network.joint_transcription_fc(
            inference_transcription_out)
    logger.info("Shape of Transcription-Network Output after Joint-transciption-fc: {}".format(
        builder.getTensorShape(inference_transcription_out)))

    anchor_types_dict = {
        inference_transcription_out: popart.AnchorReturnType("ALL"),
        inference_transcription_out_lens: popart.AnchorReturnType("ALL"),
    }

    proto = builder.getModelProto()
    dataflow = popart.DataFlow(conf.device_iterations, anchor_types_dict)

    return proto, inference_transcription_out, inference_transcription_out_lens, dataflow


def create_inference_transcription_session(device, model_conf, conf):
    session_options = conf_utils.get_session_options(conf)

    # building model and dataflow
    builder = popart.Builder()
    inference_inputs = create_inputs_for_inference(builder, model_conf, conf)

    proto, inference_transcription_out,\
        inference_transcription_out_lens, dataflow = create_model_and_dataflow_for_inference(builder,
                                                                                             model_conf,
                                                                                             conf,
                                                                                             inference_inputs)

    inference_session, inference_anchors = conf_utils.create_session_anchors(proto,
                                                                             [],
                                                                             device,
                                                                             dataflow,
                                                                             session_options,
                                                                             training=False)

    return inference_session, inference_anchors, inference_inputs, inference_transcription_out, inference_transcription_out_lens


def setup_validation_data_pipeline(conf, transducer_config):
    """ sets up and returns the data-loader for validation """
    logger.info("Setting up datasets for validation ...")

    val_manifests = [os.path.join(
        conf.data_dir, "librispeech-dev-clean-wav.json")]

    # set right absolute path for sentpiece_model
    transducer_config["tokenizer"]["sentpiece_model"] = os.path.join(conf.data_dir, '..',
                                                                     transducer_config["tokenizer"]["sentpiece_model"])
    tokenizer_kw = config.tokenizer(transducer_config)
    val_tokenizer = Tokenizer(**tokenizer_kw)

    val_dataset_kw, val_features_kw, val_splicing_kw, val_specaugm_kw = config.input(
        transducer_config, "val")

    sampler = IpuSimpleSampler(
        conf.samples_per_step, conf.num_instances, conf.instance_idx)

    assert(conf.samples_per_step % conf.num_instances == 0)
    samples_per_step_per_instance = conf.samples_per_step // conf.num_instances
    logger.debug("DaliDataLoader SamplesPerStepPerInstance = {}".format(
        samples_per_step_per_instance))
    val_loader = DaliDataLoader(gpu_id=None,
                                dataset_path=conf.data_dir,
                                config_data=val_dataset_kw,
                                config_features=val_features_kw,
                                json_names=val_manifests,
                                batch_size=samples_per_step_per_instance,
                                sampler=sampler,
                                pipeline_type="val",
                                device_type="cpu",
                                tokenizer=val_tokenizer)
    conf.max_spec_len_after_stacking = round(
        val_loader.max_spec_len_before_stacking / val_splicing_kw["frame_subsampling"])
    conf.num_symbols = val_tokenizer.num_labels + 1

    val_feat_proc = torch.nn.Sequential(
        val_specaugm_kw and features.SpecAugment(
            optim_level=0, **val_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=0, **val_splicing_kw),
        features.FillPadding(
            optim_level=0, max_seq_len=conf.max_spec_len_after_stacking),
    )
    return val_loader, val_feat_proc, val_tokenizer


def convert_embed_weight(x):
    x = x.astype(np.float32)
    return x


def convert_lstm_weight(x):
    """ convert onnx lstm weight tensor to pytorch lstm weight tensor """
    x = x.astype(np.float32)
    assert(x.ndim == 3)
    assert(x.shape[0] == 1)
    x = np.squeeze(x, 0)
    assert(x.shape[0] % 4 == 0)
    xs = np.array_split(x, 4)
    # aiOnnx uses IOFC weights order, while torch uses IFCO.
    y = np.concatenate([xs[i] for i in (0, 2, 3, 1)], 0)
    return y


def convert_lstm_bias(x):
    """ convert onnx lstm bias tensor to pytorch lstm bias tensors """
    x = x.astype(np.float32)
    assert(x.ndim == 2)
    assert(x.shape[0] == 1)
    x = np.squeeze(x, 0)
    assert(x.shape[0] % 2 == 0)
    onnx_bias_splits = np.array_split(x, 2)
    pytorch_biases = []
    for x in onnx_bias_splits:
        xs = np.array_split(x, 4)
        # aiOnnx uses IOFC biases order, while torch uses IFCO.
        y = np.concatenate([xs[i] for i in (0, 2, 3, 1)], 0)
        pytorch_biases.append(y)

    # pytorch_biases[0] corresponds to torch LSTM.bias_ih
    # pytorch_biases[1] corresponds to torch LSTM.bias_hh

    return pytorch_biases[0], pytorch_biases[1]


def convert_fc_weight(x):
    x = x.astype(np.float32)
    x = np.transpose(x)
    return x


def convert_fc_bias(x):
    x = x.astype(np.float32)
    assert(x.ndim == 2)
    assert(x.shape[0] == 1)
    x = np.squeeze(x, 0)
    return x


def create_pytorch_rnnt_model(transducer_config, n_classes):
    ref_config = transducer_config["rnnt"]
    pytorch_rnnt_model = RNNT(n_classes=n_classes, **ref_config)
    pytorch_rnnt_model.cpu()
    pytorch_rnnt_model.eval()
    return pytorch_rnnt_model


def update_pytorch_rnnt_model(pytorch_rnnt_model, decoder_weights_path):
    """ updates the weights of given pytorch RNNT model with the weights required for decoding """
    decoder_weights_dict = np.load(decoder_weights_path, allow_pickle=True)[()]

    embedding_module = pytorch_rnnt_model.prediction.embed

    embedding_weight_popart = decoder_weights_dict["prediction_net_embedding/embedding_matrix"]
    embedding_weight_pytorch = convert_embed_weight(embedding_weight_popart)

    assert(list(embedding_module.weight.shape) ==
           list(embedding_weight_pytorch.shape))
    embedding_module.weight = torch.nn.parameter.Parameter(
        torch.tensor(embedding_weight_pytorch))

    lstm_module = pytorch_rnnt_model.prediction.dec_rnn.lstm

    for layer in range(lstm_module.num_layers):
        lstm_input_weights_key = "prediction_net_rnn_{}/lstm_input_weights".format(
            layer)
        lstm_output_weights_key = "prediction_net_rnn_{}/lstm_output_weights".format(
            layer)
        lstm_biases_key = "prediction_net_rnn_{}/lstm_biases".format(layer)
        lstm_ih_weight_pytorch = convert_lstm_weight(
            decoder_weights_dict[lstm_input_weights_key])
        lstm_hh_weight_pytorch = convert_lstm_weight(
            decoder_weights_dict[lstm_output_weights_key])
        lstm_ih_bias_pytorch, lstm_hh_bias_pytorch = convert_lstm_bias(
            decoder_weights_dict[lstm_biases_key])

        weight_ih_key = "weight_ih_l{}".format(layer)
        weight_ih = lstm_module.__getattr__(weight_ih_key)
        assert(list(weight_ih.shape) == list(lstm_ih_weight_pytorch.shape))
        lstm_module.__setattr__(weight_ih_key, torch.nn.parameter.Parameter(
            torch.tensor(lstm_ih_weight_pytorch)))

        weight_hh_key = "weight_hh_l{}".format(layer)
        weight_hh = lstm_module.__getattr__(weight_hh_key)
        assert(list(weight_hh.shape) == list(lstm_hh_weight_pytorch.shape))
        lstm_module.__setattr__(weight_hh_key, torch.nn.parameter.Parameter(
            torch.tensor(lstm_hh_weight_pytorch)))

        bias_ih_key = "bias_ih_l{}".format(layer)
        bias_ih = lstm_module.__getattr__(bias_ih_key)
        assert(list(bias_ih.shape) == list(lstm_ih_bias_pytorch.shape))
        lstm_module.__setattr__(bias_ih_key, torch.nn.parameter.Parameter(
            torch.tensor(lstm_ih_bias_pytorch)))

        bias_hh_key = "bias_hh_l{}".format(layer)
        bias_hh = lstm_module.__getattr__(bias_hh_key)
        assert(list(bias_hh.shape) == list(lstm_hh_bias_pytorch.shape))
        lstm_module.__setattr__(bias_hh_key, torch.nn.parameter.Parameter(
            torch.tensor(lstm_hh_bias_pytorch)))

    joint_pred_module_fc = pytorch_rnnt_model.joint_pred

    joint_pred_fc_weight_pytorch = convert_fc_weight(
        decoder_weights_dict["joint_net_prediction_fc/weights"])
    joint_pred_fc_bias_pytorch = convert_fc_bias(
        decoder_weights_dict["joint_net_prediction_fc/bias"])

    assert(list(joint_pred_module_fc.weight.shape) ==
           list(joint_pred_fc_weight_pytorch.shape))
    joint_pred_module_fc.weight = torch.nn.parameter.Parameter(
        torch.tensor(joint_pred_fc_weight_pytorch))
    assert(list(joint_pred_module_fc.bias.shape) ==
           list(joint_pred_fc_bias_pytorch.shape))
    joint_pred_module_fc.bias = torch.nn.parameter.Parameter(
        torch.tensor(joint_pred_fc_bias_pytorch))

    joint_out_module_fc = pytorch_rnnt_model.joint_net[2]

    joint_out_fc_weight_pytorch = convert_fc_weight(
        decoder_weights_dict["joint_net_out_fc/weights"])
    joint_out_fc_bias_pytorch = convert_fc_bias(
        decoder_weights_dict["joint_net_out_fc/bias"])

    assert(list(joint_out_module_fc.weight.shape) ==
           list(joint_out_fc_weight_pytorch.shape))
    joint_out_module_fc.weight = torch.nn.parameter.Parameter(
        torch.tensor(joint_out_fc_weight_pytorch))
    assert(list(joint_out_module_fc.bias.shape) ==
           list(joint_out_fc_bias_pytorch.shape))
    joint_out_module_fc.bias = torch.nn.parameter.Parameter(
        torch.tensor(joint_out_fc_bias_pytorch))

    logger.info("Pytorch CPU RNN-T model updated with weights for decoding.")
    return


# Pads numpy array's dimension 0 to the target size by zeros
def pad(t, dim0_target):
    pad_size = dim0_target - t.shape[0]
    if pad_size == 0:
        return t
    pading_shape = np.zeros([t.ndim, 2], dtype=np.int32)
    # Padding dimension 0 to the left by 0 and to the right by pad_size
    # Remaining dimensions are not padded
    pading_shape[0, 1] = pad_size
    tp = np.pad(t, pading_shape, constant_values=0)
    return tp


def evaluate(conf, onnx_path, val_loader, val_feat_proc,
             inference_session, inference_anchors, inference_inputs,
             inference_transcription_out, inference_transcription_out_lens,
             pytorch_rnnt_model, greedy_decoder, detokenize):

    start_time = time.time()

    logger.info(
        "Getting trained weights for inference from {}".format(onnx_path))
    inference_session.resetHostWeights(
        onnx_path, ignoreWeightsInModelWithoutCorrespondingHostWeight=True)
    inference_session.weightsFromHost()

    feats_data = []
    feat_lens_data = []
    txt_data = []
    txt_lens_data = []

    samples_per_step_per_instance = conf.samples_per_step // conf.num_instances

    # No need to compute losses for validation
    agg = {'preds': [], 'txts': [], 'idx': []}
    overall_scores, overall_words = (0, 0)
    logger.info("Running transcription network on evaluation dataset")
    for audio, audio_lens, txt, txt_lens in val_loader:
        feats, feat_lens = val_feat_proc([audio, audio_lens])

        feats = feats.numpy()
        feat_lens = feat_lens.numpy()
        # txt is of np.array type as implemented in reference code
        txt_lens = txt_lens.numpy()

        feats = pad(feats, samples_per_step_per_instance)
        feat_lens = pad(feat_lens, samples_per_step_per_instance)
        txt = pad(txt, samples_per_step_per_instance)
        txt_lens = pad(txt_lens, samples_per_step_per_instance)

        stepio = popart.PyStepIO(
            {
                inference_inputs["mel_spec_input"]: feats.astype(conf.precision),
                inference_inputs["input_length"]: feat_lens.astype(np.int32),
            }, inference_anchors)

        inference_session.run(stepio)

        # converting to torch tensor
        feats = torch.tensor(inference_anchors[inference_transcription_out])
        feat_lens = torch.tensor(
            inference_anchors[inference_transcription_out_lens])
        feat_lens = torch.flatten(feat_lens)
        step_size = feat_lens.shape[0]
        feats = torch.reshape(feats,
                              (step_size,
                               feats.shape[-2], feats.shape[-1]))

        txt = torch.tensor(txt)
        txt_lens = torch.tensor(txt_lens)

        feats_data.append(feats)
        feat_lens_data.append(feat_lens)
        txt_data.append(txt)
        txt_lens_data.append(txt_lens)

    num_cpus = os.cpu_count()
    num_workers = max(1, min(16, num_cpus // conf.num_instances))
    logger.info(
        "Creating multiprocessor pool with {} workers and function for decoding".format(num_workers))
    greedy_decoding_processor_pool = multiprocessing.pool.ThreadPool(
        processes=num_workers)
    greedy_decoding_func = partial(greedy_decoder.decode, pytorch_rnnt_model)

    pred_results = []
    ground_truths_dekotenized = []
    logger.info("Submitting jobs for greedy decoding")

    feat_iter = zip(feats_data, feat_lens_data, txt_data, txt_lens_data)
    for feats, feat_lens, txt, txt_lens in feat_iter:

        step_size = feat_lens.shape[0]
        batch_size = step_size // conf.device_iterations

        for bind in range(conf.device_iterations):
            feats_b = feats[bind:bind + batch_size]
            feat_lens_b = feat_lens[bind:bind + batch_size]
            txt_b = txt[bind:bind + batch_size]
            txt_lens_b = txt_lens[bind:bind + batch_size]

            pred_results.append(greedy_decoding_processor_pool.apply_async(greedy_decoding_func,
                                                                           (feats_b, feat_lens_b)))
            ground_truths_dekotenized.append(
                helpers.gather_transcripts([txt_b], [txt_lens_b], detokenize))

    logger.info("Generating predictions and computing Word Error Rate (WER)")
    pred_iter = zip(pred_results, ground_truths_dekotenized)
    for idx, (pred_result, gts_detokenized) in enumerate(pred_iter):
        preds_detokenized = helpers.gather_predictions(
            [pred_result.get()], detokenize)

        batch_wer, batch_scores, batch_words = metrics.word_error_rate(
            preds_detokenized, gts_detokenized)
        agg['preds'] += preds_detokenized
        agg['txts'] += gts_detokenized

    wer, scores, num_words, _ = helpers.process_evaluation_epoch(agg)
    logger.info("Total time for Transducer Decoding = {:.1f} secs".format(
        time.time() - start_time))

    greedy_decoding_processor_pool.close()
    greedy_decoding_processor_pool.join()

    return wer, scores, num_words


def dist_wer(scores, num_words):
    scores = mpi_utils.mpi_reduce(scores, average=False)
    num_words = mpi_utils.mpi_reduce(num_words, average=False)
    if num_words != 0:
        wer = 1.0*scores/num_words
    else:
        wer = float('inf')
    return wer


if __name__ == "__main__":

    parser = conf_utils.add_conf_args(run_mode="validation")
    conf = conf_utils.get_conf(parser)

    runtime_conf = conf_utils.RunTimeConf(conf, run_mode='validation')

    transducer_config = config.load(conf.model_conf_file)

    val_loader, val_feat_proc, val_tokenizer = setup_validation_data_pipeline(
        runtime_conf, transducer_config)

    pytorch_rnnt_model = create_pytorch_rnnt_model(
        transducer_config, val_tokenizer.num_labels + 1)

    greedy_decoder = TransducerGreedyDecoder(
        blank_idx=0, max_symbols_per_step=conf.max_symbols_per_step, shift_labels_by_one=True)

    device = device_module.acquire_device(
        conf, runtime_conf.local_replication_factor, runtime_conf)
    inference_session, inference_anchors, inference_inputs, \
        inference_transcription_out, inference_transcription_out_lens = \
        create_inference_transcription_session(
            device, conf.model_conf, runtime_conf)

    decoder_weights_path = os.path.join(
        conf.model_dir, "decoder_weights_validation.npy")
    update_pytorch_rnnt_model(pytorch_rnnt_model, decoder_weights_path)

    onnx_path = os.path.join(conf.model_dir, "rnnt_checkpoint_validation.onnx")
    wer, scores, num_words = evaluate(runtime_conf, onnx_path,
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

    if runtime_conf.num_instances > 1:
        wer = dist_wer(scores, num_words)
    if runtime_conf.instance_idx == 0:
        logger.info("Global Word Error Rate (WER) = {}".format(wer))
