# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import time
import os
import sys
import json
from tqdm import tqdm
from collections import deque
import ctypes
import logging_util

from deep_voice_model import PopartDeepVoice
import deep_voice_data
import conf_utils
import text_utils

# set up logging
logger = logging_util.get_basic_logger('DEEP_VOICE')


so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "custom_ops.so")
if os.path.exists(so_path):
    ctypes.cdll.LoadLibrary(so_path)
else:
    logger.warn("Could not find custom_ops.so. To enable gradient-clipping, "
                "execute `make all` before running this script.")


def _get_popart_type(np_type):
    return {
        np.float16: 'FLOAT16',
        np.float32: 'FLOAT'
    }[np_type]


def create_inputs_for_training(builder, conf):
    """ defines the input tensors for the deep voice model """

    if conf.num_io_tiles > 0:
        exchange_strategy = popart.ExchangeStrategy.OverlapInnerLoop
    else:
        exchange_strategy = popart.ExchangeStrategy.JustInTime
    inputs = dict()

    inputs["text_input"] = builder.addInputTensor(
        popart.TensorInfo("INT32", [conf.samples_per_device, conf.max_text_sequence_length]),
        popart.InputSettings(popart.TileSet.IO, exchange_strategy),
        "text_input"
    )
    inputs["mel_spec_input"] = builder.addInputTensor(
        popart.TensorInfo(_get_popart_type(conf.precision),
                          [conf.samples_per_device,
                          conf.mel_bands * conf.n_frames_per_pred,
                          conf.max_spectrogram_length]),
        popart.InputSettings(popart.TileSet.IO, exchange_strategy),
        "mel_spec_input"
    )

    inputs["speaker_id"] = builder.addInputTensor(
        popart.TensorInfo("INT32", [conf.samples_per_device, 1]),
        popart.InputSettings(popart.TileSet.IO, exchange_strategy),
        "speaker_id"
    )

    inputs["mag_spec_input"] = builder.addInputTensor(
        popart.TensorInfo(_get_popart_type(conf.precision),
                          [conf.samples_per_device,
                          (conf.n_fft//2 + 1) * conf.n_frames_per_pred,
                          conf.max_spectrogram_length]),
        popart.InputSettings(popart.TileSet.IO, exchange_strategy),
        "mag_spec_input"
    )

    inputs["done_labels"] = builder.addInputTensor(
        popart.TensorInfo("INT32", [conf.samples_per_device, 1, conf.max_spectrogram_length]),
        popart.InputSettings(popart.TileSet.IO, exchange_strategy),
        "done_labels"
    )

    return inputs


def create_model_and_dataflow_for_training(builder, conf, inputs, anchor_mode='train'):
    """ builds the deep-voice model, loss function and dataflow for training """

    def temporal_slice(tensor, start, end):
        """ slices tensors along the temporal (last) dimension """
        tensor_shape = builder.getTensorShape(tensor)
        slice_starts = builder.aiOnnx.constant(np.array([0, 0, start]).astype('int32'),
                                               'spec_slice_starts')
        slice_ends = builder.aiOnnx.constant(np.array([tensor_shape[0], tensor_shape[1], end]).astype('int32'),
                                             'spec_slice_ends')
        return builder.aiOnnx.slice([tensor, slice_starts, slice_ends])

    def type_cast(tensor, in_type, out_type):
        if in_type != out_type:
            return builder.aiOnnx.cast([tensor], out_type)
        else:
            return tensor

    def get_attention_mask(g=0.2):
        """ returns attention mask for guided attention """
        attention_mask = np.zeros((conf.max_text_sequence_length, conf.max_spectrogram_length), dtype=conf.precision)
        for n in range(conf.max_text_sequence_length):
            for t in range(conf.max_spectrogram_length):
                attention_mask[n, t] = 1 - np.exp(-(n / conf.max_text_sequence_length -
                                                    t / conf.max_spectrogram_length) ** 2 / (2 * g * g))
        attention_mask = builder.aiOnnx.constant(attention_mask, 'attention_mask')
        return attention_mask

    def get_done_mask(done_labels, num_timesteps):
        """ returns done mask for spectrogram loss computation """
        done_labels_sliced = temporal_slice(done_labels, 1, num_timesteps)
        done_mask = builder.aiOnnx.add(
            [builder.aiOnnx.constant(np.array(1.0).astype(np.float32)),
             builder.aiOnnx.neg([done_labels_sliced])])
        return done_mask

    deep_voice_model = PopartDeepVoice(conf, builder,
                                       for_inference=False)

    main_outputs, aux_outputs, name_to_tensor = deep_voice_model(inputs["text_input"],
                                                                 inputs["mel_spec_input"],
                                                                 inputs["speaker_id"])

    num_timesteps = builder.getTensorShape(inputs["mel_spec_input"])[-1]
    float_type = _get_popart_type(conf.precision)

    # type cast tensors before loss computation (in case of doing experiments with FP16)
    mel_input_fp32_cast = type_cast(temporal_slice(inputs["mel_spec_input"], 1, num_timesteps),
                                    float_type, 'FLOAT')
    mel_output_fp32_cast = type_cast(temporal_slice(main_outputs["mel_spec_output"], 0, num_timesteps-1),
                                     float_type, 'FLOAT')

    mag_spec_input_fp32_cast = type_cast(temporal_slice(inputs["mag_spec_input"], 1, num_timesteps),
                                         float_type, 'FLOAT')
    mag_spec_output_fp32_cast = type_cast(temporal_slice(main_outputs["mag_spec_output"], 0, num_timesteps-1),
                                          float_type, 'FLOAT')

    done_flag_output_fp32_cast = type_cast(main_outputs["done_flag_output"], float_type, 'FLOAT')
    done_labels_fp32_cast = type_cast(inputs["done_labels"], 'INT32', 'FLOAT')

    done_mask = get_done_mask(done_labels_fp32_cast, num_timesteps)

    # mel-spectrogram reconstruction loss for decoder
    mel_spec_l1_loss = builder.aiGraphcore.l1loss(
        [builder.aiOnnx.mul([done_mask, builder.aiOnnx.add([mel_output_fp32_cast,
                                                            builder.aiOnnx.neg([mel_input_fp32_cast])])])],
        1.0, reduction=popart.ReductionType.Mean)

    # linear-scale spectrogram loss for converter
    mag_spec_l1_loss = builder.aiGraphcore.l1loss(
        [builder.aiOnnx.mul([done_mask, builder.aiOnnx.add([mag_spec_output_fp32_cast,
                                                            builder.aiOnnx.neg([mag_spec_input_fp32_cast])])])],
        1.0, reduction=popart.ReductionType.Mean)

    # loss for done-flags
    done_flag_loss = builder.aiGraphcore.l1loss(
        [builder.aiOnnx.add([done_flag_output_fp32_cast, builder.aiOnnx.neg([done_labels_fp32_cast])])],
        1.0, reduction=popart.ReductionType.Mean)

    total_loss = builder.aiOnnx.add([mel_spec_l1_loss, mag_spec_l1_loss])
    total_loss = builder.aiOnnx.add([total_loss, done_flag_loss])

    # add desired output tensors
    builder.addOutputTensor(main_outputs["mel_spec_output"])
    builder.addOutputTensor(main_outputs["mag_spec_output"])
    builder.addOutputTensor(aux_outputs["speaker_embedding_matrix"])
    for attention_distribution in aux_outputs["attention_scores_arrays"]:
        builder.addOutputTensor(attention_distribution)

    anchor_types_dict = {
        mel_spec_l1_loss: popart.AnchorReturnType("ALL"),
        mag_spec_l1_loss: popart.AnchorReturnType("ALL"),
        done_flag_loss: popart.AnchorReturnType("ALL"),
    }
    loss_dict = {"mel_spec_l1_loss": mel_spec_l1_loss,
                 "mag_spec_l1_loss": mag_spec_l1_loss,
                 "done_flag_loss": done_flag_loss}

    if conf.use_guided_attention:
        attention_mask = get_attention_mask(g=conf.guided_attention_g)
        masked_attention = builder.aiOnnx.mul([attention_mask, aux_outputs["attention_scores_arrays"][0]])
        for attention_distribution in aux_outputs["attention_scores_arrays"][1:]:
            masked_attention = builder.aiOnnx.add([masked_attention,
                                                   builder.aiOnnx.mul([attention_mask, attention_distribution])])
        attention_loss = builder.aiGraphcore.l1loss([masked_attention], 1.0, reduction=popart.ReductionType.Mean)
        anchor_types_dict[attention_loss] = popart.AnchorReturnType("ALL")
        loss_dict["attention_loss"] = attention_loss
        total_loss = builder.aiOnnx.add([total_loss, attention_loss])

    loss_dict["total_loss"] = total_loss

    if anchor_mode == 'inference':
        anchor_types_dict[aux_outputs["speaker_embedding_matrix"]] = popart.AnchorReturnType("ALL")
        for attention_distribution in aux_outputs["attention_scores_arrays"]:
            anchor_types_dict[attention_distribution] = popart.AnchorReturnType("ALL")
        anchor_types_dict[main_outputs["mel_spec_output"]] = popart.AnchorReturnType("ALL")
        anchor_types_dict[main_outputs["mag_spec_output"]] = popart.AnchorReturnType("ALL")

    proto = builder.getModelProto()
    dataflow = popart.DataFlow(conf.batches_per_step, anchor_types_dict)

    return proto, loss_dict, dataflow, main_outputs, aux_outputs, name_to_tensor


def get_output_names_files(conf):
    """ initializes text files to write various loss values """

    outputs = ['mag_spec', 'mel_spec', 'done_flag', 'total']
    if conf.use_guided_attention:
        outputs.append('attention')
    out_dict = dict()

    for out_var in outputs:
        train_out_filename = os.path.join(conf.model_dir, 'train_' + out_var + '_losses.txt')
        val_out_filename = os.path.join(conf.model_dir, 'val_' + out_var + '_losses.txt')
        open(train_out_filename, 'w').close()
        open(val_out_filename, 'w').close()
        out_dict['train_' + out_var] = train_out_filename
        out_dict['val_' + out_var] = val_out_filename

    return out_dict


def prep_loss_lists(dataset, output_names_files):
    """ prepares lists to save loss values over all steps """

    losses_dict = dict()

    for out_var in output_names_files.keys():
        if 'train_' in out_var:
            losses_dict[out_var] = deque(maxlen=dataset.num_steps_train_set)
        elif 'val_' in out_var:
            losses_dict[out_var] = deque(maxlen=dataset.num_steps_valid_set)

    return losses_dict


def update_and_reset_loss_data(loss_data, output_names_files, mode='train'):
    """ writes latest loss aggregate values to files and resets lists """

    for out_var in output_names_files.keys():
        if mode in out_var:
            with open(output_names_files[out_var], 'a') as f:
                f.write(str(np.mean(loss_data[out_var])) + '\n')
            logger.info(out_var + ' loss: ' + str(np.mean(loss_data[out_var])))
            loss_data[out_var].clear()
    return


if __name__ == '__main__':

    logger.info("Deep Voice Training in Popart")

    parser = conf_utils.add_conf_args(run_mode='training')
    # create configuration object
    conf = conf_utils.get_conf(parser)
    # get the popart session options
    session_options = conf_utils.get_session_options(conf)
    # get the device to run on
    device = conf_utils.get_device(conf)

    # setting numpy seed
    np.random.seed(conf.numpy_seed)

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
    deep_voice_model_inputs = create_inputs_for_training(builder, conf)
    proto, loss_dict, dataflow, _, _, _ = create_model_and_dataflow_for_training(builder,
                                                                                 conf,
                                                                                 deep_voice_model_inputs)

    # experiment with learning rate schedules here
    lrs = [conf.init_lr] * conf.num_epochs
    optimizer = popart.SGD(
        {"defaultLearningRate": (lrs[0], False),
         "defaultWeightDecay": (0, True)})

    # create training session
    logger.info("Creating the training session")
    training_session, anchors = \
        conf_utils.create_session_anchors(proto,
                                          loss_dict["total_loss"],
                                          device,
                                          dataflow,
                                          session_options,
                                          training=True,
                                          optimizer=optimizer)

    if not conf.no_validation:  # Create the validation session

        logger.info("Creating the validation session")
        (validation_session,
         validation_anchors) = conf_utils.create_session_anchors(proto,
                                                                 loss_dict["total_loss"],
                                                                 device,
                                                                 dataFlow=dataflow,
                                                                 options=session_options,
                                                                 training=False)

    logger.info("Sending weights from Host")
    training_session.weightsFromHost()

    # setup dataset
    logger.info("Initializing dataset ...")
    if conf.dataset == 'VCTK':
        dataset = deep_voice_data.TransformedVCTKDataSet(conf)
    else:
        logger.error("Not a dataset type that is supported: {}".format(conf.dataset))
        sys.exit(-1)

    if not conf.no_pre_load_data:
        if not conf.generated_data:
            logger.info("Loading full training dataset into memory (this may take a few minutes)")
        all_step_data = []

        dataset_iterator = dataset.get_step_data_iterator()
        tqdm_iter = tqdm(dataset_iterator, disable=not sys.stdout.isatty())

        for step_data in tqdm_iter:
            all_step_data.append(step_data)

        if not conf.no_validation:

            val_all_step_data = []
            if not conf.generated_data:
                logger.info("Loading full validation dataset into memory")
            val_dataset_iterator = dataset.get_step_data_iterator(train_mode=False)
            val_tqdm_iter = tqdm(val_dataset_iterator,
                                 disable=not sys.stdout.isatty() or conf.generated_data)

            for val_step_data in val_tqdm_iter:
                val_all_step_data.append(val_step_data)

    output_names_files = get_output_names_files(conf)
    loss_data = prep_loss_lists(dataset, output_names_files)

    for epoch in range(conf.num_epochs):
        logger.info("Epoch # %d / %d" % (epoch + 1, conf.num_epochs))

        training_session.updateOptimizerFromHost(popart.SGD({"defaultLearningRate":
                                                            (lrs[epoch], False)}))
        logger.info("Current Learning_rate: {}".format(lrs[epoch]))

        if not conf.no_pre_load_data:
            tqdm_iter = tqdm(all_step_data, disable=not sys.stdout.isatty())
            if not conf.no_validation:
                val_tqdm_iter = tqdm(val_all_step_data,
                                     disable=not sys.stdout.isatty() or conf.generated_data)
        else:
            dataset_iterator = dataset.get_step_data_iterator()
            tqdm_iter = tqdm(dataset_iterator, disable=not sys.stdout.isatty())

            if not conf.no_validation:
                val_dataset_iterator = dataset.get_step_data_iterator(train_mode=False)
                val_tqdm_iter = tqdm(val_dataset_iterator,
                                     disable=not sys.stdout.isatty() or conf.generated_data)

        epoch_start_time = time.time()
        for step_data in tqdm_iter:
            mel_spec_data, text_data, speaker_data, mag_spec_data, done_data = step_data

            # convert raw text to sequences
            text_data = text_utils.convert_numpy_text_array_to_numpy_sequence_array(text_data,
                                                                                    conf.max_text_sequence_length)

            stepio = popart.PyStepIO(
                {
                    deep_voice_model_inputs["text_input"]: text_data,
                    deep_voice_model_inputs["mel_spec_input"]: mel_spec_data,
                    deep_voice_model_inputs["speaker_id"]: speaker_data,
                    deep_voice_model_inputs["mag_spec_input"]: mag_spec_data,
                    deep_voice_model_inputs["done_labels"]: done_data,
                }, anchors)

            training_session.run(stepio)

            # its important to append copies of anchor arrays (otherwise just references to anchors will be appended)
            loss_data['train_' + 'mel_spec'].append(np.copy(anchors[loss_dict["mel_spec_l1_loss"]]))
            loss_data['train_' + 'mag_spec'].append(np.copy(anchors[loss_dict["mag_spec_l1_loss"]]))
            loss_data['train_' + 'done_flag'].append(np.copy(anchors[loss_dict["done_flag_loss"]]))
            total_loss = (anchors[loss_dict["mel_spec_l1_loss"]] + anchors[loss_dict["mag_spec_l1_loss"]] +
                          anchors[loss_dict["done_flag_loss"]])
            if conf.use_guided_attention:
                loss_data['train_' + 'attention'].append(np.copy(anchors[loss_dict["attention_loss"]]))
                total_loss = total_loss + anchors[loss_dict["attention_loss"]]
            loss_data['train_' + 'total'].append(total_loss)

            tqdm_iter.set_description("Current training loss: " + str(np.mean(loss_data['train_' + 'total'])),
                                      refresh=sys.stdout.isatty())

        epoch_duration_secs = time.time() - epoch_start_time
        num_queries_processed = dataset.num_steps_train_set * conf.batches_per_step * conf.batch_size
        training_throughput = num_queries_processed / epoch_duration_secs
        print("Training throughput: {:.2f} Queries/Sec".format(training_throughput))

        if not conf.generated_data:
            if (epoch % conf.checkpoint_interval == 0 or (not conf.no_validation and
                                                          epoch % conf.validation_interval == 0)):
                # Saving current model to checkpoint
                ckpt_filename = os.path.join(conf.model_dir, 'checkpoint_{}.onnx'.format(epoch))
                logger.info('Saving model to {}'.format(ckpt_filename))
                training_session.modelToHost(ckpt_filename)

        # update loss info files
        update_and_reset_loss_data(loss_data, output_names_files, mode='train')

        if not conf.no_validation and (epoch % conf.validation_interval == 0) and not conf.generated_data:

            # setup validation session with current weights
            validation_session.resetHostWeights(ckpt_filename)
            validation_session.weightsFromHost()

            for val_step_data in val_tqdm_iter:

                val_mel_spec_data, val_text_data, val_speaker_data, val_mag_spec_data, val_done_data = val_step_data

                # convert raw text to sequences
                val_text_data = text_utils.convert_numpy_text_array_to_numpy_sequence_array(val_text_data,
                                                                                            conf.max_text_sequence_length)

                validation_stepio = popart.PyStepIO(
                    {
                        deep_voice_model_inputs["text_input"]: val_text_data,
                        deep_voice_model_inputs["mel_spec_input"]: val_mel_spec_data,
                        deep_voice_model_inputs["speaker_id"]: val_speaker_data,
                        deep_voice_model_inputs["mag_spec_input"]: val_mag_spec_data,
                        deep_voice_model_inputs["done_labels"]: val_done_data,
                    }, validation_anchors)

                validation_session.run(validation_stepio)

                loss_data['val_' + 'mel_spec'].append(np.copy(validation_anchors[loss_dict["mel_spec_l1_loss"]]))
                loss_data['val_' + 'mag_spec'].append(np.copy(validation_anchors[loss_dict["mag_spec_l1_loss"]]))
                loss_data['val_' + 'done_flag'].append(np.copy(validation_anchors[loss_dict["done_flag_loss"]]))
                total_val_loss = (validation_anchors[loss_dict["mel_spec_l1_loss"]] +
                                  validation_anchors[loss_dict["mag_spec_l1_loss"]] +
                                  validation_anchors[loss_dict["done_flag_loss"]])
                if conf.use_guided_attention:
                    loss_data['val_' + 'attention'].append(np.copy(validation_anchors[loss_dict["attention_loss"]]))
                    total_val_loss = total_val_loss + validation_anchors[loss_dict["attention_loss"]]
                loss_data['val_' + 'total'].append(total_val_loss)

                val_tqdm_iter.set_description("Current validation loss: " + str(np.mean(loss_data['val_' + 'total'])),
                                              refresh=sys.stdout.isatty())

            # write the training weights to the device
            training_session.resetHostWeights(ckpt_filename)
            training_session.weightsFromHost()

            update_and_reset_loss_data(loss_data, output_names_files, mode='val')
