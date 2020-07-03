# Copyright 2020 Graphcore Ltd.
import pytest
import numpy as np
from unittest.mock import Mock
from tempfile import TemporaryDirectory
import os
import subprocess
import re

import popart
import conf_utils
from deep_voice_train import create_inputs_for_training
from deep_voice_model import PopartDeepVoice


def get_test_conf(batch_size, replication_factor, num_ipus):
    conf = Mock()
    conf.dataset = 'VCTK'
    conf.precision = np.float32
    conf.batch_size = batch_size
    conf.replication_factor = replication_factor
    conf.samples_per_device = int(conf.batch_size / conf.replication_factor)
    conf.batches_per_step = 50
    conf.simulation = False
    conf.select_ipu = 'AUTO'
    conf.num_ipus = num_ipus
    return conf


def assert_lists_equal(alist, blist):
    assert(all([a == b for a, b in zip(alist, blist)]))


@pytest.mark.category1
@pytest.mark.parametrize("batch_size,replication_factor,num_ipus", [(2, 1, 1), (4, 2, 2)])
def test_train_graph_build(batch_size, replication_factor, num_ipus):
    """ testing build for deep-voice training graph for different batch sizes and replication factors """
    builder = popart.Builder()
    conf = get_test_conf(batch_size=batch_size, replication_factor=replication_factor, num_ipus=num_ipus)
    conf = conf_utils.set_model_conf(conf, print_model_conf=False)

    deep_voice_model_inputs = create_inputs_for_training(builder, conf)
    deep_voice_model = PopartDeepVoice(conf, builder,
                                       for_inference=False)

    main_outputs, aux_outputs, name_to_tensor = deep_voice_model(deep_voice_model_inputs["text_input"],
                                                                 deep_voice_model_inputs["mel_spec_input"],
                                                                 deep_voice_model_inputs["speaker_id"])

    # checking if all outputs exist
    assert(len(main_outputs) == 3)
    assert("mel_spec_output" in main_outputs)
    assert("mag_spec_output" in main_outputs)
    assert("done_flag_output" in main_outputs)

    assert(len(aux_outputs) == 2)
    assert("attention_scores_arrays" in aux_outputs)
    assert("speaker_embedding_matrix" in aux_outputs)

    # checking if all output shapes are correct
    assert_lists_equal(builder.getTensorShape(main_outputs["mel_spec_output"]),
                       [conf.samples_per_device,
                        conf.mel_bands,
                        conf.max_spectrogram_length])
    assert_lists_equal(builder.getTensorShape(main_outputs["mag_spec_output"]),
                       [conf.samples_per_device,
                        conf.n_fft // 2 + 1,
                        conf.max_spectrogram_length])
    assert_lists_equal(builder.getTensorShape(main_outputs["done_flag_output"]),
                       [conf.samples_per_device,
                        1,
                        conf.max_spectrogram_length])

    for att_dist in aux_outputs["attention_scores_arrays"]:
        assert_lists_equal(builder.getTensorShape(att_dist),
                           [conf.samples_per_device, conf.max_text_sequence_length, conf.max_spectrogram_length])
    assert_lists_equal(builder.getTensorShape(aux_outputs["speaker_embedding_matrix"]),
                       [conf.num_speakers, conf.speaker_embedding_dim])


@pytest.mark.ipus(2)
@pytest.mark.category2
@pytest.mark.parametrize("batch_size,replication_factor,num_ipus", [(2, 1, 1), (4, 2, 2)])
def test_deep_voice_train(batch_size, replication_factor, num_ipus):

    with TemporaryDirectory() as tmp_dir:
        cmd = ["python3", "deep_voice_train.py"]
        args = "--data_dir {} --model_dir {} --generated_data --num_epochs 1 " \
               "--batch_size {} --replication_factor {} --num_ipus {}".format(tmp_dir, tmp_dir,
                                                                              batch_size,
                                                                              replication_factor,
                                                                              num_ipus)
        args = args.split(" ")
        cmd.extend(args)
        output = subprocess.check_output(cmd, cwd=os.path.dirname(__file__)).decode("utf-8")
        strings_to_match = ["Training throughput", "Queries/Sec"]
        regexes = [re.compile(s) for s in strings_to_match]
        for i, r in enumerate(regexes):
            match = r.search(output)
            assert match, "Output of command: '{}' contained no match for: {} " \
                          "\nOutput was:\n{}".format(cmd, strings_to_match[i], output)
