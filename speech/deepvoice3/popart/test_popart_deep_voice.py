# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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


def get_test_conf(global_batch_size, replication_factor, num_ipus, num_io_tiles):
    conf = Mock()
    conf.dataset = 'VCTK'
    conf.precision = np.float32
    conf.global_batch_size = global_batch_size
    conf.replication_factor = replication_factor
    conf.replica_batch_size = int(conf.global_batch_size / conf.replication_factor)
    conf.device_iterations = 50
    conf.simulation = False
    conf.select_ipu = 'AUTO'
    conf.num_ipus = num_ipus
    conf.num_io_tiles = num_io_tiles
    return conf


def assert_lists_equal(alist, blist):
    assert(all([a == b for a, b in zip(alist, blist)]))


@pytest.mark.ipus(2)
@pytest.mark.parametrize("global_batch_size,replication_factor,num_ipus,num_io_tiles", [(2, 1, 1, 0), (4, 2, 2, 0), (8, 1, 1, 32)])
def test_train_graph_build(global_batch_size, replication_factor, num_ipus, num_io_tiles):
    """ testing build for deep-voice training graph for different global batch sizes and replication factors """
    builder = popart.Builder()
    conf = get_test_conf(global_batch_size=global_batch_size, replication_factor=replication_factor, num_ipus=num_ipus, num_io_tiles=num_io_tiles)
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
                       [conf.replica_batch_size,
                        conf.mel_bands,
                        conf.max_spectrogram_length])
    assert_lists_equal(builder.getTensorShape(main_outputs["mag_spec_output"]),
                       [conf.replica_batch_size,
                        conf.n_fft // 2 + 1,
                        conf.max_spectrogram_length])
    assert_lists_equal(builder.getTensorShape(main_outputs["done_flag_output"]),
                       [conf.replica_batch_size,
                        1,
                        conf.max_spectrogram_length])

    for att_dist in aux_outputs["attention_scores_arrays"]:
        assert_lists_equal(builder.getTensorShape(att_dist),
                           [conf.replica_batch_size, conf.max_text_sequence_length, conf.max_spectrogram_length])
    assert_lists_equal(builder.getTensorShape(aux_outputs["speaker_embedding_matrix"]),
                       [conf.num_speakers, conf.speaker_embedding_dim])


@pytest.mark.ipus(2)
@pytest.mark.parametrize("global_batch_size,replication_factor,num_ipus,num_io_tiles", [(2, 1, 1, 0), (4, 2, 2, 0), (8, 1, 1, 32)])
@pytest.mark.ipu_version("ipu2")
def test_deep_voice_train(global_batch_size, replication_factor, num_ipus, num_io_tiles):

    with TemporaryDirectory() as tmp_dir:
        cmd = ["python3", "deep_voice_train.py"]
        args = "--data_dir {} --model_dir {} --generated_data --num_epochs 1 " \
               "--global_batch_size {} --replication_factor {} --num_ipus {} " \
               "--num_io_tiles {}".format(tmp_dir, tmp_dir,
                                          global_batch_size,
                                          replication_factor,
                                          num_ipus,
                                          num_io_tiles,
                                          )
        args = args.split(" ")
        cmd.extend(args)

        try:
            output = subprocess.check_output(
                cmd, cwd=os.path.dirname(__file__), stderr=subprocess.PIPE
            ).decode("utf-8")
        except subprocess.CalledProcessError as e:
            print(f"TEST FAILED")
            print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
            print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
            raise

        strings_to_match = ["Training throughput", "Queries/Sec"]
        regexes = [re.compile(s) for s in strings_to_match]
        for i, r in enumerate(regexes):
            match = r.search(output)
            assert match, "Output of command: '{}' contained no match for: {} " \
                          "\nOutput was:\n{}".format(cmd, strings_to_match[i], output)
