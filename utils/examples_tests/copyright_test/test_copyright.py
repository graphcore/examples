# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import argparse
import datetime
import fileinput
import inspect
import os
import re
import sys
import configparser
from pathlib import Path


C_FILE_EXTS = ['c', 'cpp', 'C', 'cxx', 'c++', 'h', 'hpp']

EXCLUDED = [
    'gnn/cluster_gcn/tensorflow2/run_cluster_gcn_notebook.ipynb',
    'vision/cnns/tensorflow1/training/Datasets/imagenet_preprocessing.py',
    'vision/cnns/tensorflow1/inference/models/optimize_for_infer.py',
    'nlp/bert/popart/bert_data/tokenization.py',
    'nlp/bert/popart/bert_data/squad_utils.py',
    'nlp/bert/popart/bert_data/create_pretraining_data.py',
    'nlp/bert/popart/tests/torch_bert.py',
    'nlp/gpt/popxl/data/WikicorpusTextFormatting.py',
    'miscellaneous/monte_carlo_ray_tracing/poplar/light',
    'recommendation/click_through_rate/tensorflow1/common/Dice.py',
    'vision/ssd/tensorflow1/bounding_box_utils/bounding_box_utils.py',
    'vision/ssd/tensorflow1/keras_layers/keras_layer_DecodeDetections.py',
    'vision/ssd/tensorflow1/keras_layers/keras_layer_L2Normalization.py',
    'vision/ssd/tensorflow1/keras_layers/keras_layer_AnchorBoxes.py',
    'ai_for_simulation/cosmoflow/tensorflow1/models/resnet.py',
    'ai_for_simulation/cosmoflow/tensorflow1/data_gen/__init__.py',
    'ai_for_simulation/cosmoflow/tensorflow1/models/__init__.py',
    'ai_for_simulation/cosmoflow/tensorflow1/models/cosmoflow.py',
    'ai_for_simulation/cosmoflow/tensorflow1/models/cosmoflow_v1.py',
    'ai_for_simulation/cosmoflow/tensorflow1/models/layers.py',
    'ai_for_simulation/cosmoflow/tensorflow1/utils/argparse.py',
    'vision/yolo_v3/tensorflow1/tests/original_model/backbone.py',
    'vision/yolo_v3/tensorflow1/tests/original_model/__init__.py',
    'vision/yolo_v3/tensorflow1/tests/original_model/common.py',
    'vision/yolo_v3/tensorflow1/tests/original_model/config.py',
    'vision/yolo_v3/tensorflow1/tests/original_model/yolov3.py',
    "vision/efficientdet/tensorflow2/backbone",
    "vision/efficientdet/tensorflow2/tf2",
    "vision/efficientdet/tensorflow2/object_detection",
    "vision/efficientdet/tensorflow2/visualize",
    "vision/efficientdet/tensorflow2/dataloader.py",
    "vision/efficientdet/tensorflow2/nms_np.py",
    "vision/cnns/pytorch/datasets/libjpeg-turbo",
    "vision/cnns/pytorch/datasets/turbojpeg",
    'speech/transformer_transducer/popart/training/utils/preprocessing_utils.py',
    'speech/transformer_transducer/popart/training/utils/download_librispeech.py',
    'speech/transformer_transducer/popart/training/utils/download_utils.py',
    'speech/transformer_transducer/popart/training/utils/convert_librispeech.py',
    'speech/transformer_transducer/popart/training/common/metrics.py',
    'speech/transformer_transducer/popart/training/common/rnn.py',
    'speech/transformer_transducer/popart/training/common/helpers.py',
    'speech/transformer_transducer/popart/training/common/data/features.py',
    'speech/transformer_transducer/popart/training/common/data/helpers.py',
    'speech/transformer_transducer/popart/training/common/data/text.py',
    'speech/transformer_transducer/popart/training/common/data/__init__.py',
    'speech/transformer_transducer/popart/training/common/data/dali/pipeline.py',
    'speech/transformer_transducer/popart/training/common/data/dali/iterator.py',
    'speech/transformer_transducer/popart/training/common/data/dali/sampler.py',
    'speech/transformer_transducer/popart/training/common/data/dali/data_loader.py',
    'speech/transformer_transducer/popart/training/common/data/dali/__init__.py',
    'speech/transformer_transducer/popart/training/common/text/cleaners.py',
    'speech/transformer_transducer/popart/training/common/text/numbers.py',
    'speech/transformer_transducer/popart/training/common/text/symbols.py',
    'speech/transformer_transducer/popart/training/common/text/__init__.py',
    'speech/transformer_transducer/popart/training/rnnt_reference/model.py',
    'speech/transformer_transducer/popart/training/rnnt_reference/config.py',
    'speech/transformer_transducer/popart/custom_ops/feat_augmentation/torch_random.h',
    'speech/transformer_transducer/popart/custom_ops/feat_augmentation/np_fp16_convert.hpp',
    'speech/transformer_transducer/popart/custom_ops/feat_augmentation/np_fp16_convert.cpp',
    'speech/transformer_transducer/popart/custom_ops/feat_augmentation/torch_random.cpp',
    'speech/transformer_transducer/popart/custom_ops/feat_augmentation/MT19937RNGEngine.h',
    'speech/transformer_transducer/popart/custom_ops/rnnt_loss/torch_reference/transducer.py',
    'speech/transformer_transducer/popart/custom_ops/rnnt_loss/torch_reference/setup.py',
    'speech/transformer_transducer/popart/custom_ops/rnnt_loss/torch_reference/transducer.cpp',
    'speech/transformer_transducer/popart/custom_ops/rnnt_loss/torch_reference/ref_transduce.py',
    'speech/transformer_transducer/popart/custom_ops/rnnt_loss/torch_reference/test.py',
    'speech/transformer_transducer/popart/custom_ops/rnnt_loss/test/transducer.py',
    'speech/fastpitch/pytorch/waveglow/loss_function.py',
    'speech/fastpitch/pytorch/waveglow/model.py',
    'speech/fastpitch/pytorch/waveglow/denoiser.py',
    'speech/fastpitch/pytorch/waveglow/data_function.py',
    'speech/fastpitch/pytorch/waveglow/arg_parser.py',
    'speech/fastpitch/pytorch/tacotron2/model.py',
    'speech/fastpitch/pytorch/tacotron2/loss_function.py',
    'speech/fastpitch/pytorch/tacotron2/data_function.py',
    'speech/fastpitch/pytorch/tacotron2/arg_parser.py',
    'speech/fastpitch/pytorch/pitch_transform.py',
    'speech/fastpitch/pytorch/models.py',
    'speech/fastpitch/pytorch/loss_functions.py',
    'speech/fastpitch/pytorch/inference.py',
    'speech/fastpitch/pytorch/fastpitch/transformer_jit.py',
    'speech/fastpitch/pytorch/fastpitch/transformer.py',
    'speech/fastpitch/pytorch/fastpitch/model_jit.py',
    'speech/fastpitch/pytorch/fastpitch/arg_parser.py',
    'speech/fastpitch/pytorch/extract_mels.py',
    'speech/fastpitch/pytorch/data_functions.py',
    'speech/fastpitch/pytorch/common/text/text_processing.py',
    'speech/fastpitch/pytorch/common/text/symbols.py',
    'speech/fastpitch/pytorch/common/text/numerical.py',
    'speech/fastpitch/pytorch/common/text/letters_and_numbers.py',
    'speech/fastpitch/pytorch/common/text/datestime.py',
    'speech/fastpitch/pytorch/common/text/cmudict.py',
    'speech/fastpitch/pytorch/common/text/cleaners.py',
    'speech/fastpitch/pytorch/common/text/acronyms.py',
    'speech/fastpitch/pytorch/common/text/abbreviations.py',
    'speech/fastpitch/pytorch/common/text/__init__.py',
    'speech/fastpitch/pytorch/common/tb_dllogger.py',
    'speech/fastpitch/pytorch/common/stft.py',
    'speech/fastpitch/pytorch/common/layers.py',
    'speech/fastpitch/pytorch/common/audio_processing.py'
]


def check_file(path, language, amend):
    if os.stat(path).st_size == 0:
        # Empty file
        return True

    comment = "#" if language == "python" else "//"
    found_copyright = False
    first_line_index = 0
    with open(path, "r") as f:
        regexp = r"{} Copyright \(c\) \d+ Graphcore Ltd. All (r|R)ights (r|R)eserved.".format(comment)

        # Skip blank, comments and shebang
        for line in f:
            can_ignore_line = line == '\n' or line.startswith(comment) or line.startswith("#!")
            if (not can_ignore_line) or re.match(regexp, line):
                break
            if line.startswith("#!"):
                first_line_index += 1

        # Check first line after skips
        if re.match(regexp, line):
            found_copyright = True

    if not found_copyright:
        if amend:
            now = datetime.datetime.now()
            year = now.year
            copyright_msg = '{} Copyright (c) {} Graphcore Ltd. All rights reserved.'.format(comment, year)
            index = 0
            for line in fileinput.FileInput(path, inplace=1):
                if index == first_line_index:
                    line = copyright_msg + line
                print(line[:-1])
                index += 1

        return False

    return True


def read_git_submodule_paths():
    try:
        config = configparser.ConfigParser()
        config.read('.gitmodules')
        module_paths = [config[k]['path'] for k in config.sections()]
        print(f"Git submodule paths: {module_paths}")
        return module_paths
    except:
        print(f"No Git submodules found.")
        return []


def is_child_of_excluded(file_path):
    excluded = [Path(p).absolute() for p in EXCLUDED]
    return any(e in Path(file_path).parents for e in excluded if e.is_dir())


def test_copyrights(amend=False):
    """A test to ensure that every source file has the correct Copyright"""
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    root_path = os.path.abspath(os.path.join(cwd, "..", "..", ".."))

    git_module_paths = read_git_submodule_paths()

    bad_files = []
    excluded = [os.path.join(root_path, p) for p in EXCLUDED]
    for path, _, files in os.walk(root_path):
        for file_name in files:
            file_path = os.path.join(path, file_name)

            if file_path in excluded or is_child_of_excluded(file_path):
                continue

            # CMake builds generate .c and .cpp files
            # so we need to exclude all those:
            if '/CMakeFiles/' in file_path:
                continue

            # Also exclude git submodules from copyright checks:
            if any(path in file_path for path in git_module_paths):
                continue

            if file_name.endswith('.py'):
                if not check_file(file_path, "python", amend):
                    bad_files.append(file_path)

            if file_name.split('.')[-1] in C_FILE_EXTS:
                if not check_file(file_path, "c", amend):
                    bad_files.append(file_path)

    if len(bad_files) != 0:
        sys.stderr.write("ERROR: The following files do not have "
                         "copyright notices:\n\n")
        for f in bad_files:
            sys.stderr.write("    {}\n".format(f))
        raise RuntimeError(f"{len(bad_files)} files do not have copyright notices: {bad_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copyright header test")
    parser.add_argument("--amend",
                        action="store_true",
                        help="Amend copyright headers in files.")

    opts = parser.parse_args()
    try:
        test_copyrights(opts.amend)
    except AssertionError:
        sys.exit(1)
