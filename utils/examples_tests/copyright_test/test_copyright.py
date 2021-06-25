# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import argparse
import datetime
import fileinput
import inspect
import os
import re
import sys
import configparser

C_FILE_EXTS = ['c', 'cpp', 'C', 'cxx', 'c++', 'h', 'hpp']


EXCLUDED = ['applications/tensorflow/cnns/training/Datasets/imagenet_preprocessing.py',
            'applications/tensorflow/cnns/inference/ssd/bounding_box_utils/bounding_box_utils.py',
            'applications/tensorflow/cnns/inference/ssd/keras_layers/keras_layer_DecodeDetections.py',
            'applications/tensorflow/cnns/inference/ssd/keras_layers/keras_layer_L2Normalization.py',
            'applications/tensorflow/cnns/inference/ssd/keras_layers/keras_layer_AnchorBoxes.py',
            'applications/tensorflow/cnns/models/optimize_for_infer.py',
            'applications/popart/bert/bert_data/tokenization.py',
            'applications/popart/bert/bert_data/squad_utils.py',
            'applications/popart/bert/bert_data/create_pretraining_data.py',
            'applications/popart/bert/tests/torch_bert.py',
            'applications/tensorflow/click_through_rate/common/Dice.py',
            'code_examples/tensorflow/cosmoflow/models/resnet.py']


def check_file(path, language, amend):
    comment = "#" if language == "python" else "//"
    found_copyright = False
    first_line_index = 0
    empty_file = False
    with open(path, "r") as f:
        first_line = f.readline()
        # if the first line is encoding, then read the second line
        if first_line.startswith("{} coding=utf-8".format(comment)):
            first_line = f.readline()

        if first_line == '':
            empty_file = True

        if language == "python" and first_line.startswith("#!"):
            first_line_index += 1
            first_line = f.readline()

        regexp = r"{} Copyright \(c\) \d+ Graphcore Ltd. All (r|R)ights (r|R)eserved.".format(comment)

        if re.match(regexp, first_line):
            found_copyright = True

    if not empty_file and not found_copyright:
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

            if file_path in excluded:
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
