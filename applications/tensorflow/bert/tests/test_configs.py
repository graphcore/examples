# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Tests the test configs match the key parts of the main configs.
"""
import glob
import os
import json
import unittest
import pytest

THIS_MODULE_PATH = os.path.dirname(__file__)



parameters_to_check = ['batch_size', 'gradient_accumulation_count',
                       'base_learning_rate']

parameters_to_exclude = ["num_hidden_layers", "do_training", "pipeline_stages",
                         "init_checkpoint", "device_mapping", "predict_file"]


def load_config(path):
    print(path)
    with open(path[0]) as f:
        config = json.load(f)
    return config


def check_params(test_config, main_config):
    print(test_config)
    for key in test_config:
        if key not in parameters_to_exclude:
            assert test_config[key] == main_config[key], f"{key}: {test_config[key]} != {main_config[key]}"



@pytest.mark.parametrize('size', ('base', 'large'))
def test_pretrain_phase1(size):
    this_squad = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', f'pretrain_{size}_128_phase1.json'))
    that_squad = glob.glob(os.path.join(THIS_MODULE_PATH, '..', 'configs', f'pretrain_{size}_128_phase1.json'))
    test_config = load_config(this_squad)
    main_config = load_config(that_squad)

    check_params(test_config, main_config)


@pytest.mark.parametrize('size', ('base', 'large'))
def test_pretrain_phase2(size):
    this_squad = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', f'pretrain_{size}_384_phase2.json'))
    that_squad = glob.glob(os.path.join(THIS_MODULE_PATH, '..', 'configs', f'pretrain_{size}_384_phase2.json'))
    test_config = load_config(this_squad)
    main_config = load_config(that_squad)

    check_params(test_config, main_config)


@pytest.mark.parametrize('size', ('base', 'large'))
def test_squad(size):
    this_squad = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', f'squad_{size}.json'))
    that_squad = glob.glob(os.path.join(THIS_MODULE_PATH, '..', 'configs', f'squad_{size}.json'))
    test_config = load_config(this_squad)
    main_config = load_config(that_squad)

    check_params(test_config, main_config)


def test_squad_inf(size='large_inference_2ipu'):
    this_squad = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', f'squad_{size}.json'))
    that_squad = glob.glob(os.path.join(THIS_MODULE_PATH, '..', 'configs', f'squad_{size}.json'))
    test_config = load_config(this_squad)
    main_config = load_config(that_squad)

    check_params(test_config, main_config)


@pytest.mark.parametrize('size', ('base', 'large'))
def test_glue(size):
    this_squad = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', f'glue_{size}.json'))
    that_squad = glob.glob(os.path.join(THIS_MODULE_PATH, '..', 'configs', f'glue_{size}.json'))
    test_config = load_config(this_squad)
    main_config = load_config(that_squad)

    check_params(test_config, main_config)


@pytest.mark.parametrize('size', ('POD64', 'POD128'))
def test_pretrain_phase1_pod(size):
    this_squad = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', f'pretrain_large_128_phase1_{size}.json'))
    that_squad = glob.glob(os.path.join(THIS_MODULE_PATH, '..', 'configs', f'pretrain_large_128_phase1_{size}.json'))
    test_config = load_config(this_squad)
    main_config = load_config(that_squad)

    check_params(test_config, main_config)


@pytest.mark.parametrize('size', ('POD64', 'POD128'))
def test_pretrain_phase2_pod(size):
    this_squad = glob.glob(os.path.join(THIS_MODULE_PATH, 'configs', f'pretrain_large_384_phase2_{size}.json'))
    that_squad = glob.glob(os.path.join(THIS_MODULE_PATH, '..', 'configs', f'pretrain_large_384_phase2_{size}.json'))
    test_config = load_config(this_squad)
    main_config = load_config(that_squad)

    check_params(test_config, main_config)
