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


import pytest
import os
import tensorflow as tf


class TestModel:

    def test_load_yaml_config(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        assert config['batch_size']

    def test_build_computation_stages(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        model = ConformerAM(config)
        model._build_computational_stages()

    def test_get_pipeline_depth(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file, **{'data_path': './train', 'dict_path': './sample_train_units.txt', 'use_synthetic_data': True})
        model = ConformerAM(config)
        model._build_dataset()
        model._build_computational_stages()
        assert model.get_pipeline_depth() == 36

    @pytest.mark.parametrize("maxlen, dmodel", [(32, 100), (33, 200), (200, 256)])
    def test_pos_embedding_build(self, maxlen, dmodel):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        model = ConformerAM(config)
        pos_emb = model._build_pos_embedding(maxlen, dmodel)
        batch_size = 1
        with tf.Session() as sess:
            res = sess.run(pos_emb)
            assert res.shape == (batch_size, maxlen, dmodel)

    @pytest.mark.parametrize("batch_size, feature_dim, scope", [(1, 144, 'test1'), (2, 256, 'test2')])
    def test_encoder_embedding_build(self, batch_size, feature_dim, scope):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        model = ConformerAM(config)
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        input_len = tf.constant([model.config['maxlen_in'], ])
        input = tf.random_normal(shape=(batch_size, model.config['maxlen_in'], feature_dim, 1))
        encoder_embedding = model._build_encoder_embedding(input, input_len, scope)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x, pos_emb, mask_adder = sess.run(encoder_embedding)
            assert x.shape == (batch_size, model.config['maxlen_in'] // 4 - 1, model.config['adim'])
            assert pos_emb.shape == (1, (model.config['maxlen_in'] // 4 - 1), model.config['adim'])
            assert mask_adder.shape == (1, 1, 1, model.config['maxlen_in'] // 4 - 1)

    @pytest.mark.parametrize("batch_size, lq, lv, scope_name", [(1, 10, 20, 'test1'), (2, 20, 30, 'test2')])
    def test_self_attention_build(self, batch_size, lq, lv, scope_name):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        model = ConformerAM(config)
        query = tf.random_normal(shape=(batch_size, lq, model.config['adim']))
        key = tf.random_normal(shape=(batch_size, lv, model.config['adim']))
        value = tf.random_normal(shape=(batch_size, lv, model.config['adim']))

        self_attention = model._build_self_attention(query, key, value, scope_name)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(self_attention)
            assert output.shape == (batch_size, lq, model.config['adim'])

    @pytest.mark.parametrize("batch_size, length, scope_name", [(1, 65, 'test1')])
    def test_conv_build(self, batch_size, length, scope_name):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        model = ConformerAM(config)
        input = tf.random_normal(shape=(batch_size, length, model.config['adim']))
        conv = model._build_conv_module(input, scope_name)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(conv)
            assert output.shape == (batch_size, length, model.config['adim'])

    @pytest.mark.parametrize("batch_size, n_head, length_q, length_v", [(1, 1, 4, 5)])
    def test_relative_shift_build(self, batch_size, n_head, length_q, length_v):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        model = ConformerAM(config)
        input = tf.ones(shape=(batch_size, n_head, length_q, length_v))
        shift = model._relative_shift(input)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(shift)
            assert output.shape == (batch_size, n_head, length_q, length_v)

    @pytest.mark.parametrize("batch_size, prefix", [(1, 'test_encoder1')])
    def test_encoder_layer_build(self, batch_size, prefix):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        model = ConformerAM(config)
        x = tf.random_normal(shape=(batch_size, model.config['maxlen_in'], model.config['adim']))
        mask_adder = tf.random_normal(shape=(batch_size, 1, 1, model.config['maxlen_in']))
        pos_emb = tf.random_normal(shape=(1, model.config['maxlen_in'], model.config['adim']))
        layer = model._build_encoder_layer(x, mask_adder, pos_emb, prefix)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(layer)
            assert output.shape == (batch_size, model.config['maxlen_in'], model.config['adim'])

    @pytest.mark.parametrize("batch_size, prefix", [(1, 'test_decoder1')])
    def test_decoder_layer_build(self, batch_size, prefix):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from model import AMConfig
        from model import ConformerAM
        from util import get_config
        cfg = get_config(1)
        cfg.configure_ipu_system()
        config_file = 'configs/train_fp32_kl_loss.yaml'
        config = AMConfig.from_yaml(config_file)
        model = ConformerAM(config)
        tgt = tf.ones(shape=(batch_size, model.config['maxlen_tgt'], model.config['adim']))
        tgt_mask = tf.random_normal(shape=(batch_size, 1, model.config['maxlen_tgt'], model.config['maxlen_tgt']))
        mem = tf.random_normal(shape=(batch_size, model.config['maxlen_in'], model.config['adim']))
        mem_mask = tf.random_normal(shape=(batch_size, 1, 1, model.config['maxlen_in']))
        layer = model._build_decoder_layer(tgt, tgt_mask, mem, mem_mask, prefix)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(layer)
            assert output.shape == (batch_size, model.config['maxlen_tgt'], model.config['adim'])
