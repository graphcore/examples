# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
#
# This file has been modified by Graphcore Ltd.
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python import ipu
from transformers import BertConfig
from transformers.models.bert.modeling_tf_bert import TFBertForPreTraining
from transformers.modeling_tf_utils import shape_list

from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.ipu_pretraining_model import gather_positions, IpuTFBertForPreTraining
from data_utils.wikipedia.load_wikipedia_data import get_pretraining_dataset
from model.losses import mlm_loss, nsp_loss
from model.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from utilities.assign_pipeline_stages import PipelineStagesAssigner
from utilities.ipu_utils import set_random_seeds


def hf_mlm_compute_loss(labels, logits):
    """
    Adapted to our dataset with labels with non-masked-tokens marker=0.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    masked_lm_active_loss = tf.not_equal(tf.reshape(tensor=labels["mlm_long_labels"], shape=(-1,)), 0)
    masked_lm_reduced_logits = tf.boolean_mask(
        tensor=tf.reshape(tensor=logits[0], shape=(-1, shape_list(logits[0])[2])),
        mask=masked_lm_active_loss,
    )
    masked_lm_labels = tf.boolean_mask(
        tensor=tf.reshape(tensor=labels["mlm_long_labels"], shape=(-1,)), mask=masked_lm_active_loss
    )
    masked_lm_loss = loss_fn(y_true=masked_lm_labels, y_pred=masked_lm_reduced_logits)
    masked_lm_loss = tf.reduce_mean(input_tensor=masked_lm_loss)
    return masked_lm_loss


def hf_nsp_compute_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    next_sentence_active_loss = tf.not_equal(tf.reshape(tensor=labels["next_sentence_label"], shape=(-1,)), -100)
    next_sentence_reduced_logits = tf.boolean_mask(
        tensor=tf.reshape(tensor=logits[1], shape=(-1, 2)), mask=next_sentence_active_loss
    )
    next_sentence_label = tf.boolean_mask(
        tensor=tf.reshape(tensor=labels["next_sentence_label"], shape=(-1,)), mask=next_sentence_active_loss
    )
    next_sentence_loss = loss_fn(y_true=next_sentence_label, y_pred=next_sentence_reduced_logits)
    next_sentence_loss = tf.reduce_mean(input_tensor=next_sentence_loss)
    return next_sentence_loss


def get_path():
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.dirname(path)


def load_dataset(micro_batch_size,
                 dataset_dir,
                 seq_length=128):
    dataset, _ = get_pretraining_dataset(micro_batch_size=micro_batch_size,
                                         dataset_dir=dataset_dir,
                                         max_seq_length=seq_length,
                                         max_predictions_per_seq=20,
                                         distributed_worker_count=1,
                                         seed=42,
                                         data_type=tf.float16,
                                         test=True)
    iterator = iter(dataset)
    sample = next(iterator)
    return sample


class TestConvertHFToFunctionalBertPretraining:

    @classmethod
    def setup_class(cls):
        cls.micro_batch_size = 2
        cls.seq_length = 128
        cls.max_predictions_per_seq = 20
        cls.sample_inputs, cls.sample_labels = load_dataset(
            micro_batch_size=cls.micro_batch_size,
            dataset_dir=Path(get_path()).joinpath("data_utils").joinpath("wikipedia"),
            seq_length=cls.seq_length,
        )

        cfg = ipu.config.IPUConfig()
        cfg.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
        cfg.configure_ipu_system()
        cls.strategy = ipu.ipu_strategy.IPUStrategy()
        with cls.strategy.scope():

            config = BertConfig(
                vocab_size=30528,
                num_hidden_layers=4,
                num_attention_heads=4,
                hidden_size=256,
                intermediate_size=768,
                max_predictions_per_seq=cls.max_predictions_per_seq,
            )

            set_random_seeds(seed=42)
            cls.orig_model = TFBertForPreTraining(config)
            cls.orig_outputs = cls.orig_model(cls.sample_inputs)
            cls.orig_logits = gather_positions(
                cls.orig_outputs.prediction_logits,
                cls.sample_inputs['masked_lm_positions']
            )

            set_random_seeds(seed=42)
            ipu_subclass_model = IpuTFBertForPreTraining(config)
            cls.sub_outputs = ipu_subclass_model(cls.sample_inputs)

            cls.functional_model = convert_tf_bert_model(
                ipu_subclass_model,
                cls.sample_inputs,
                post_process_bert_input_layer,
                replace_layers=True,
                use_outlining=True,
                embedding_serialization_factor=2
            )
            cls.functional_model.compile(loss={'mlm___cls': mlm_loss, 'nsp___cls': nsp_loss})
            cls.func_outputs = cls.functional_model(cls.sample_inputs)

    def test_get_pretraining_dataset(self):
        assert set(self.sample_inputs.keys()) == {
            'input_ids',
            'token_type_ids',
            'attention_mask',
            'masked_lm_positions'
        }
        for key, val in self.sample_inputs.items():
            if key == 'masked_lm_positions':
                assert tuple(val.shape) == (self.micro_batch_size, self.max_predictions_per_seq)
            else:
                assert tuple(val.shape) == (self.micro_batch_size, self.seq_length)

        assert len(self.sample_labels) == 3
        assert self.sample_labels[0].shape.as_list() == [self.micro_batch_size, self.max_predictions_per_seq]
        assert self.sample_labels[1].shape.as_list() == [self.micro_batch_size, 1]
        assert self.sample_labels[2].shape.as_list() == [self.micro_batch_size, self.seq_length]

    def test_inference_outputs(self):
        np.testing.assert_almost_equal(
            self.orig_logits.numpy(),
            self.sub_outputs.prediction_logits.numpy(),
            decimal=5
        )
        np.testing.assert_almost_equal(
            self.sub_outputs.prediction_logits.numpy(),
            self.func_outputs[0].numpy(),
            decimal=5
        )
        np.testing.assert_almost_equal(
            self.sub_outputs.seq_relationship_logits.numpy(),
            self.func_outputs[1].numpy(),
            decimal=5
        )

    def test_inference_losses(self):
        labels = {
            "mlm_labels": self.sample_labels[0],
            "next_sentence_label": self.sample_labels[1],
            "mlm_long_labels": self.sample_labels[2]
        }

        orig_mlm_loss = hf_mlm_compute_loss(labels, self.orig_outputs)
        orig_nsp_loss = hf_nsp_compute_loss(labels, self.orig_outputs)
        orig_loss = orig_mlm_loss + orig_nsp_loss

        func_mlm_loss = self.functional_model.loss['mlm___cls'](labels["mlm_labels"], self.func_outputs[0])
        func_nsp_loss = self.functional_model.loss['nsp___cls'](labels["next_sentence_label"], self.func_outputs[1])
        func_loss = func_mlm_loss + func_nsp_loss

        np.testing.assert_almost_equal(orig_mlm_loss, func_mlm_loss, decimal=5)
        np.testing.assert_almost_equal(orig_nsp_loss, func_nsp_loss, decimal=5)
        np.testing.assert_almost_equal(orig_loss, func_loss, decimal=5)


class TestAssignPipelineStages:
    @classmethod
    def setup_class(cls):
        bert_config = BertConfig(max_predictions_per_seq=20)
        model = IpuTFBertForPreTraining(config=bert_config)
        sample_inputs, _ = load_dataset(
            micro_batch_size=2,
            dataset_dir=Path(get_path()).joinpath("data_utils").joinpath("wikipedia"),
            seq_length=128,
        )
        cfg = ipu.config.IPUConfig()
        cfg.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
        cfg.configure_ipu_system()
        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            cls.functional_model = convert_tf_bert_model(
                model,
                sample_inputs,
                post_process_bert_input_layer,
                replace_layers=True,
                use_outlining=True
            )

    @pytest.mark.parametrize(
        "pipeline_stages, expected_stages",
        [([["emb"],
           ["hid", "hid", "hid", "hid"],
           ["hid", "hid", "hid", "hid"],
           ["hid", "hid", "hid", "hid", "enc_out"],
           ["pool", "heads"]
           ],
          # The first zeros correspond to lambda layers (tf.ops) that process the input before the embeddings. The last
          # elements 4, 3, 4, 4 of the list correspond to the pooler, the layer to reduce the logits, and the two heads,
          # respectively, which are listed in this order in the model list of assignments.
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 3, 4, 4]),
         ([["emb"],
           ["hid", "hid"],
           ["hid", "hid"],
           ["hid", "hid"],
           ["hid", "hid"],
           ["hid", "hid"],
           ["hid", "hid", "enc_out"],
           ["pool", "heads"]
           ],
          # As above, the first zeros correspond to lambda layers (tf.ops), while the last 7, 6, 7, 7 elements
          # correspond to the pooler, reduce logits layer and the heads, in the order stored in the model.
          [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 6, 7, 7]),
         ])
    def test_pipeline(self, pipeline_stages, expected_stages):
        assignments = self.functional_model.get_pipeline_stage_assignment()
        pipeline_assigner = PipelineStagesAssigner(PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES)
        assignments = pipeline_assigner.assign_pipeline_stages(assignments, pipeline_stages)
        self.functional_model.set_pipeline_stage_assignment(assignments)
        assert [a.pipeline_stage for a in assignments] == expected_stages
