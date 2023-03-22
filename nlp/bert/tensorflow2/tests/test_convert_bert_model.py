# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python import ipu
from transformers import BertConfig
from transformers.models.bert.modeling_tf_bert import TFBertForPreTraining
from transformers.modeling_tf_utils import TFMaskedLanguageModelingLoss, TFNextSentencePredictionLoss

from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.ipu_pretraining_model import gather_positions, IpuTFBertForPreTraining
from data_utils.wikipedia.load_wikipedia_data import get_pretraining_dataset
from model.losses import mlm_loss, nsp_loss
from model.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from utilities.assign_pipeline_stages import PipelineStagesAssigner
from utilities.ipu_utils import set_random_seeds


def get_path():
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.dirname(path)


def load_dataset(micro_batch_size, dataset_dir, seq_length=128):
    dataset, _ = get_pretraining_dataset(
        micro_batch_size=micro_batch_size,
        dataset_dir=dataset_dir,
        max_seq_length=seq_length,
        max_predictions_per_seq=20,
        vocab_size=30400,
        seed=42,
        data_type=tf.float16,
        distributed_worker_count=1,
        distributed_worker_index=1,
        generated_dataset=False,
        test=True,
    )
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
        cfg.configure_ipu_system()
        cls.strategy = ipu.ipu_strategy.IPUStrategy()
        with cls.strategy.scope():

            config = BertConfig(
                vocab_size=30400,
                num_hidden_layers=4,
                num_attention_heads=4,
                hidden_size=256,
                intermediate_size=768,
            )

            set_random_seeds(seed=42)
            cls.orig_model = TFBertForPreTraining(config)
            sample_inputs_orig = cls.sample_inputs.copy()
            masked_lm_positions = sample_inputs_orig.pop("masked_lm_positions")
            cls.orig_outputs = cls.orig_model(sample_inputs_orig)
            cls.orig_logits = gather_positions(cls.orig_outputs.prediction_logits, masked_lm_positions)

            set_random_seeds(seed=42)
            ipu_subclass_model = IpuTFBertForPreTraining(config)
            cls.sub_outputs = ipu_subclass_model(cls.sample_inputs)

            cls.functional_model = convert_tf_bert_model(
                ipu_subclass_model,
                cls.sample_inputs,
                post_process_bert_input_layer,
                replace_layers=True,
                use_outlining=True,
                embedding_serialization_factor=2,
                use_prediction_bias=True,
                use_cls_layer=True,
                use_qkv_bias=True,
                use_qkv_split=True,
                use_projection_bias=True,
            )
            cls.functional_model.compile(loss={"mlm___cls": mlm_loss, "nsp___cls": nsp_loss})
            cls.func_outputs = cls.functional_model(cls.sample_inputs)

    def test_get_pretraining_dataset(self):
        assert set(self.sample_inputs.keys()) == {
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "masked_lm_positions",
        }
        for key, val in self.sample_inputs.items():
            if key == "masked_lm_positions":
                assert tuple(val.shape) == (self.micro_batch_size, self.max_predictions_per_seq)
            else:
                assert tuple(val.shape) == (self.micro_batch_size, self.seq_length)

        assert len(self.sample_labels) == 3
        assert self.sample_labels[0].shape.as_list() == [self.micro_batch_size, self.max_predictions_per_seq]
        assert self.sample_labels[1].shape.as_list() == [self.micro_batch_size, 1]
        assert self.sample_labels[2].shape.as_list() == [self.micro_batch_size, self.seq_length]

    def test_inference_outputs(self):
        np.testing.assert_almost_equal(self.orig_logits.numpy(), self.sub_outputs.prediction_logits.numpy(), decimal=5)
        np.testing.assert_almost_equal(
            self.sub_outputs.prediction_logits.numpy(), self.func_outputs[0].numpy(), decimal=5
        )
        np.testing.assert_almost_equal(
            self.sub_outputs.seq_relationship_logits.numpy(), self.func_outputs[1].numpy(), decimal=5
        )

    def test_inference_losses(self):
        # Get the losses from our loss functions
        func_mlm_loss = self.functional_model.loss["mlm___cls"](self.sample_labels[0], self.func_outputs[0])
        func_nsp_loss = self.functional_model.loss["nsp___cls"](self.sample_labels[1], self.func_outputs[1])
        func_loss = func_mlm_loss + func_nsp_loss

        # Map the labels to the format expected by the original loss functions
        # Replace `0`s in mlm labels with `-100`s
        def replace_elements(tensor, replace, replace_with):
            mask = tf.equal(tensor, replace)
            replace = tf.reshape(tf.multiply(tf.ones(tf.size(tensor), tf.int32), replace_with), tensor.shape)
            return tf.where(mask, replace, tensor)

        # Check separate loss function implementations
        # Check MLM loss function
        orig_mlm_loss = TFMaskedLanguageModelingLoss().hf_compute_loss(
            replace_elements(self.sample_labels[2], 0, -100), self.orig_outputs["prediction_logits"]
        )
        # Mean reduce is not part of this loss function, so must be
        # applied separately.
        orig_mlm_loss = tf.reduce_mean(orig_mlm_loss)
        np.testing.assert_almost_equal(orig_mlm_loss, func_mlm_loss, decimal=5)

        # Check NSP loss function
        orig_nsp_loss = TFNextSentencePredictionLoss().hf_compute_loss(
            self.sample_labels[1], self.orig_outputs["seq_relationship_logits"]
        )
        # Mean reduce is not part of this loss function, so must be
        # applied separately.
        orig_nsp_loss = tf.reduce_mean(orig_nsp_loss)
        np.testing.assert_almost_equal(orig_nsp_loss, func_nsp_loss, decimal=5)

        # Check the separate losses summed
        orig_loss = orig_mlm_loss + orig_nsp_loss
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
        cfg.configure_ipu_system()
        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            cls.functional_model = convert_tf_bert_model(
                model,
                sample_inputs,
                post_process_bert_input_layer,
                replace_layers=True,
                use_outlining=True,
                use_prediction_bias=True,
                use_cls_layer=True,
                use_qkv_bias=True,
                use_qkv_split=True,
                use_projection_bias=True,
            )

    @pytest.mark.parametrize(
        "pipeline_stages, expected_stages",
        [
            (
                [
                    ["emb"],
                    ["hid", "hid", "hid", "hid"],
                    ["hid", "hid", "hid", "hid"],
                    ["hid", "hid", "hid", "hid", "enc_out"],
                    ["pool", "heads"],
                ],
                # The first zeros correspond to lambda layers (tf.ops) that process the input before the embeddings. The last
                # elements 4, 3, 4, 4 of the list correspond to the pooler, the layer to reduce the logits, and the two heads,
                # respectively, which are listed in this order in the model list of assignments.
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 3, 4, 4],
            ),
            (
                [
                    ["emb"],
                    ["hid", "hid"],
                    ["hid", "hid"],
                    ["hid", "hid"],
                    ["hid", "hid"],
                    ["hid", "hid"],
                    ["hid", "hid", "enc_out"],
                    ["pool", "heads"],
                ],
                # As above, the first zeros correspond to lambda layers (tf.ops), while the last 7, 6, 7, 7 elements
                # correspond to the pooler, reduce logits layer and the heads, in the order stored in the model.
                [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 6, 7, 7],
            ),
        ],
    )
    def test_pipeline(self, pipeline_stages, expected_stages):
        assignments = self.functional_model.get_pipeline_stage_assignment()
        pipeline_assigner = PipelineStagesAssigner(PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES)
        assignments = pipeline_assigner.assign_pipeline_stages(assignments, pipeline_stages)
        self.functional_model.set_pipeline_stage_assignment(assignments)
        assert [a.pipeline_stage for a in assignments] == expected_stages
