# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
from tensorflow.python import ipu
from transformers import TFBertForQuestionAnswering

from data_utils.squad_v1.load_squad_data import get_squad_data
from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.losses import QuestionAnsweringLossFunction, question_answering_loss
from utilities.ipu_utils import set_random_seeds
from utilities.metric_enqueuer import wrap_loss_in_enqueuer


def load_dataset(micro_batch_size):
    train_dataset, _, _, _, _ = get_squad_data(micro_batch_size, "./cache/", generated_dataset=True)
    iterator = iter(train_dataset)
    sample = next(iterator)
    return sample


class TestConvertHFToFunctionalBertFineTuning:
    @classmethod
    def setup_class(cls):
        cls.micro_batch_size = 2
        cls.seq_length = 128
        cls.max_predictions_per_seq = 20
        cls.max_position_embeddings = 512
        cls.sample_inputs, cls.sample_outputs = load_dataset(
            micro_batch_size=cls.micro_batch_size,
        )

        cfg = ipu.config.IPUConfig()
        cfg.configure_ipu_system()
        cls.strategy = ipu.ipu_strategy.IPUStrategy()
        with cls.strategy.scope():

            set_random_seeds(seed=42)
            cls.orig_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased")
            cls.orig_outputs = cls.orig_model(cls.sample_inputs)

            set_random_seeds(seed=42)
            ipu_subclass_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased")
            cls.sub_outputs = ipu_subclass_model(cls.sample_inputs)
            # Convert to functional model and replace some layers to optimise performance.
            cls.functional_model = convert_tf_bert_model(
                ipu_subclass_model,
                cls.sample_inputs,
                post_process_bert_input_layer,
                replace_layers=True,
                use_outlining=True,
                embedding_serialization_factor=1,
                rename_outputs={"tf.compat.v1.squeeze": "start_positions", "tf.compat.v1.squeeze_1": "end_positions"},
                use_prediction_bias=True,
                use_cls_layer=True,
            )
            qa_loss = wrap_loss_in_enqueuer(
                QuestionAnsweringLossFunction, ["end_positions_loss", "start_positions_loss"]
            )()
            cls.functional_model.compile(
                loss={"end_positions": qa_loss, "start_positions": qa_loss},
            )
            cls.func_outputs = cls.functional_model(cls.sample_inputs)

    def test_get_squad_dataset(self):
        assert set(self.sample_inputs.keys()) == {
            "input_ids",
            "token_type_ids",
            "attention_mask",
        }

        for key, val in self.sample_inputs.items():
            assert tuple(val.shape) == (self.micro_batch_size, self.max_position_embeddings)

        assert len(self.sample_outputs) == 2
        assert self.sample_outputs["start_positions"].shape.as_list()[0] == self.micro_batch_size
        assert self.sample_outputs["end_positions"].shape.as_list()[0] == self.micro_batch_size

    def test_inference_outputs(self):
        np.testing.assert_almost_equal(
            self.orig_outputs["start_logits"].numpy(), self.sub_outputs["start_logits"].numpy(), decimal=5
        )
        np.testing.assert_almost_equal(
            self.orig_outputs["end_logits"].numpy(), self.sub_outputs["end_logits"].numpy(), decimal=5
        )

    def test_inference_losses(self):
        labels = {
            "start_positions": self.sample_outputs["start_positions"],
            "end_positions": self.sample_outputs["end_positions"],
        }

        orig_start_loss = question_answering_loss(labels["start_positions"], self.orig_outputs[0])
        orig_end_loss = question_answering_loss(labels["end_positions"], self.orig_outputs[1])
        func_start_loss = question_answering_loss(labels["start_positions"], self.sub_outputs[0])
        func_end_loss = question_answering_loss(labels["end_positions"], self.sub_outputs[1])

        np.testing.assert_almost_equal(orig_start_loss, func_start_loss, decimal=5)
        np.testing.assert_almost_equal(orig_end_loss, func_end_loss, decimal=5)
