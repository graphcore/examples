# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
from tensorflow.python import ipu
from transformers import TFBertForSequenceClassification

from data_utils.glue.load_glue_data import get_glue_data
from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.losses import classification_loss, ClassificationLossFunction
from utilities.ipu_utils import set_random_seeds
from utilities.metric_enqueuer import wrap_loss_in_enqueuer


def load_dataset(task_name, micro_batch_size, generated=False):
    data_output = get_glue_data(
        task=task_name, micro_batch_size=micro_batch_size, cache_dir="./cache/", generated_dataset=generated
    )
    train_dataset = data_output[0]
    raw_dataset = data_output[-1]
    iterator = iter(train_dataset)
    sample = next(iterator)
    return sample, raw_dataset


class TestConvertHFToFunctionalBertClassification:
    @classmethod
    def setup_class(cls):
        cls.micro_batch_size = 2
        cls.seq_length = 128
        cls.max_predictions_per_seq = 20
        cls.max_position_embeddings = 512
        samples, raw_dataset = load_dataset("mrpc", micro_batch_size=cls.micro_batch_size, generated="generated")
        cls.sample_inputs, cls.sample_outputs = samples
        cfg = ipu.config.IPUConfig()
        cfg.configure_ipu_system()
        cls.strategy = ipu.ipu_strategy.IPUStrategy()
        with cls.strategy.scope():

            set_random_seeds(seed=42)
            cls.orig_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
            cls.orig_outputs = cls.orig_model(cls.sample_inputs)

            set_random_seeds(seed=42)
            ipu_subclass_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
            cls.sub_outputs = ipu_subclass_model(cls.sample_inputs)
            # Convert to functional model and replace some layers to optimise performance.
            cls.functional_model = convert_tf_bert_model(
                ipu_subclass_model,
                cls.sample_inputs,
                post_process_bert_input_layer,
                replace_layers=True,
                use_outlining=True,
                embedding_serialization_factor=1,
                rename_outputs={"classifier": "labels"},
                use_cls_layer=True,
            )
            qa_loss = wrap_loss_in_enqueuer(ClassificationLossFunction, ["labels"])()
            cls.functional_model.compile(loss={"labels": qa_loss})
            cls.func_outputs = cls.functional_model(cls.sample_inputs)

    def test_inference_outputs(self):
        np.testing.assert_almost_equal(
            self.orig_outputs["logits"].numpy(), self.sub_outputs["logits"].numpy(), decimal=5
        )

    def test_inference_losses(self):
        orig_start_loss = classification_loss(self.sample_outputs["labels"], self.orig_outputs["logits"])
        func_start_loss = classification_loss(self.sample_outputs["labels"], self.sub_outputs["logits"])
        np.testing.assert_almost_equal(orig_start_loss, func_start_loss, decimal=5)


@pytest.mark.parametrize("task_name,task", [("cola", ("sentence",)), ("rte", ("sentence1", "sentence2"))])
def test_get_glue_dataset(task_name, task):
    micro_batch_size = 2
    sample, raw_dataset = load_dataset(task_name, micro_batch_size)
    sample_inputs = sample[0]
    sample_outputs = sample[1]
    assert set(sample_inputs.keys()) == {
        "input_ids",
        "token_type_ids",
        "attention_mask",
    }

    assert len(sample_outputs) == 1
    assert sample_outputs["labels"].shape.as_list()[0] == micro_batch_size
    assert set(task).issubset(list(raw_dataset["train"].element_spec.keys()))
