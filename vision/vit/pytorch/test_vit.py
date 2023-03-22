# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2018- The Hugging Face team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import os
import re
import subprocess
import unittest

import pytest
import requests
import torch
import yaml
from attrdict import AttrDict
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

from models import VisionTransformer


cmd_finetune = [
    "python",
    "finetune.py",
    "--config",
    "b16_cifar10",
    "--training-steps",
    "150",
    "--replication-factor",
    "2",
    "--warmup-steps",
    "20",
]


cmd_finetune_als = [
    "python",
    "finetune.py",
    "--config",
    "b16_cifar10",
    "--training-steps",
    "150",
    "--replication-factor",
    "2",
    "--warmup-steps",
    "20",
    "--auto-loss-scaling",
    "True",
]


cmd_pretrain = [
    "python",
    "pretrain.py",
    "--micro-batch-size",
    "8",
    "--config",
    "b16_in1k_pretrain",
    "--gradient-accumulation",
    "8",
    "--training-steps",
    "5",
    "--epochs",
    "0",
    "--replication-factor",
    "2",
    "--iterations",
    "2",
    "--dataset",
    "generated",
]


def run_vit(cmd):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.PIPE).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


def extract_step_metrics(line):
    loss = float(re.findall("loss:\s+([\d\.]+)", line)[0])
    acc = float(re.findall("accuracy:\s+([\d\.]+)", line)[0])
    return loss, acc


class TestViT(unittest.TestCase):
    @pytest.mark.ipus(8)
    def test_finetuning_loss(self):
        # Run default configuration
        out = run_vit(cmd_finetune)
        loss = 100.0
        for line in out.split("\n"):
            if line.find("Step: 149/") != -1:
                loss, acc = extract_step_metrics(line)
                break
        self.assertGreater(acc, 0.88)
        self.assertLess(loss, 0.31)

        # Run automatic loss scaling configuration
        out_als = run_vit(cmd_finetune_als)
        loss_als = 100.0
        for line in out_als.split("\n"):
            if line.find("Step: 149/") != -1:
                split = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                loss_als, acc_als = extract_step_metrics(line)
                break
        self.assertGreater(acc_als, 0.88)
        self.assertLess(loss_als, 0.30)

    @pytest.mark.ipus(8)
    def test_pretraining_loss(self):
        out = run_vit(cmd_pretrain)
        loss = 100.0

        for line in out.split("\n"):
            if line.find("Step: 4/") != -1:
                loss, acc = extract_step_metrics(line)
                break

        # test if pretrain normally starts with generated data
        self.assertLess(acc, 1)
        self.assertLess(loss, 15)


class TestViTModel(unittest.TestCase):
    def setUp(self):
        self.url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.config = AttrDict(
            {
                "loss": "CELoss",
                "hidden_size": 768,
                "representation_size": None,
                "attention_probs_dropout_prob": 0.1,
                "hidden_dropout_prob": 0.1,
                "drop_path_rate": 0,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "mlp_dim": 3072,
                "num_labels": 1000,
                "patches_size": 16,
                "mixup": False,
                "recompute_mid_layers": [1, 4, 7, 10],
                "byteio": False,
            }
        )
        self.pattern = {
            "patch_embeddings.projection": "patch_embeddings",
            "attention.attention": "attn",
            "attention.output.dense": "attn.out",
            "intermediate.dense": "ffn.fc1",
            "output.dense": "ffn.fc2",
            "layernorm_before": "attention_norm",
            "layernorm_after": "ffn_norm",
            "layernorm.weight": "encoder.encoder_norm.weight",
            "layernorm.bias": "encoder.encoder_norm.bias",
            "classifier": "head",
        }

        self.inputs = self.prepare_inputs()
        self.model_ref = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.model = VisionTransformer(self.config, img_size=224, num_labels=1000, representation_size=None)
        self.model.eval()

    def find_matched_key(self, src):
        if src.startswith("vit."):
            src = src[4:]
        for k, v in self.pattern.items():
            src = src.replace(k, v)
        return src

    def prepare_inputs(self):
        image = Image.open(requests.get(self.url, stream=True).raw)
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs

    def hf_inference(self):
        # https://github.com/huggingface/transformers/blob/95f888fd6a30f6d2fc5614347522eb854dcffbd6/tests/test_modeling_vit.py
        outputs = self.model_ref(**self.inputs)
        logits = outputs.logits
        return logits[0][:3]

    def inference(self):
        src = self.model_ref.named_parameters()
        dst = dict(self.model.named_parameters())

        for sname, sparam in src:
            dname = self.find_matched_key(sname)
            dst[dname].data.copy_(sparam)

        outputs = self.model(self.inputs["pixel_values"])
        return outputs[0][:3]

    def test_inference(self):
        hf_results = self.hf_inference()
        results = self.inference()
        # [-0.2744, 0.8215, -0.0836]
        self.assertTrue(torch.allclose(hf_results, results, atol=1e-4))
