# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import poptorch


def parse_option():
    parser = argparse.ArgumentParser("knn")
    parser.add_argument("--train", type=str)
    parser.add_argument("--validation", type=str)
    parser.add_argument(
        "--nb_knn", default=[10, 20], nargs="+", type=int, help="Number of NN to use. 20 is usually working the best."
    )
    parser.add_argument("--temperature", default=0.07, type=float, help="Temperature used in the voting coefficient")

    parser.add_argument("--di", type=int, default=128, help="device iterations number")

    args = parser.parse_args()
    return args


def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        # top5 does not make sense if k < 5
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


if __name__ == "__main__":
    args = parse_option()

    train_data = torch.load(args.train)
    val_data = torch.load(args.validation)
    train_features = train_data["features"]
    train_labels = train_data["labels"]
    test_features = val_data["features"]
    test_labels = val_data["labels"]

    for k in args.nb_knn:
        top1, top5 = knn_classifier(train_features, train_labels, test_features, test_labels, k, args.temperature)
        print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
