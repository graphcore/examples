# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

import popart
import os
import numpy as np
from collections import defaultdict
import pytest

from bert import (set_library_seeds,
                  bert_config_from_args,
                  bert_pretrained_initialisers,
                  bert_add_inputs,
                  bert_add_logit_outputs,
                  get_bert_dataset,
                  Iteration,
                  bert_required_ipus,
                  bert_inference_session,
                  create_callback_stepio,
                  bert_process_infer_data)
from bert_model import Bert
from tests.utils import TestFailureError, bert_root_dir
import logging
import utils
from utils.device import acquire_device

'''
Tests the Embedding and Positional Dict lookup on the host against the same on the IPU.
Executes just the Embedding and Positional parts of the Embedding layer and compares the result.
'''

logger = logging.getLogger('BERT')


def run_embedding_layer(args):
    set_library_seeds(args.seed)

    config = bert_config_from_args(args)

    initializers = bert_pretrained_initialisers(config, args)

    logger.info("Building Model")
    # Specifying ai.onnx opset9 for the slice syntax
    # TODO: Change slice to opset10
    model = Bert(config,
                 builder=popart.Builder(
                     opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}),
                 initializers=initializers,
                 execution_mode=args.execution_mode)

    # If config.host_embedding is enabled, indices and positions will have the matrices instead of the index vector.
    indices, positions, segments, masks, labels = bert_add_inputs(args, model)
    logits = tuple([model.embedding(indices, positions, segments)])

    if args.inference:
        outputs = bert_add_logit_outputs(model, logits)
        writer = None

        dataset = get_bert_dataset(model, args, [indices, positions, segments, masks, labels])

        data_flow = popart.DataFlow(dataset.batches_per_step, outputs)

        iteration = Iteration(
            args,
            steps_per_epoch=len(dataset),
            writer=writer,
            recording_steps=args.aggregate_metrics_over_steps)

        request_ipus = bert_required_ipus(args, model)

        device = acquire_device(args, request_ipus)

        session, anchors = bert_inference_session(model, args, data_flow, device)
        logger.info("Inference Started")
        inputs = [indices, positions, segments, *masks]
        """bert_infer_loop(args, session,
                        dataset, inputs, logits, anchors,
                        iteration)"""
        save_results = args.task == "SQUAD" and not (args.synthetic_data or args.generated_data)

        start_times = defaultdict(list)
        end_times = defaultdict(list)
        # Create the stepio once outside of the inference loop:
        static_data = {}
        if args.low_latency_inference and args.task == "SQUAD":
            stepio = create_callback_stepio(static_data, anchors, start_times, end_times,
                                            dataset.batches_per_step, args.replication_factor)
        else:
            stepio = None

        output = []
        logger.info(dataset)
        for data in dataset:
            static_data.update({t: data[t] for t in inputs})
            result = bert_process_infer_data(args, session, static_data, anchors,
                                             logits, iteration,
                                             start_times, end_times, stepio)
            if save_results:
                output.append(result)
            break

        device.detach()
        return output

    return None


@pytest.mark.ipus(2)
@pytest.mark.category1
def test_host_embedding(custom_ops):
    args_string = ["--config",
                   os.path.join(bert_root_dir(), 'configs/mk1/squad_base_128_inference.json'),
                   '--host-embedding=ALL',
                   '--device-connection-type=ondemand',
                   '--generated-data=true'
                   ]
    args = utils.parse_bert_args(args_string)
    args.shuffle = False
    args.host_embedding = "ALL"
    host_embedding_outputs = np.array(run_embedding_layer(args), dtype=float)
    args.host_embedding = "NONE"
    ipu_embedding_outputs = np.array(run_embedding_layer(args), dtype=float)

    if np.allclose(host_embedding_outputs, ipu_embedding_outputs, rtol=0.3):
        logger.info("Passed")
    else:
        logger.info("Failed")
        raise TestFailureError("outputs do not match")
