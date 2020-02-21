# Copyright 2020 Graphcore Ltd.
import os
import ctypes
import numpy as np
from pathlib import Path
import popart
import numpy as np
from collections import defaultdict

from bert import (set_library_seeds,
                  bert_config_from_args,
                  bert_pretrained_initialisers,
                  bert_add_inputs,
                  bert_add_infer_outputs,
                  acquire_device,
                  get_bert_dataset,
                  Iteration,
                  calc_required_ipus,
                  bert_inference_session,
                  create_callback_stepio,
                  enable_realtime_scheduling,
                  bert_process_infer_data,
                  disable_realtime_scheduling)
from bert_model import Bert, BertConfig
from tests.utils import TestFailureError
import logging
import utils

'''
Tests the Embedding and Positional Dict lookup on the host against the same on the IPU.
Executes just the Embedding and Positional parts of the Embedding layer and compares the result.
'''


logger = logging.getLogger('BERT')


def bert_embedding_only_graph(model, indices, positions):
    with model.builder.nameScope("Embedding"):
        with model.embedding_scope:
            x = model.embedding_onnx(indices, model.config.vocab_length, "Embedding_Dict")

        with model.embedding_split_scope:
            x_pos = model.embedding_onnx(positions,
                                         model.config.max_positional_length,
                                         "Positional_Dict",
                                         init_fn=model.config.positional_embedding_init_fn)

            x = model.builder.aiOnnx.add([x, x_pos])
            outputs = [x]
            return tuple(outputs)
    return None


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
    logits = bert_embedding_only_graph(model, indices, positions)

    if args.inference:
        outputs = bert_add_infer_outputs(model, logits)
        losses = []
        writer = None
        embedding_dict, positional_dict = model.get_model_embeddings()

        dataset = get_bert_dataset(model, args, [indices, positions, segments, masks, labels], embedding_dict, positional_dict)

        data_flow = popart.DataFlow(dataset.batches_per_step, outputs)

        iteration = Iteration(
            args,
            batches_per_step=dataset.batches_per_step,
            steps_per_epoch=len(dataset),
            writer=writer,
            recording_steps=args.aggregate_metrics_over_steps)

        request_ipus, required_ipus = calc_required_ipus(args, model)

        device = acquire_device(args, request_ipus)

        session, anchors = bert_inference_session(model, args, data_flow, losses, device)
        logger.info("Inference Started")
        inputs = [indices, positions, segments, *masks]
        """bert_infer_loop(args, session,
                        dataset, inputs, logits, anchors,
                        iteration)"""
        save_results = args.task == "SQUAD" and not args.synthetic_data

        repeat_count = 1

        start_times = defaultdict(list)
        end_times = defaultdict(list)
        # Create the stepio once outside of the inference loop:
        static_data = {}
        if args.low_latency_inference and args.task == "SQUAD":
            stepio = create_callback_stepio(static_data, anchors, start_times, end_times,
                                            dataset.batches_per_step)
        else:
            stepio = None

        enable_realtime_scheduling(args)

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

        disable_realtime_scheduling(args)

        device.detach()
        return output

    return None


def test_host_embedding():
    args_string = ["--config",
                   'configs/squad_base_inference.json',
                   '--host-embedding=true',
                   '--synthetic-data=true'
                   ]
    args = utils.parse_bert_args(args_string)
    args.shuffle = False
    args.host_embedding = True
    host_embedding_outputs = np.array(run_embedding_layer(args), dtype=float)
    args.host_embedding = False
    ipu_embedding_outputs = np.array(run_embedding_layer(args), dtype=float)

    if np.allclose(host_embedding_outputs, ipu_embedding_outputs, rtol=0.3):
        logger.info("Passed")
    else:
        logger.info("Failed")
        raise TestFailureError("outputs do not match")
