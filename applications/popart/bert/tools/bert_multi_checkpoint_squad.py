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

import os
import sys
import shutil
import logging
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import popart
import onnx

# Add the bert root to the PYTHONPATH
bert_root_path = str(Path(__file__).parent.parent)
sys.path.append(bert_root_path)

from bert import (
    setup_logger,
    set_library_seeds,
    bert_config_from_args,
    bert_pretrained_initialisers,
    bert_add_inputs,
    bert_logits_graph,
    get_bert_dataset,
    Iteration,
    calc_required_ipus,
    acquire_device,
    bert_training_session,
    bert_process_data,
    bert_infer_graph,
    bert_loss_graph,
    bert_add_validation_outputs,
    bert_writer
)
from bert_model import Bert
from bert_optimizer import ScheduledOptimizerFactory
import utils

logger = logging.getLogger(__file__)


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


def run_fine_tuning_store_ckpt(bert_args,
                               model,
                               ckpt_name,
                               session,
                               dataset,
                               predictions,
                               losses,
                               labels,
                               anchors):

    writer = bert_writer(bert_args)
    iteration = Iteration(bert_args,
                          batches_per_step=dataset.batches_per_step,
                          steps_per_epoch=len(dataset),
                          writer=writer,
                          recording_steps=bert_args.aggregate_metrics_over_steps)
    optimizer_factory = ScheduledOptimizerFactory(bert_args,
                                                  iteration,
                                                  "SGD",
                                                  model.tensors)

    for iteration.epoch in range(iteration.start_epoch, bert_args.epochs):
        for data in dataset:
            bert_process_data(bert_args, session, labels, data, anchors,
                              losses, predictions, iteration, optimizer_factory)

    model_fn = os.path.join(bert_args.checkpoint_dir,
                            "squad_output", f"squad_final_{ckpt_name}.onnx")
    session.modelToHost(model_fn)


def training_run(bert_args, config, initializers, checkpoint_paths):
    logger.info("Building Model")
    model = Bert(config,
                 builder=popart.Builder(
                     opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}),
                 initializers=initializers,
                 execution_mode=bert_args.execution_mode)

    indices, positions, segments, masks, labels = bert_add_inputs(bert_args, model)
    logits = bert_logits_graph(model, indices, positions, segments, masks, bert_args.execution_mode)

    predictions, probs = bert_infer_graph(model, logits)
    losses = bert_loss_graph(model, probs, labels)
    outputs = bert_add_validation_outputs(model, predictions, losses)

    embedding_dict, positional_dict = model.get_model_embeddings()
    dataset = get_bert_dataset(model,
                               bert_args,
                               [indices, positions, segments, masks, labels],
                               embedding_dict,
                               positional_dict)

    data_flow = popart.DataFlow(dataset.batches_per_step, outputs)

    request_ipus, _ = calc_required_ipus(bert_args, model)
    device = acquire_device(bert_args, request_ipus)

    logger.info(f"Dataset length: {len(dataset)}")

    writer = bert_writer(bert_args)
    iteration = Iteration(
        bert_args,
        batches_per_step=dataset.batches_per_step,
        steps_per_epoch=len(dataset),
        writer=writer,
        recording_steps=bert_args.aggregate_metrics_over_steps)
    optimizer_factory = ScheduledOptimizerFactory(bert_args,
                                                  iteration,
                                                  "SGD",
                                                  model.tensors)
    session, anchors = bert_training_session(
        model, bert_args, data_flow, losses, device, optimizer_factory)

    for path in checkpoint_paths:
        ckpt_name = os.path.splitext(os.path.basename(path))[0]
        session.resetHostWeights(os.path.abspath(path))
        session.weightsFromHost()

        logger.info(f"Fine-tuning started for checkpoint: {path}")

        run_fine_tuning_store_ckpt(bert_args,
                                   model,
                                   ckpt_name,
                                   session,
                                   dataset,
                                   predictions,
                                   losses,
                                   labels,
                                   anchors)

    device.detach()


def finetune_checkpoints(self_args, args):
    set_library_seeds(args.seed)
    config = bert_config_from_args(args)

    checkpoint_paths = glob.glob(os.path.join(
        self_args.checkpoint_dir, self_args.model_search_string))
    checkpoint_paths.sort()

    os.makedirs(os.path.join(self_args.checkpoint_dir,
                             "squad_output"), exist_ok=True)

    if len(checkpoint_paths) < 1:
        raise FileNotFoundError(
            f"Did not find any checkpoints at path: {args.checkpoint_dir}")

    # Load an initial model to configure the IO tensors and the session
    args.onnx_checkpoint = checkpoint_paths[0]
    initializers = bert_pretrained_initialisers(config, args)

    training_run(args,
                 config,
                 initializers,
                 checkpoint_paths)

    logger.info("Fine-Tuning Complete")


def parse_args(arg_list=None):
    pparser = argparse.ArgumentParser()
    pparser.add_argument("--checkpoint-dir", type=str, required=True)
    pparser.add_argument("--model-search-string", type=str, default="*.onnx")
    pparser.add_argument("--no-logger-setup", action="store_true",
                         help="Don't set up the logger here - used if this script is being called by another "
                         "and the parent has already setup the logger.")
    return pparser.parse_known_args(arg_list)


def main(arg_list=None):
    run_args, remaining_args = parse_args(arg_list)
    remaining_args += ["--checkpoint-dir", run_args.checkpoint_dir]

    bert_args = utils.parse_bert_args(remaining_args)
    print(bert_args)
    if not run_args.no_logger_setup:
        setup_logger(logging.getLevelName('INFO'))

    logger.info("Program Start")

    # `parse_bert_args` will suffix the user-supplied checkpoint path with the current date/time.
    # To avoid modifying core Bert code, we'll just remove the suffix (we don't need the created
    # config).
    shutil.rmtree(bert_args.checkpoint_dir)
    bert_args.checkpoint_dir = os.path.dirname(bert_args.checkpoint_dir)

    logger.info(
        f"Fine-Tuning over checkpoints in directory {bert_args.checkpoint_dir}")
    finetune_checkpoints(run_args, bert_args)


if __name__ == "__main__":
    main()
