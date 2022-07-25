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
import logging
import argparse
import json
import multiprocessing
import shutil
import tempfile
import re
import sys
from io import StringIO
from functools import partial, reduce
from pathlib import Path
from collections import defaultdict

import popart
import onnx

# Add the bert root to the PYTHONPATH
bert_root_path = str(Path(__file__).parent.parent)
sys.path.append(bert_root_path)

from bert import (  # noqa: E402
    setup_logger,
    set_library_seeds,
    bert_config_from_args,
    bert_pretrained_initialisers,
    bert_add_inputs,
    bert_add_logit_outputs,
    get_bert_dataset,
    Iteration,
    bert_required_ipus,
    bert_inference_session,
    bert_infer_loop
)
from bert_model import Bert  # noqa: E402
import utils  # noqa: E402
from utils.device import acquire_device  # noqa: E402

logger = logging.getLogger(__file__)


class NoResultRecordedException(Exception):
    pass


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


def result_into_recursive_path(result_dict, checkpoint_path, root_path, result):
    """
     Place a result in the recursive defaultdict according to its path relative to the root.
     Note: Modifies the result_dict dict in place.
    """
    # Since we're using a recursive glob to find the checkpoints, the results path should adhere to the same path layout
    relpath = Path(os.path.relpath(checkpoint_path, root_path))

    # We'll keep a pointer to the current path leaf and navigate through.
    # The last element will get set to the result
    ptr = result_dict
    for part in relpath.parts[:-1]:
        ptr = ptr[part]
    ptr[relpath.parts[-1]] = result


def run_inference_extract_result(args, session, dataset, inputs, logits, anchors, iteration):
    # Record the result entry from the Bert log without modifying the bert runtime code.
    dataset_logger = logging.getLogger("bert_data.squad_dataset")
    log_capture_string = StringIO()
    log_capture_handler = logging.StreamHandler(log_capture_string)
    log_capture_handler.setLevel(logging.INFO)
    dataset_logger.addHandler(log_capture_handler)
    bert_infer_loop(args, session, dataset, inputs, logits, anchors, None, None, None, iteration)
    log_contents = log_capture_string.getvalue()

    if log_contents is None or not log_contents:
        raise NoResultRecordedException(
            "Resulting log was empty - is logging enabled?")

    result = None
    for line in log_contents.splitlines():
        matches = re.search(r"F1 Score: ([\d\.]+) \| Exact Match: ([\d\.]+)", line)
        if matches is not None:
            f1 = float(matches[1])
            exact_match = float(matches[2])
            result = {"F1": f1, "exact_match": exact_match}

    if result is None:
        raise NoResultRecordedException(
            "Log didn't include F1 - have we recorded the correct logs?")

    return result


def pooled_validation_run(bert_args,
                          config,
                          initializers,
                          checkpoint_paths,
                          num_processes=1,
                          available_ipus=16):
    logger.info("Building Model")
    model = Bert(config,
                 initializers=initializers)

    indices, positions, segments, masks, labels = bert_add_inputs(
        bert_args, model)
    logits = bert_logits_graph(model, indices, positions, segments, masks, bert_args.pipeline)
    inputs = [indices, positions, segments, *masks]
    outputs = bert_add_logit_outputs(model, logits)

    with tempfile.TemporaryDirectory() as temp_results_path:
        # Inject the checkpoint-specific squad results directory into the dataset args otherwise
        # they overwrite each other when multithreaded
        bert_args.squad_results_dir = temp_results_path

        dataset = get_bert_dataset(
            model, bert_args, [indices, positions, segments, masks, labels])
        logger.info(f"Dataset length: {len(dataset)}")

        data_flow = popart.DataFlow(dataset.device_iterations, outputs)

        iteration = Iteration(
            bert_args,
            device_iterations=dataset.device_iterations,
            steps_per_epoch=len(dataset),
            writer=None,
            recording_steps=bert_args.aggregate_metrics_over_steps)

        request_ipus, _ = bert_required_ipus(bert_args, model)

        if request_ipus * num_processes > available_ipus:
            raise ValueError(
                "Cannot run with requested number of processes - too many IPUs required")

        device = acquire_device(bert_args, request_ipus)

        session, anchors = bert_inference_session(
            model, bert_args, data_flow, device)

        model_results = recursive_defaultdict()
        for path in checkpoint_paths:
            session.resetHostWeights(str(path.absolute()))
            session.weightsFromHost()

            logger.info(f"Inference started for checkpoint: {path.absolute()}")
            result = run_inference_extract_result(bert_args,
                                                  session,
                                                  dataset,
                                                  inputs,
                                                  logits,
                                                  anchors,
                                                  iteration)

            result_into_recursive_path(model_results, path, bert_args.checkpoint_dir, result)

        device.detach()
    return model_results


def merge_pooled_results(results_pooled):

    def deep_merge_dicts(dict_a, dict_b, path=None):
        if path is None:
            path = []

        for key in dict_b:
            if key in dict_a:
                if isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
                    deep_merge_dicts(dict_a[key], dict_b[key], path + [str(key)])
                else:
                    pass
            else:
                dict_a[key] = dict_b[key]
        return dict_a

    return reduce(deep_merge_dicts, results_pooled)


def perform_validations(num_processes, checkpoint_paths, args, config, initializers, available_ipus):
    if num_processes == 1:
        return pooled_validation_run(args,
                                     config,
                                     initializers,
                                     checkpoint_paths,
                                     num_processes=1,
                                     available_ipus=available_ipus)

    num_processes = min(num_processes, len(checkpoint_paths))
    pool = multiprocessing.Pool(num_processes)
    func = partial(pooled_validation_run,
                   args,
                   config,
                   initializers,
                   num_processes=num_processes,
                   available_ipus=available_ipus)

    # This will find the optimal mapping of checkpoints->processes to avoid excess
    # graph compilations
    checkpoint_paths_pooled = (checkpoint_paths[i*len(checkpoint_paths) // num_processes:
                                                (i+1)*len(checkpoint_paths) // num_processes]
                               for i in range(num_processes))

    results_pooled = pool.map(func, checkpoint_paths_pooled)
    pool.close()
    pool.join()

    return merge_pooled_results(results_pooled)


def validate_checkpoints(self_args, args):
    set_library_seeds(args.seed)
    config = bert_config_from_args(args)

    checkpoint_paths = [p for p in Path(
        args.checkpoint_dir).rglob(self_args.model_search_string)]

    if len(checkpoint_paths) < 1:
        raise FileNotFoundError(
            f"Did not recursively find any checkpoints at path: {args.checkpoint_dir}")

    # Load an initial model to configure the IO tensors and the session
    args.onnx_checkpoint = checkpoint_paths[0]
    initializers = bert_pretrained_initialisers(config, args)

    results = perform_validations(self_args.num_processes,
                                  checkpoint_paths,
                                  args,
                                  config,
                                  initializers,
                                  self_args.available_ipus)

    with open(os.path.join(self_args.checkpoint_dir, "validation_result.json"), 'w') as fh:
        json.dump(results, fh, indent=4)

    return results


def find_checkpoint_config(checkpoint_dir):
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not supplied and could not find one "
                                f"with the checkpoints: {config_path}.")
    return config_path


def parse_args(arg_list=None):
    pparser = argparse.ArgumentParser()
    pparser.add_argument("--num-processes", type=int, default=1)
    pparser.add_argument("--checkpoint-dir", type=str, required=True)
    pparser.add_argument("--model-search-string", type=str, default="*.onnx")
    pparser.add_argument("--no-logger-setup", action="store_true",
                         help="Don't set up the logger here - used if this script is being called by another "
                         "and the parent has already setup the logger.")
    pparser.add_argument("--available-ipus", type=int, default=16,
                         help="The maximum number of available IPUs to allow for the validation processes.")
    return pparser.parse_known_args(arg_list)


def main(arg_list=None):
    run_args, remaining_args = parse_args(arg_list)
    remaining_args += ["--checkpoint-dir", run_args.checkpoint_dir]

    # If no config is supplied, try to load the config that should have been saved with the ckpts.
    if "--config" not in remaining_args:
        config_path = find_checkpoint_config(run_args.checkpoint_dir)
        remaining_args += ["--config", config_path]

    bert_args = utils.parse_bert_args(remaining_args)
    if not run_args.no_logger_setup:
        setup_logger(logging.getLevelName('INFO'))

    # Force variable weights in inference mode - otherwise we can't override the model weights for
    # validating each new checkpoint.
    bert_args.variable_weights_inference = True
    # Required to allow squeezed models to fit.
    bert_args.max_copy_merge_size = 32000

    logger.info("Program Start")

    # `parse_bert_args` will suffix the user-supplied checkpoint path with the current date/time.
    # To avoid modifying core Bert code, we'll just remove the suffix (we don't need the created
    # config).
    shutil.rmtree(bert_args.checkpoint_dir)
    bert_args.checkpoint_dir = os.path.dirname(bert_args.checkpoint_dir)

    logger.info(
        f"Validating over checkpoints in directory {bert_args.checkpoint_dir}")
    return validate_checkpoints(run_args, utils.get_validation_args(bert_args))


if __name__ == "__main__":
    main()
