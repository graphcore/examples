# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Train the Temporal Graph Network (https://arxiv.org/abs/2006.10637) on IPU."""

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import sklearn.metrics
import tensorflow.compat.v1 as tf
import tqdm

import dataloader
import model
import utils


def stats(outputs: Iterable[Dict[str, np.ndarray]],
          start_time: float, validation: bool, first_epoch: bool) -> Dict[str, float]:
    """
    Accumulate statistics from train/val/test batches.

    Notes:
        Compile time:
        During the first training epoch, we can measure the compile time by
        measuring the latency of the first batch. This is due to how TF
        implicitly leaves compilation of the graph until just before it's
        required, which in this case is the execution of the first batch.
        The compute time for the data itself is orders of magnitude less than
        the time taken to compile, so the measurement is a very good
        approximation.
    """
    total = 0
    loss = 0.0
    metrics_total = 0
    average_precision = 0.0
    roc_auc = 0.0

    for n_batch, output in enumerate(outputs):
        count = int(output["count"])
        total += count
        loss += count * float(output["loss"])
        if "probs" in output:
            labels = np.tile([[1], [0]],
                             (1, output["probs"].shape[1])).flatten()
            probs = output["probs"].flatten()
            metrics_total += count
            average_precision += count * sklearn.metrics.average_precision_score(
                labels, probs)
            roc_auc += count * sklearn.metrics.roc_auc_score(labels, probs)

        # Measuring compile time
        if first_epoch and n_batch == 0:
            compile_time = time.time() - start_time

    result = dict(loss=loss / total)
    if metrics_total:
        result.update(
            average_precision=average_precision / metrics_total,
            roc_auc=roc_auc / metrics_total,
        )
    result.update(count=total, duration=time.time() - start_time)

    if not validation:
        result.update(throughput=f"{total / result['duration']} samples/sec")

    if first_epoch:
        result.update(compile_time=compile_time)

    return result


def run_training(
    data: Path,
    n_epoch: int,
    n_batch: Optional[int],
    batch_size: int,
    nodes_size: int,
    edges_size: int,
    validate_every: Optional[int],
    cache_dataset: bool,
    target: utils.Target,
    dtype: np.dtype,
    save: Optional[Path],
    load: Optional[Path],
) -> Iterator[Dict[str, Any]]:
    """Execute a complete training run, yielding summaries for each epoch.

    Arguments:

      data -- path to store JODIE-Wikipedia, automatically downloaded if not present

      n_epoch -- total number of training epochs to run

      n_batch -- limit on batches per epoch (note: since the data cannot be shuffled,
                 this effectively reduces the size of the training dataset)

      validate_every -- number of training of epochs to run before running, or None
                        to disable validation

      cache_dataset -- read the dataset once (note: this reduces the diversity of
                       negative samples over training, increasing validation loss)

      target -- device type

      dtype -- either np.float32 or np.float16, to set the minimum precision used
               for model and optimisation

      save -- path to save final model state, in NPZ format

      load -- path to load initial model state, in NPZ format, as saved using "save"

    Yields:

      {"n_epoch": <int>,              -- number of training epochs completed
       "part": "train"|"val"|"test",  -- dataset partition
       "count": <int>,                -- total number of examples
       "loss": <float>,               -- mean contrastive loss
       "average_precision": <float>,  -- ("val"/"test" only), average precision
       "roc_auc": <float>,            -- ("val"/"test" only), area under the ROC curve
       "duration": <float>            -- time (seconds) to execute the epoch
       }
    """

    loader = dataloader.Data(data, dtype=dtype, batch_size=batch_size,
                             nodes_size=nodes_size, edges_size=edges_size)
    settings = dict(
        n_nodes=loader.data.num_nodes,
        memory_size=100,
        time_embedding_size=100,
        dropout=0.1,
        learning_rate=1e-4,
        target=target,
    )
    make_runner = (utils.IpuLoopRunner if target is utils.Target.IPU else
                   functools.partial(utils.IteratorRunner, target=target))

    if target is utils.Target.IPU:
        config = utils.ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        config.floating_point_behaviour.esr = (
            utils.ipu.config.StochasticRoundingBehaviour.ON)
        config.floating_point_behaviour.oflo = True
        config.floating_point_behaviour.inv = True
        config.floating_point_behaviour.div0 = True
        config.configure_ipu_system()

    # Build all the ops we will need to run
    with tf.Graph().as_default() as graph:
        runners = {}
        for part in ["train", "val"]:

            def modelfn(**args: tf.Tensor) -> Dict[str, tf.Tensor]:
                with tf.variable_scope("model", reuse=part != "train"):
                    return model.tgn(is_training=part == "train",
                                     **settings,
                                     **args)

            part_batches = n_batch or loader.n_batches(part)
            dataset = loader.dataset(part).take(part_batches)
            if cache_dataset:
                dataset = dataset.cache()
            runners[part] = make_runner(  # type:ignore[operator]
                fn=modelfn,
                dataset=dataset,
                n_batch=part_batches)

        if target is utils.Target.IPU:
            utils.ipu.utils.move_variable_initialization_to_cpu()

        variables = {
            v.name: v
            for v in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        }
        if load:
            loaded = np.load(load)
            initialize_variables = tf.group(
                [tf.assign(v, loaded[k]) for k, v in variables.items()])
        else:
            initialize_variables = tf.group(
                [v.initializer for v in variables.values()])
        reset_memory = tf.group([
            v.initializer
            for v in graph.get_collection(model.TGN_MEMORY_VARIABLES_KEY)
        ])

    # Run the program
    with tf.Session(graph=graph) as session:

        def epoch(n_epoch: int, part: str) -> Dict[str, Any]:
            t0 = time.time()
            outputs = tqdm.tqdm(runners[part](session))

            # Include compilation stats if it's the first training epoch
            epoch_stats = stats(
                outputs,
                t0,
                part == "val",
                n_epoch == 1 and part == "train"
            )

            return dict(n_epoch=n_epoch, part=part, **epoch_stats)

        def should_validate(n_epoch: int) -> bool:
            if validate_every is None:
                return False
            return n_epoch == 1 or (n_epoch % validate_every == 0)

        session.run(initialize_variables)
        for n in range(1, n_epoch + 1):
            session.run(reset_memory)
            yield epoch(n, "train")
            if should_validate(n):
                yield epoch(n, "val")
        if save:
            np.savez(save, **session.run(variables))


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",
                        default="data/JODIE",
                        type=Path,
                        help="folder to find/put the data")
    parser.add_argument(
        "-t",
        "--target",
        choices=[target.name for target in utils.Target],
        default=utils.Target.IPU.name
        if utils.IPU_AVAILABLE else utils.Target.DEFAULT.name,
        help="select device to run on (default: IPU if available)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=np.dtype,
        help="main floating point datatype (default: float16 for IPU)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("train", "profile", "benchmark"),
        default="train",
        help="'train' - a full training run with regular validation"
        "; 'profile' - run a few steps of training, no validation,"
        " for use with the graph analyser"
        "; 'benchmark' - enable dataset caching (slight metrics degradation)"
        " and infrequent validation",
    )
    parser.add_argument(
        "--n-epoch",
        default=None,
        type=int,
        help=(
            "Number of epochs to train for. If provided, overrides number "
            "specified by choice of --mode."
        )
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help=(
            "Batch size for training and inference. If provided, overrides number "
            "specified by choice of --mode."
        )
    )
    parser.add_argument(
        "--nodes-size",
        default=None,
        type=int,
        help=(
            "Padding for nodes. If provided, overrides number specified by choice of"
            "--mode."
        )
    )
    parser.add_argument(
        "--edges-size",
        default=None,
        type=int,
        help=(
            "Padding for edges. If provided, overrides number specified by choice of"
            "--mode."
        )
    )
    parser.add_argument(
        "--validate-every",
        default=None,
        type=int,
        help=(
            "Number of epochs to validate after. If provided, overrides number "
            "specified by choice of --mode."
        )
    )
    parser.add_argument("--load", type=Path, help="path to load model")
    parser.add_argument("--save", type=Path, help="path to save model")

    args = vars(parser.parse_args())
    args["target"] = utils.Target[args["target"]]
    if args["dtype"] is None:
        args["dtype"] = np.float16 if args[
            "target"] is utils.Target.IPU else np.float32

    all_configs = dict(
        train=dict(
            n_epoch=25,
            n_batch=None,
            batch_size=200,
            nodes_size=1200,
            edges_size=4000,
            validate_every=1,
            cache_dataset=False,
        ),
        profile=dict(
            n_epoch=1,
            n_batch=5,
            batch_size=200,
            nodes_size=1200,
            edges_size=4000,
            validate_every=None,
            cache_dataset=False,
        ),
        benchmark=dict(
            n_epoch=25,
            n_batch=None,
            batch_size=200,
            nodes_size=1200,
            edges_size=4000,
            validate_every=25,
            cache_dataset=True,
        ),
    )

    # Merge configs into args, except where values are specified in the cmd
    for key, value in all_configs[args.pop("mode")].items():
        if args.get(key) is None:
            args[key] = value

    # Run
    for log in run_training(**args):
        # Printing epoch results individually
        for k, v in log.items():
            print(f"{k}: {v}, ", end="")
        print("\n", flush=True)


if __name__ == "__main__":
    tf.disable_v2_behavior()
    _main()
