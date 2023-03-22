# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorflow as tf
from ogb.graphproppred import Evaluator
from ogb.lsc import PCQM4Mv2Evaluator

import wandb
import xpu
from argparser import parse_args
from data_utils.input_spec import create_inputs_from_features
from data_utils.load_dataset import load_raw_dataset
from data_utils.preprocess_dataset import preprocess_dataset
from model.utils import (
    check_loaded_weights,
    create_model,
    get_loss_functions,
    get_metrics,
    get_tf_dataset,
    load_checkpoint_into_model,
)
from plotting import Plotting
from utils import convert_loss_and_metric_reductions_to_fp32, get_optimizer, Timer, str_dtype_to_tf_dtype


def get_evaluator(dataset_name):
    if dataset_name in (
        "pcqm4mv2",
        "generated",
        "pcqm4mv2_28features",
        "pcqm4mv2_conformers",
        "pcqm4mv2_conformers_28features",
    ):
        return PCQM4Mv2Evaluator(), "mae"
    else:
        return Evaluator(name=dataset_name), "rocauc"


def format_predictions(dataset_name, y_true=None, y_pred=None):
    return dict(
        y_true=format_out_tensor(dataset_name, y_true) if y_true is not None else None,
        y_pred=format_out_tensor(dataset_name, y_pred) if y_pred is not None else None,
    )


def format_out_tensor(dataset_name, out_tensor):
    if dataset_name in (
        "pcqm4mv2",
        "generated",
        "pcqm4mv2_28features",
        "pcqm4mv2_conformers",
        "pcqm4mv2_conformers_28features",
    ):
        return np.ravel(out_tensor)
    else:
        return out_tensor[:, None]


def save_nparray_as_csv_to_wandb(base_dir, name, array):
    f_name = f"{base_dir}/{name}.csv"
    np.savetxt(f_name, array, delimiter=",")
    wandb.save(f_name, policy="now", base_path=base_dir)


def run_inference(
    case,
    graph_data,
    checkpoint_paths,
    cfg,
    optimizer_options={},
    losses=[],
    loss_weights=[],
    metrics=[],
    stochastic_rounding=False,
    eval_mode=None,
    ensemble=False,
    tmpdir=".",
):
    if case == "valid":
        fold = "valid"
        full_name = "validation"
    elif case == "clean_train":
        fold = "train"
        full_name = "clean training"
    elif case == "test-dev":
        fold = "test-dev"
        full_name = "test-dev"
    elif case == "test-challenge":
        fold = "test-challenge"
        full_name = "test-challenge"
    else:
        raise ValueError(f"inference 'case' {case} not recognised.")
    if not eval_mode:
        eval_mode = cfg.model.eval_mode

    if cfg.model.valid_batch_size is not None:
        cfg.model.micro_batch_size = cfg.model.valid_batch_size

    input_spec = create_inputs_from_features(dataset=graph_data, cfg=cfg, fold=fold)
    plotter = Plotting()
    logging.info(f"Running evaluation on {full_name} set...")
    logging.info(f"Checkpoints: {checkpoint_paths}")
    # Use default 1 replica for validation
    strategy = xpu.configure_and_get_strategy(
        num_replicas=1, num_ipus_per_replica=1, stochastic_rounding=stochastic_rounding, cfg=cfg
    )
    with strategy.scope():
        evaluator, result_name = get_evaluator(dataset_name=cfg.dataset.dataset_name)

        batch_generator, ground_truth_and_masks = get_tf_dataset(
            preprocessed_dataset=graph_data,
            split_name=fold,
            shuffle=False,
            options=cfg,
            pad_remainder=True,
            input_spec=input_spec,
            ensemble=ensemble,
        )

        ds = batch_generator.get_tf_dataset()
        ground_truth, include_mask = ground_truth_and_masks
        ground_truth = ground_truth[include_mask]
        model = create_model(batch_generator, graph_data, cfg, input_spec=input_spec)
        model.compile(
            optimizer=get_optimizer(**optimizer_options),
            loss=losses,
            loss_weights=loss_weights,
            weighted_metrics=metrics,
            steps_per_execution=batch_generator.batches_per_epoch,
        )

        if cfg.model.dtype == "float16":
            # the loss reduction is set by backend.floatx by default
            # must be forced to reduce in float32 to avoid overflow
            convert_loss_and_metric_reductions_to_fp32(model)

        all_preds = OrderedDict()

        if checkpoint_paths == None:
            checkpoint_paths = {0: None}
        for epoch, checkpoint_path in checkpoint_paths.items():
            with Timer(f"{checkpoint_path}"):
                if checkpoint_path != None:
                    model.load_weights(checkpoint_path).expect_partial()
                if checkpoint_path != None:
                    model.load_weights(checkpoint_path).expect_partial()

                if type(epoch) is not int:
                    # non numeric epoch values dont play nicely with wandb
                    log = {"epoch": cfg.model.epochs}
                else:
                    log = {"epoch": epoch}
                if eval_mode in ("keras", "both"):
                    start_time = time.time()
                    results = model.evaluate(ds, steps=batch_generator.batches_per_epoch)
                    end_time = time.time()
                    duration = end_time - start_time
                    throughput = batch_generator.n_graphs_per_epoch / duration
                    log.update({"throughput": f"{throughput} graphs / s"})

                    loss_names = model.compiled_loss._output_names
                    if len(loss_names) > 1:
                        loss_names = ["Total"] + loss_names
                    loss_names = [n + "_Loss" for n in loss_names]
                    metric_names = [m._name for m in model.compiled_metrics.metrics]

                    n_losses = len(results) - len(metric_names)

                    assert n_losses >= 1, f"loss_names: {loss_names}, metric_names:{metric_names}, results: {results}"

                    loss_names = loss_names[:n_losses]
                    if type(epoch) is int or epoch == "FINAL":
                        log.update({f"keras_{case}_{name}": r for r, name in zip(results, loss_names + metric_names)})
                    else:
                        log.update(
                            {f"keras_{case}_{name}_{epoch}": r for r, name in zip(results, loss_names + metric_names)}
                        )

                if eval_mode in ("ogb", "both"):
                    start_time = time.time()
                    prediction = model.predict(ds, steps=batch_generator.batches_per_epoch)
                    end_time = time.time()
                    duration = end_time - start_time
                    throughput = batch_generator.n_graphs_per_epoch / duration
                    log.update({"throughput": f"{throughput} graphs / s"})
                    if isinstance(prediction, list) and len(prediction) > 1:
                        prediction = prediction[0]
                    prediction = prediction.squeeze()

                    if cfg.dataset.normalize_labels:
                        prediction = graph_data.denormalize(prediction)

                    if len(include_mask) > len(prediction):
                        include_mask = include_mask[: len(prediction)]
                        ground_truth = ground_truth[: len(prediction)]

                    # include_mask may be shorter than the predictions —
                    #   that is fine (it will just be padding after that point)
                    prediction = prediction[: len(include_mask)][include_mask.squeeze() == 1]

                    # Always in tmp - these are therefore only stored on wandb
                    if cfg.wandb:
                        save_nparray_as_csv_to_wandb(
                            tmpdir, f"{cfg.dataset.dataset_name}-{fold}-predictions-{wandb.run.id}", prediction
                        )
                        save_nparray_as_csv_to_wandb(
                            tmpdir, f"{cfg.dataset.dataset_name}-{fold}-ground-truth-{wandb.run.id}", ground_truth
                        )

                    formatted_predictions = format_predictions(
                        dataset_name=cfg.dataset.dataset_name, y_true=ground_truth, y_pred=prediction
                    )
                    # we will use the official AUC evaluator from the OGB repo, not the keras one
                    result = evaluator.eval(formatted_predictions)
                    if case in ["test-dev", "test-challenge"]:
                        # save predictions for test-dev and test-challenge
                        evaluator.save_test_submission(
                            input_dict=formatted_predictions, dir_path=cfg.submission_results_dir, mode=case
                        )

                    if type(epoch) is int or epoch == "FINAL":
                        log.update({f"{case}_{result_name}": result[result_name]})
                    else:
                        log.update({f"{case}_{result_name}_{epoch}": result[result_name]})

                    all_preds[epoch] = formatted_predictions

                logging.info(log)

                if cfg.wandb:
                    wandb.log(log)

        return all_preds


if __name__ == "__main__":
    # Config Options
    logging.basicConfig(level=logging.INFO)
    cfg = parse_args()

    if cfg.wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, config=cfg.as_dict())

    tf.keras.mixed_precision.set_global_policy(cfg.model.dtype)

    logging.info(f"Dataset: {cfg.dataset.dataset_name}")
    graph_data = load_raw_dataset(cfg.dataset.dataset_name, Path(cfg.dataset.cache_path), cfg)

    graph_data = preprocess_dataset(dataset=graph_data, options=cfg)

    optimizer_options = dict(
        name=cfg.model.opt.lower(),
        learning_rate=cfg.model.lr,
        dtype=str_dtype_to_tf_dtype(cfg.model.dtype),
        m_dtype=str_dtype_to_tf_dtype(cfg.model.adam_m_dtype),
        v_dtype=str_dtype_to_tf_dtype(cfg.model.adam_v_dtype),
        clip_value=cfg.model.grad_clip_value,
        loss_scale=cfg.model.loss_scaling,
        gradient_accumulation_factor=cfg.ipu_opts.gradient_accumulation_factor,
        replicas=cfg.ipu_opts.replicas,
    )
    losses, loss_weights = get_loss_functions(graph_data, cfg)
    metrics = get_metrics(graph_data.denormalize, cfg)

    if cfg.checkpoint_path:
        checkpoint_paths = {-1: cfg.checkpoint_path}
        # Filter the checkpoints for only checkpoints that exist
        checkpoint_paths = {e: p for e, p in checkpoint_paths.items() if os.path.exists(p + ".index")}
    else:
        checkpoint_paths = None

    run_inference(
        cfg.inference_fold,
        graph_data,
        checkpoint_paths,
        cfg,
        optimizer_options=optimizer_options,
        losses=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
