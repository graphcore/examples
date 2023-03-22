# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os
import math
import argparse
import time
from datetime import datetime
import json
from dataclasses import asdict
import torch
import poptorch

import nbfnet_utils
from nbfnet import NBFNet
import data as nbfnet_data
import hyperparameters


def training(model, dataloader, optim, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    total_count = 0
    for batch in dataloader:
        if device == "ipu":
            loss, count = model(**batch)
            loss, count = loss.mean(), count.sum()  # reduction across replicas
        else:
            optim.zero_grad()
            count, loss = model(**batch)
            loss.backward()
            optim.step()
        total_loss += float(loss) * count
        total_count += count
    return total_loss / total_count


def inference(model, dataloader, metrics, annot=""):
    """Performs inference over one epoch"""
    if annot:
        annot = annot + "_"
    model.eval()
    results = {annot + metric: 0 for metric in metrics}
    total_count = 0
    for batch in dataloader:
        prediction, count, mask, _ = model(**batch)
        if isinstance(count, torch.Tensor):
            count = count.sum()
        prediction = prediction[mask]

        true_score = prediction[:, 0:1]
        rank = torch.sum(true_score <= prediction, dim=-1)
        for metric in metrics:
            if metric == "MR":
                results[annot + metric] += float(torch.sum(rank))
            elif metric == "MRR":
                results[annot + metric] += float(torch.sum(1 / rank))
            else:
                s, k = metric.split("@")
                assert s == "hits"
                results[annot + metric] += float(torch.sum(rank <= int(k)))
        total_count += count

    for key in results.keys():
        results[key] /= total_count
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("--profile", action="store_true", help="create memory and execution profile")
    parser.add_argument("--profile_dir", type=str, default="", help="directory for storing profile")
    parser.add_argument("--device", choices=["ipu", "cpu"], default="ipu")

    args = parser.parse_args()
    config = hyperparameters.config_from_yaml(args.config)
    config.execution.device = args.device

    logger = nbfnet_utils.create_logger()

    dataset = nbfnet_data.build_dataset(**asdict(config.dataset), path="./data")

    dataloader = dict(
        train=nbfnet_data.DataWrapper(
            nbfnet_data.NBFData(
                data=dataset[0],
                batch_size=config.execution.batch_size_train,
                is_training=True,
                num_relations=dataset.num_relations,
                num_negatives=config.execution.num_negative,
                check_negatives=config.execution.check_negatives,
            )
        ),
        valid=nbfnet_data.DataWrapper(
            nbfnet_data.NBFData(
                data=dataset[1],
                batch_size=config.execution.batch_size_test,
                is_training=False,
            )
        ),
        test=nbfnet_data.DataWrapper(
            nbfnet_data.NBFData(
                data=dataset[2],
                batch_size=config.execution.batch_size_test,
                is_training=False,
            )
        ),
    )
    # Increment relation indices to allow padding id 0
    num_relations = dataset.num_relations + 1

    model = NBFNet(**asdict(config.model), num_relations=num_relations)
    assert config.execution.dtype in ("float32", "float16"), f"dtype " f"{config.execution.dtype} not supported"
    if config.execution.dtype == "float16":
        model.half()

    if args.device == "ipu":
        train_opts = poptorch.Options()
        test_opts = poptorch.Options()

        if config.execution.pipeline:
            pipeline = config.execution.pipeline
            logger.info(f"Pipelined execution")
            logger.info(pipeline)
        else:
            pipeline = {
                "preprocessing": 0,
                "prediction": 0,
                **{f"layer{i}": 0 for i in range(len(config.model.hidden_dims))},
            }
        pipeline_plan = [poptorch.Stage(k).ipu(v) for k, v in pipeline.items()]
        pipelined_strategy = poptorch.PipelinedExecution(*pipeline_plan)
        train_opts.setExecutionStrategy(pipelined_strategy)
        train_opts.replicationFactor(config.execution.replicas)
        train_opts.deviceIterations(config.execution.device_iterations)
        train_opts.Training.gradientAccumulation(config.execution.gradient_accumulation)
        train_opts.autoRoundNumIPUs(True)
        test_opts.setExecutionStrategy(pipelined_strategy)
        test_opts.deviceIterations(len(set(pipeline.values())))
        test_opts.autoRoundNumIPUs(True)

        for partition in ["train", "valid", "test"]:
            dataloader[partition] = poptorch.DataLoader(
                options=train_opts if partition == "train" else test_opts,
                dataset=dataloader[partition],
                batch_size=1,
                collate_fn=nbfnet_data.custom_collate,
            )

        optim = poptorch.optim.AdamW(
            model.parameters(),
            lr=config.execution.lr,
            bias_correction=False,
            weight_decay=0.0,
            eps=1e-8,
            betas=(0.9, 0.999),
            loss_scaling=config.execution.loss_scale,
            second_order_momentum_accum_type=torch.float32,
        )

        model_train = poptorch.trainingModel(model, options=train_opts, optimizer=optim)
        model_valid = poptorch.inferenceModel(model, options=test_opts)
        model_test = poptorch.inferenceModel(model, options=test_opts)

    else:
        optim = torch.optim.AdamW(
            model.parameters(), lr=config.execution.lr, weight_decay=0.0, eps=1e-8, betas=(0.9, 0.999)
        )

        model_train, model_valid, model_test = model, model, model

    if args.profile:
        config.execution.num_epochs = 1
        profile_dir = args.profile_dir or os.path.join(
            ".", "profiles", f"{config.dataset.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        hyperparameters.config_to_yaml(
            config, os.path.join(profile_dir, f"config_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.yaml")
        )
        eng_opts = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))
        eng_opts.setdefault("autoReport.all", "true")
        eng_opts.setdefault("debug.allowOutOfMemory", "true")
        eng_opts.setdefault("autoReport.directory", profile_dir)
        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(eng_opts)
        logger.info(f"Writing profile to {profile_dir}")

    logstr = (
        f"Start training on {config.dataset.name}. "
        f"{dataset[0].num_nodes} entities, "
        f"{len(dataset[0].edge_type)} edges, "
        f"{num_relations} relation types, "
        f"{len(dataset[0].target_edge_type)} training triples"
    )
    if config.execution.do_valid:
        logstr += f", {len(dataset[1].target_edge_type)} validation triples"
    if config.execution.do_test:
        logstr += f", {len(dataset[2].target_edge_type)} test triples"
    logger.info("*" * len(logstr))
    logger.info(logstr)
    logger.info("*" * len(logstr))

    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
        logger.info(f"{name}: Shape {list(param.size())} ({param.numel()} parameters), " f"{param.dtype}")
    logger.info(f"{num_params} total parameters")

    for epoch in range(config.execution.num_epochs):
        t_start = time.time()
        loss = training(model_train, dataloader["train"], optim, args.device)
        dur = time.time() - t_start
        nbfnet_utils.log_results(
            logger,
            {
                "Training Loss": loss,
                "Duration (train) [s]": dur,
                "Throughput (train) [triples/s]": len(dataset[0].target_edge_type) / dur,
            },
            epoch=epoch + 1,
            partition="train",
        )
        if config.execution.do_valid:
            t_start = time.time()
            if args.device == "ipu":
                model_train.detachFromDevice()
            results = inference(model_valid, dataloader["valid"], metrics=("MR", "MRR", "hits@1", "hits@3", "hits@10"))
            if args.device == "ipu":
                model_valid.detachFromDevice()
            dur = time.time() - t_start
            nbfnet_utils.log_results(
                logger,
                {
                    **results,
                    "Duration (valid) [s]": dur,
                    "Throughput (valid) [triples/s]": len(dataset[1].target_edge_type) / dur,
                },
                epoch=epoch + 1,
                partition="validation",
            )

    if config.execution.do_test:
        t_start = time.time()
        if args.device == "ipu" and model_train.isAttachedToDevice():
            model_train.detachFromDevice()
        results = inference(
            model_test, dataloader["test"], metrics=["MR", "MRR", "hits@1", "hits@3", "hits@10"], annot="test"
        )
        dur = time.time() - t_start
        nbfnet_utils.log_results(
            logger,
            {
                **results,
                "Duration (test) [s]": dur,
                "Throughput (test) [triples/s]": len(dataset[2].target_edge_type) / dur,
            },
            epoch=epoch + 1,
            partition="test",
        )
