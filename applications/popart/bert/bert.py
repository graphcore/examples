# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

import ctypes
import datetime
import errno
import glob
import logging
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
from collections import defaultdict
from functools import reduce
from itertools import chain

import numpy as np
import popart
import popdist
import popdist.popart
from distutils import version
LooseVersion = version.LooseVersion
from torch.utils.tensorboard import SummaryWriter

import utils
import utils.popvision as popvision
from bert_data import get_pretraining_dataset, get_squad_dataset
from bert_model import Bert, BertConfig
from bert_optimizer import ScheduledOptimizerFactory, LinearOptimizerFactory
from bert_tf_loader import load_initializers_from_tf
from utils.device import acquire_device, device_is_replicated
from utils.distributed import popdist_root, distributed_barrier
from utils.iteration import Iteration, PretrainingIteration
from utils.inference import (create_callback_stepio,
                             realtime_scheduling,
                             compute_latency_from_durations,
                             compute_latency_from_callbacks)
from utils import packed_bert_utils, load_initializers_from_onnx

logger = logging.getLogger('BERT')

so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "custom_ops.so")
if os.path.exists(so_path):
    ctypes.cdll.LoadLibrary(so_path)
else:
    logger.warning("Could not find custom_ops.so. Execute `make` before running this script.")


def set_library_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


def bert_config_from_args(args):
    return BertConfig(**{k: getattr(args, k)
                         for k in BertConfig._fields if hasattr(args, k)})


def bert_add_embedding_inputs(args, model, sequence_info):
    if args.host_embedding == "NONE":
        indices = model.builder.addInputTensor(sequence_info, "indices")
        positions = model.builder.addInputTensor(sequence_info, "positions")
    else:   # "ALL", "WORD", "MERGE"
        expanded_sequence_info = popart.TensorInfo(
            "FLOAT16", [args.micro_batch_size * args.sequence_length, args.hidden_size])
        indices = model.builder.addInputTensor(
            expanded_sequence_info, "indices_expanded")
        if args.host_embedding == "ALL":
            positions = model.builder.addInputTensor(
                expanded_sequence_info, "pos_expanded")
        else:
            positions = model.builder.addInputTensor(sequence_info, "positions")

    return indices, positions


def bert_add_inputs(args, model):
    sequence_info = popart.TensorInfo(
        "UINT32", [args.micro_batch_size * args.sequence_length])
    indices, positions = bert_add_embedding_inputs(args, model, sequence_info)

    segments = model.builder.addInputTensor(sequence_info, "segments")
    labels = []
    masks = []
    mask_info = popart.TensorInfo("UINT32", [args.micro_batch_size, 1])
    if args.task == "PRETRAINING":
        masks.append(
            model.builder.addInputTensor(mask_info, "mask_tokens_mask_idx"))
        masks.append(
            model.builder.addInputTensor(mask_info, "sequence_mask_idx"))
        mlm_info = popart.TensorInfo(
            "UINT32", [args.micro_batch_size, args.mask_tokens])
        labels.append(model.builder.addInputTensor(mlm_info, "mask_labels"))
        nsp_info = popart.TensorInfo(
            "UINT32", [args.micro_batch_size])
        labels.append(model.builder.addInputTensor(nsp_info, "nsp_labels"))
    elif args.task == "SQUAD":
        masks.append(model.builder.addInputTensor(mask_info, "seq_pad_idx"))
        if not args.inference:
            labels_info = popart.TensorInfo(
                "UINT32", [args.micro_batch_size])
            labels.append(model.builder.addInputTensor(
                labels_info, "start_labels"))
            labels.append(model.builder.addInputTensor(
                labels_info, "end_labels"))
    return indices, positions, segments, masks, labels


def bert_infer_graph(model, logits, include_probs=True):
    # NOTE: include_probs added as we don't need to calculate the softmax if we only care about accuracy and not
    # about loss
    probs = None
    if model.config.task == "SQUAD":
        with model.squad_scope:
            predictions = list(
                model.builder.aiOnnx.argmax([logit],
                                            axis=1,
                                            keepdims=0,
                                            debugContext=f"{logit}/ArgMax")
                for logit in logits)
            if include_probs:
                probs = list(
                    model.builder.aiOnnx.softmax(
                        [logit], axis=1, debugContext=f"{logit}/Softmax")
                    for logit in logits)

                for prob in probs:
                    model.builder.setInplacePreferences(
                        prob, {"SoftmaxInplace": -1})
    elif model.config.task == "PRETRAINING":
        with model.nsp_scope:
            nsp_predictions = model.builder.aiOnnx.argmax(
                [logits[1]], axis=1, keepdims=0, debugContext="ArgMax")
            if include_probs:
                nsp_probs = model.builder.aiOnnx.softmax([logits[1]],
                                                         axis=1,
                                                         debugContext="Softmax")
        with model.mlm_scope:
            mlm_predictions = model.builder.aiOnnx.argmax(
                [logits[0]], axis=2, keepdims=0, debugContext="ArgMax")
            if include_probs:
                mlm_probs = model.builder.aiOnnx.softmax([logits[0]],
                                                         axis=2,
                                                         debugContext="Softmax")
        predictions = [mlm_predictions, nsp_predictions]
        if include_probs:
            probs = [mlm_probs, nsp_probs]
    return predictions, probs


def get_ignore_index(label):
    if 'mask' in label:
        return 0
    return None


def get_loss_scope(model, label):
    if model.config.task == "SQUAD":
        scope = model.squad_scope
    elif 'nsp' in label:
        scope = model.nsp_scope
    else:
        scope = model.mlm_scope
    return scope


def bert_loss_graph(args, model, probs, labels):
    if args.gradient_reduction_type == "Sum":
        reduction_type = popart.ReductionType.Sum
    else:
        reduction_type = popart.ReductionType.Mean

    def loss(prob, label):
        ignore_index = get_ignore_index(label)
        scope = get_loss_scope(model, label)
        with scope:
            if ignore_index is not None:
                nllloss = model.builder.aiGraphcore.nllloss(
                    [prob, label],
                    reduction=reduction_type,
                    ignoreIndex=ignore_index,
                    debugContext=f"{label}/loss")
            else:
                nllloss = model.builder.aiGraphcore.nllloss(
                    [prob, label],
                    reduction=reduction_type,
                    debugContext=f"{label}/loss")
        return nllloss

    losses = [loss(*p_l) for p_l in zip(probs, labels)]

    if len(losses) > 1:
        with model.final_loss_scope:
            final_loss = model.builder.aiOnnx.sum(losses, "FinalLoss")
    else:
        final_loss = losses[0]

    return losses, final_loss


def bert_perplexity_graph(args, model, logits, labels):
    with model.mlm_scope:
        mlm_probs = model.builder.aiOnnx.softmax(
            [logits[0]], axis=2, debugContext="Softmax")

    losses, final_loss = bert_loss_graph(args, model, [mlm_probs], [labels[0]])

    losses.append(None)

    return losses, final_loss


def bert_accuracy_calculation(builder, prediction, label, ignore_index=None):
    # Prediction will be the output of an ArgMax -> INT32
    with builder.nameScope("Accuracy"):
        label = builder.aiOnnx.cast([label], "INT32")
        results = builder.aiOnnx.equal([prediction, label])
        results = builder.aiOnnx.cast([results], "INT32")
        if ignore_index is not None:
            _ii = builder.aiOnnx.constant(np.array(ignore_index).astype(np.int32), f"{label}_ignore_index")
            mask = builder.aiOnnx.equal([label, _ii], "Mask")
            mask = builder.aiOnnx.logical_not([mask], "~Mask")
            mask = builder.aiOnnx.cast([mask], "INT32")
            results = builder.aiOnnx.mul([results, mask], "MaskApply")
            total_attempted = builder.aiOnnx.reducesum([mask],
                                                       axes=range(len(builder.getTensorShape(mask))),
                                                       keepdims=0,
                                                       debugContext="TotalAttempted")
        else:
            total_attempted = builder.aiOnnx.constant(np.array(np.prod(builder.getTensorShape(label))).astype(np.int32), f"{label}_total")
        total_correct = builder.aiOnnx.reducesum([results],
                                                 axes=range(len(builder.getTensorShape(label))),
                                                 keepdims=0,
                                                 debugContext="TotalCorrect")
        total_correct = builder.aiOnnx.cast([total_correct], "FLOAT")
        total_attempted = builder.aiOnnx.cast([total_attempted], "FLOAT")
        accuracy = builder.aiOnnx.div([total_correct, total_attempted])
    return accuracy, total_attempted


def bert_add_validation_outputs(args, model, predictions, labels, losses):
    outputs = {}
    accuracies = []
    avg_losses = []
    for pred, label, loss in zip(predictions, labels, losses):
        with get_loss_scope(model, label):
            accuracy, num_attempted = bert_accuracy_calculation(model.builder, pred, label, get_ignore_index(label))
            accuracies.append(accuracy)
            outputs[accuracy] = popart.AnchorReturnType("SUM")

            if loss is not None:
                loss = model.builder.aiOnnx.cast([loss], "FLOAT")
                if args.gradient_reduction_type == "Sum":
                    loss = model.builder.aiOnnx.div([loss, num_attempted])
                avg_losses.append(loss)
                outputs[loss] = popart.AnchorReturnType("SUM")
    for out in outputs.keys():
        model.builder.addOutputTensor(out)
    return outputs, accuracies, avg_losses


def bert_add_outputs(args, model, logits, labels):
    if args.inference:
        accuracies = None
        losses = []
        if args.task == "PRETRAINING":
            # If this is a pretraining session, labels for NSP and MLM are already within the dataset,
            # so we can always calculate prediction performance
            predictions, _ = bert_infer_graph(model, logits, include_probs=False)

            if args.inference_lm_perplexity:
                losses, _ = bert_perplexity_graph(args, model, logits, labels)
            else:
                losses = [None, None]

            outputs, accuracies, losses = bert_add_validation_outputs(args, model, predictions, labels, losses)
        else:
            if args.inference_lm_perplexity:
                raise RuntimeError("Masked LM perplexity is only supported in pretraining.")

            outputs = bert_add_logit_outputs(model, logits)

        writer = None
        final_loss = None
    else:
        predictions, probs = bert_infer_graph(model, logits)
        losses, final_loss = bert_loss_graph(args, model, probs, labels)
        outputs, accuracies, losses = bert_add_validation_outputs(args, model, predictions, labels, losses)
        writer = bert_writer(args)

    return outputs, accuracies, losses, final_loss, writer


def bert_add_logit_outputs(model, logits):
    outputs = {}
    for logit in logits:
        outputs[logit] = popart.AnchorReturnType("ALL")
    for out in outputs.keys():
        model.builder.addOutputTensor(out)
    return outputs


def bert_optimizer_location_settings(args):
    storage = popart.TensorStorage.OnChip
    if args.optimizer_state_offchip:
        storage = popart.TensorStorage.OffChip
    rts = popart.ReplicatedTensorSharding.Off
    if args.replicated_tensor_sharding:
        rts = popart.ReplicatedTensorSharding.On

    return popart.TensorLocationSettings(popart.TensorLocation(storage, rts))


def bert_session_options(args, model):
    engine_options = {}
    options = popart.SessionOptions()
    options.virtualGraphMode = popart.VirtualGraphMode.Manual
    options.enableFloatingPointChecks = args.floating_point_exceptions
    options.enableStochasticRounding = args.stochastic_rounding
    options.enablePrefetchDatastreams = not args.minimum_latency_inference

    # These options are necessary to allow poplar to overlap processing of
    # multiple iterations in the host side
    options.defaultBufferingDepth = args.buffering_depth
    options.rearrangeAnchorsOnHost = False
    engine_options["exchange.streamBufferOverlap"] = "hostRearrangeOnly"

    options.enableOutlining = not args.no_outlining
    options.subgraphCopyingStrategy = popart.SubgraphCopyingStrategy.JustInTime
    partials_type = "half" if args.enable_half_partials else "float"
    options.partialsTypeMatMuls = partials_type
    options.convolutionOptions = {'partialsType': partials_type}
    if args.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = args.replication_factor
        engine_options["target.syncReplicasIndependently"] = "true"
    if args.use_popdist:
        popdist.popart.configureSessionOptions(options)
    # Increasing the outlineThreshold prevents creating subgraphs of cheap Ops
    # such as add or reshapeInplace.
    # Instead only reusing ops with a highSubgraphValue such as matmul or normalisation.
    options.outlineThreshold = 10.0
    if args.pipeline:
        options.enablePipelining = True
        options.autoRecomputation = popart.RecomputationType.Pipeline
        if args.recompute_checkpoint_every_layer and any(map(lambda l: l > 1, args.layers_per_ipu)):
            options.scheduleNonWeightUpdateGradientConsumersEarly = True

    options.optimizerStateTensorLocationSettings = bert_optimizer_location_settings(args)

    # RTS to shard optimizer states with multiple IPU Pods
    num_local_replicas = popdist.getNumLocalReplicas()
    num_total_replicas = popdist.getNumTotalReplicas()

    if num_total_replicas > num_local_replicas and args.replicated_tensor_sharding:
        # Fewer elements would not make sense to shard
        options.optimizerStateTensorLocationSettings.minElementsForReplicatedTensorSharding = num_local_replicas
        sharding_domain = popart.CommGroup(
            popart.CommGroupType.Consecutive, num_local_replicas)

        # Ensure all related tensors have the same sharding domain set
        options.weightTensorLocationSettings.location.shardingDomain = sharding_domain
        options.optimizerStateTensorLocationSettings.location.shardingDomain = sharding_domain
        options.accumulatorTensorLocationSettings.location.shardingDomain = sharding_domain

    if "Mean" in args.gradient_reduction_type:
        options.accumulationAndReplicationReductionType = popart.ReductionType.Mean
        options.meanAccumulationAndReplicationReductionStrategy = popart.MeanReductionStrategy.Post
        if args.gradient_reduction_type == "RunningMean":
            options.meanAccumulationAndReplicationReductionStrategy = popart.MeanReductionStrategy.Running

    if args.gradient_accumulation_factor > 1:
        options.enableGradientAccumulation = True
        options.accumulationFactor = args.gradient_accumulation_factor

        # When not replicated SyncPattern.SinglePipeline will provide better overlap
        # than this option.
        if device_is_replicated(args):
            if args.optimizer_state_offchip:
                options.accumulateOuterFragmentSettings = popart.AccumulateOuterFragmentSettings(
                    popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized, [0])
            elif args.replicated_tensor_sharding:
                # With OnChip + RTS this will cluster optimizer steps into
                # schedule bins. Improving outlining and scheduling time.
                options.accumulateOuterFragmentSettings = popart.AccumulateOuterFragmentSettings(
                    popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized)

    if args.engine_cache is not None:
        options.enableEngineCaching = True
        options.cachePath = args.engine_cache
    if args.profile:
        options.enableEngineCaching = False
    options.instrumentWithHardwareCycleCounter = args.report_hw_cycle_count
    options.disableGradAccumulationTensorStreams = not args.save_initializers_externally
    if args.max_copy_merge_size == -1:
        logger.debug("No copy merge size limit applied")
    else:
        logger.warning(
            f"Copy merge size limit set to {args.max_copy_merge_size}")
        engine_options["opt.maxCopyMergeSize"] = str(args.max_copy_merge_size)

    # Adding {"fullyConnectedPass", "TRAINING_BWD"} to some matmuls causes large
    # transposes before operations.
    if args.disable_fully_connected_pass:
        if args.task == "SQUAD" and args.sequence_length == 384:
            logger.warning(
                "Fully connected pass has been disabled. This may cause SQuAD 384 12-layer to go OOM.")
        options.enableFullyConnectedPass = False

    if args.inference and args.engine_cache is not None and not args.variable_weights_inference:
        logger.warning("Using engine cache with constant weights. Checkpoint weights will be ignored. "
                       "Use the `--variable-weights-inference` flag if checkpoint weights should be used.")

    if args.variable_weights_inference:
        options.constantWeights = False

    if args.group_host_syncs:
        options.groupHostSync = True

    if args.internal_exchange_optimisation_target is not None:
        engine_options["opt.internalExchangeOptimisationTarget"] = str(args.internal_exchange_optimisation_target)

    options.engineOptions = engine_options

    # Set synthetic data mode (if active)
    if args.synthetic_data:
        if args.synthetic_data_initializer == "zeros":
            options.syntheticDataMode = popart.SyntheticDataMode.Zeros
        else:
            options.syntheticDataMode = popart.SyntheticDataMode.RandomNormal
        logger.info(
            f"Running with Synthetic Data Type '{options.syntheticDataMode}'")
    return options


def bert_session_patterns(args):
    patterns = popart.Patterns()
    if args.disable_attention_dropout_bwd:
        patterns.enablePattern("DisableAttnDropoutBwdPattern", True)

    if args.task == "PRETRAINING" and args.gradient_accumulation_factor <= 1 and not args.inference:
        patterns.enablePattern("TiedGatherPattern", False)
        logger.warning("Running Pretraining without Gradient Accumulation will disable optimisations "
                       "for the Word Embedding weight. This will increase memory usage. "
                       "Consider enabling Gradient Accumulation.")

    if args.optimizer == "SGD" and args.optimizer_state_offchip:
        patterns.enablePattern("TiedGatherPattern", False)
        logger.warning("Remote Optimizer State with SGD/SGD+M is not a recommended configuration")

    return patterns


def compile_graph_checked(args, session):

    start_time = time.time()

    if args.compile_only:
        session.compileAndExport(args.engine_cache)
    else:
        session.prepareDevice()

    end_time = time.time()

    compile_time = end_time - start_time
    logger.info(f"Compiled. Duration {compile_time} seconds")

    if args.profile:
        popvision.save_app_info({"compile_time": compile_time})

    if args.compile_only:
        sys.exit(0)


def bert_distributed_training_session(args, **kwargs):
    try:
        import horovod.popart as hvd
        hvd.init()
    except ImportError:
        raise ImportError("Could not find the PopART horovod extension. "
                          "Please install the horovod .whl provided in the Poplar SDK.")

    session = hvd.DistributedTrainingSession(**kwargs)
    logger.info("Compiling Training Graph")
    compile_graph_checked(args, session)

    logger.info("Broadcasting weights to all instances")
    hvd.broadcast_weights(session)

    return session


def bert_training_session(model, args, feed, loss, device,
                          optimizer_factory):
    options = bert_session_options(args, model)

    patterns = bert_session_patterns(args)

    proto = model.builder.getModelProto()

    optimizer = optimizer_factory.create()

    logger.info("Creating Session")
    session_kwargs = dict(fnModel=proto,
                          loss=loss,
                          deviceInfo=device,
                          optimizer=optimizer,
                          dataFlow=feed,
                          patterns=patterns,
                          userOptions=options)
    if args.use_popdist:
        session = bert_distributed_training_session(args, **session_kwargs)
    else:
        session = popart.TrainingSession(**session_kwargs)
        logger.info("Compiling Training Graph")
        compile_graph_checked(args, session)

    session.weightsFromHost()
    session.setRandomSeed(args.seed)

    anchors = session.initAnchorArrays()

    return session, anchors


def bert_inference_session(model, args, feed, device):
    options = bert_session_options(args, model)

    patterns = bert_session_patterns(args)

    proto = model.builder.getModelProto()

    logger.info("Creating Session")
    session = popart.InferenceSession(fnModel=proto,
                                      deviceInfo=device,
                                      dataFlow=feed,
                                      patterns=patterns,
                                      userOptions=options)

    logger.info("Compiling Inference Graph")
    compile_graph_checked(args, session)

    session.weightsFromHost()
    session.setRandomSeed(args.seed)

    anchors = session.initAnchorArrays()

    return session, anchors


def bert_writer(args):
    writer = None
    if args.log_dir is not None and popdist_root(args):
        log_name = f"{os.path.basename(args.checkpoint_dir)}."\
                   f"{datetime.datetime.now().isoformat()}"
        log_dir = os.path.join(
            args.log_dir, log_name)
        writer = SummaryWriter(log_dir=log_dir)
    return writer


def get_bert_dataset(model, args, inputs):
    shapeOf = model.builder.getTensorShape
    # The inputs after the first three (ind, pos, seg) are always lists
    inputs = reduce(chain, inputs[3:], inputs[:3])
    tensor_shapes = [(tensorId, shapeOf(tensorId)) for tensorId in inputs]

    if args.task == "PRETRAINING":
        ds = get_pretraining_dataset(args, tensor_shapes)
    elif args.task == "SQUAD":
        ds = get_squad_dataset(args,
                               tensor_shapes,
                               host_embeddings=model.get_model_embeddings())
    else:
        raise RuntimeError(f"Unsupported Task {args.task} in get_bert_dataset")

    return ds


def save_model(args, session, step, epoch=None, step_in_filename=False):
    if not args.no_model_save and popdist_root(args):
        save_file = "model"
        if epoch is not None:
            save_file += f"_{epoch}"
        if step_in_filename:
            save_file += f":{step}"

        if args.save_initializers_externally:
            save_dir = Path(args.checkpoint_dir, save_file)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = args.checkpoint_dir
        save_file += '.onnx'
        save_path = os.path.join(save_dir, save_file)
        save_vars = 'vars'.join(save_path.rsplit('model', 1))
        if args.save_initializers_externally:
            if hasattr(args, 'save_vars_prev') and os.path.exists(args.save_vars_prev):
                logger.debug(f'Updating external location for vars to {args.save_vars_prev}.')
                session.updateExternallySavedTensorLocations(args.save_vars_prev, save_vars)
        session.modelToHost(save_path)
        args.save_vars_prev = save_vars
        logger.info(f"Saved model to: {save_path}.")
        if args.save_initializers_externally:
            logger.info(f"Saved variables(weights and optimizer state) to: {save_vars}.")


def bert_process_data(args,
                      session,
                      data,
                      anchors,
                      losses,
                      accuracies,
                      iteration: Iteration,
                      optimizer_factory):
    stepio = popart.PyStepIO(data, anchors)

    start = time.time()
    session.run(stepio)
    duration = time.time() - start
    hw_cycles = session.getCycleCount() if args.report_hw_cycle_count else None

    iteration.add_stats(duration,
                        hw_cycles,
                        data,
                        args,
                        anchors,
                        losses,
                        accuracies)

    if (iteration.count % iteration.steps_per_log) == 0:
        iteration.report_stats()

    if args.profile:
        sys.exit(0)

    # The following will only be true if:
    #   Learning rate mode is STEP and the current total step counter is in the schedule
    #   Learning rate mode is EPOCH and the current epoch has just changed to one in the schedule
    if optimizer_factory.should_update(iteration):
        optimizer = optimizer_factory.update_and_create(iteration)
        session.updateOptimizerFromHost(optimizer)

    iteration.count += 1


def compute_latency(args,
                    start_times,
                    end_times,
                    durations):
    if args.low_latency_inference:
        if not start_times or not end_times:
            logger.warning("No stepio callback times recorded. Using durations for fallback calculation.")
        else:
            return compute_latency_from_callbacks(start_times, end_times, args.batches_per_step)
    return compute_latency_from_durations(durations)


def bert_process_infer_data(args,
                            session,
                            data,
                            anchors,
                            logits,
                            iteration: Iteration,
                            start_times=None,
                            end_times=None,
                            stepio=None,
                            accuracies=None,
                            losses=None):
    if stepio is None:
        stepio = popart.PyStepIO(data, anchors)

    start = time.perf_counter()
    session.run(stepio)
    duration = time.perf_counter() - start
    hw_cycles = session.getCycleCount() if args.report_hw_cycle_count else None

    iteration.add_stats(duration, hw_cycles, data, args, anchors, losses, accuracies)

    mean_latency, min_latency, max_latency, p99_latency, p999_latency = compute_latency(
        args, start_times, end_times, iteration.durations)

    if (iteration.count % iteration.steps_per_log) == 0:
        iteration.report_inference_stats(mean_latency, min_latency, max_latency, p99_latency, p999_latency)

    if args.profile:
        sys.exit(0)

    iteration.count += 1

    if args.task == "PRETRAINING":
        return None
    elif args.task == "SQUAD":
        logit = anchors[logits[0]]
        return [result.reshape(-1, args.sequence_length) for result in np.split(logit, 2, axis=-1)]

    return [anchors[logit] for logit in logits]


def bert_train_loop(args,
                    session,
                    writer,
                    dataset,
                    accuracies,
                    losses,
                    anchors,
                    iteration,
                    optimizer_factory):
    start_epoch = iteration.epoch
    for iteration.epoch in range(start_epoch, iteration.epochs):
        for data in dataset:
            bert_process_data(args, session, data, anchors,
                              losses, accuracies, iteration, optimizer_factory)

            if args.steps_per_save > 0 and (iteration.count % args.steps_per_save) == 0:
                save_model(args, session,
                           iteration.count, iteration.epoch, True)

            if args.training_steps and iteration.count >= args.training_steps:
                logger.info(f"Ending Training at {iteration.count} Steps")
                return

        if args.epochs_per_save > 0 and ((iteration.epoch + 1) % iteration.epochs_per_save) == 0:
            save_model(args, session, iteration.count, iteration.epoch + 1)


def bert_infer_loop(args,
                    session,
                    dataset,
                    inputs,
                    logits,
                    anchors,
                    accuracies,
                    losses,
                    iteration: Iteration):
    save_results = args.task == "SQUAD" and not (args.synthetic_data or args.generated_data)
    micro_batches = args.batches_per_step * args.replication_factor

    # Create the stepio once outside of the inference loop:
    static_data = {}
    start_times = defaultdict(list)
    end_times = defaultdict(list)

    stepio = None
    if args.low_latency_inference and args.task == "SQUAD":
        stepio = create_callback_stepio(static_data, anchors, start_times,
                                        end_times, dataset.batches_per_step,
                                        args.replication_factor)

    with realtime_scheduling(args.realtime_scheduler):
        for iteration.epoch in range(args.epochs_inference):
            for data in dataset:
                static_data.update({t: data[t].reshape(micro_batches, -1) for t in inputs})
                result = bert_process_infer_data(args, session, static_data, anchors,
                                                 logits, iteration,
                                                 start_times, end_times, stepio,
                                                 accuracies, losses)

                if result is not None and save_results and iteration.epoch == args.epochs_inference - 1:
                    dataset.add_results(data, result)
                start_times.clear()
                end_times.clear()

    # If SQuAD run the evaluate-v1.1.py script and save the predictions
    if save_results:
        results = dataset.write_predictions()
        if args.wandb and results is not None:
            for k, v in results.items():
                wandb.run.summary[k] = v


def bert_required_ipus(args, model):
    return model.total_ipus * args.replication_factor


def bert_pretrained_initialisers(config, args):
    if args.synthetic_data:
        logger.info("Initialising from synthetic_data")
        return None

    if args.generated_data:
        logger.info("Initialising from generated_data")
        return None

    # The initialised weights will be broadcast after the session has been created
    if not popdist_root(args):
        return None

    init = None
    if args.onnx_checkpoint:
        logger.info(f"Initialising from ONNX checkpoint: {args.onnx_checkpoint}")
        init = load_initializers_from_onnx(args.onnx_checkpoint)

    if args.tf_checkpoint:
        logger.info(f"Initialising from TF checkpoint: {args.tf_checkpoint}")
        init = load_initializers_from_tf(args.tf_checkpoint, True, config, args.task)

    return init


def bert_optimizer_factory(args, model, iteration):
    if args.learning_rate_function == "Linear":
        return LinearOptimizerFactory(args,
                                      iteration,
                                      model.tensors)
    else:
        return ScheduledOptimizerFactory(args,
                                         iteration,
                                         model.tensors)


def bert_iteration(args, dataset, writer):
    if args.task == "PRETRAINING":
        return PretrainingIteration(
            args,
            steps_per_epoch=len(dataset),
            writer=writer,
            recording_steps=args.aggregate_metrics_over_steps)
    else:
        return Iteration(
            args,
            steps_per_epoch=len(dataset),
            writer=writer,
            recording_steps=args.aggregate_metrics_over_steps)


def main(args):
    set_library_seeds(args.seed)

    config = bert_config_from_args(args)

    initializers = bert_pretrained_initialisers(config, args)

    logger.info("Building Model")
    model = Bert(config,
                 pipeline=args.pipeline,
                 initializers=initializers)

    if not config.use_packed_sequence_format:
        # If config.host_embedding is enabled, indices and positions will have the matrices instead of the index vector.
        indices, positions, segments, masks, labels = bert_add_inputs(args, model)
        logits = model.build_graph(indices, positions, segments, masks)
        outputs, accuracies, losses, final_loss, writer = bert_add_outputs(args, model, logits, labels)
        dataset = get_bert_dataset(model, args, [indices, positions, segments, masks, labels])

    else:  # use_packed_sequence_format
        if args.task != "PRETRAINING":
            raise RuntimeError("Packed sequence format currently only supported for pretraining.")
        input_tensor_shapes = packed_bert_utils.add_inputs(model)
        logits = packed_bert_utils.logits_graph(model)
        losses, accuracies, final_loss, outputs = packed_bert_utils.pretraining_loss_and_accuracy(model, logits)
        writer = bert_writer(args) if not args.inference else None
        dataset = get_pretraining_dataset(args, input_tensor_shapes)

    device = acquire_device(args, bert_required_ipus(args, model))

    logger.info(f"Dataset length: {len(dataset)}")

    data_flow = popart.DataFlow(args.batches_per_step, outputs)

    iteration = bert_iteration(args, dataset, writer)

    if args.inference:
        session, anchors = bert_inference_session(
            model, args, data_flow, device)
        logger.info("Inference Started")
        inputs = [indices, positions, segments, *masks, *labels]
        bert_infer_loop(args, session,
                        dataset, inputs, logits, anchors,
                        accuracies, losses, iteration)
        device.detach()
    else:
        if not args.no_training:
            optimizer_factory = bert_optimizer_factory(args, model, iteration)
            if args.save_initializers_externally:
                save_dir = Path(args.checkpoint_dir,
                                f'model_{args.continue_training_from_epoch}')
                save_dir.mkdir(parents=True, exist_ok=True)
                weight_tensors = [item for sublist in model.tensors.values() for item in sublist]
                vars_path = f'vars_{args.continue_training_from_epoch}.onnx'
                vars_path = os.path.join(save_dir, vars_path)
                model.builder.saveInitializersExternally(weight_tensors,
                                                         vars_path)

            session, anchors = bert_training_session(model,
                                                     args,
                                                     data_flow,
                                                     final_loss,
                                                     device,
                                                     optimizer_factory)
            logger.info("Training Started")
            bert_train_loop(args, session, writer,
                            dataset, accuracies, losses, anchors,
                            iteration, optimizer_factory)

            save_model(args, session, iteration.count)
            if args.wandb_save_checkpoints:
                artifact = wandb.Artifact(name=args.wandb_save_checkpoints, type="model")
                artifact.add_dir(args.checkpoint_dir)
                wandb.log_artifact(artifact)

            device.detach()
            logger.info("Training Finished")

    return session, iteration


def setup_logger(log_level, handler=None):
    # Define a root config with a format which is simpler for console use
    root = logging.getLogger()
    root.setLevel(log_level)
    root_handler = logging.StreamHandler(sys.stdout)
    root_formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        '%Y-%m-%d %H:%M:%S')
    root_handler.setFormatter(root_formatter)
    root.handlers = [root_handler]
    if handler is not None:
        root.handlers += [handler]

    # Define a specific Handler for this file that removes the root name.
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s',
        '%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.handlers = [console]
    if handler is not None:
        logger.handlers += [handler]
    logger.propagate = False


if __name__ == "__main__":
    setup_logger(logging.INFO)

    args = utils.parse_bert_args()
    if not (args.synthetic_data or args.generated_data):
        for filepath in args.input_files:
            if len(glob.glob(filepath)) == 0:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    if args.save_initializers_externally:
        if os.path.isabs(args.checkpoint_dir):
            # This is a needed because relative path conventions are different
            # for onnx reader and popart builder object. Calling onnx external
            # data_helper explicity solves this issue, but does not support abs paths.
            # Onnx external data helper removes leading `/` while reading the checkpoint back."
            # "https://github.com/onnx/onnx/blob/v1.10.0/onnx/external_data_helper.py#L46"
            raise ValueError("Please specify relative path for `checkpoint_dir` when saving initializers externally. ")

    if args.profile:
        path = args.profile_dir
        if args.use_popdist:
            path += f"_rank{args.popdist_rank}"
        popvision.set_profiling_vars(path, args.profile_instrument)
        popvision.set_logging_vars()
        args_dict = vars(args)
        args_dict["hostname"] = socket.gethostname()
        args_dict["command"] = ' '.join(sys.argv)
        popvision.save_app_info(args_dict)
        logging_handler = popvision.get_profile_logging_handler()
    else:
        logging_handler = None

    setup_logger(logging.getLevelName(args.log_level), logging_handler)

    if args.wandb and popdist_root(args):
        import wandb
        wandb.init(project="popart-bert", config=args, sync_tensorboard=True, settings=wandb.Settings(console="wrap"))
        if args.wandb_checkpoint:
            artifact = wandb.use_artifact(args.wandb_checkpoint, type='model')
            artifact_dir = artifact.download()
            args.onnx_checkpoint = os.path.join(artifact_dir, "model.onnx")

    logger.info("Program Start")
    logger.info("Hostname: " + socket.gethostname())
    logger.info("Command Executed: " + str(sys.argv))

    # Run the main inference/training session by default
    if args.inference or not args.no_training:
        main(args)

    # If this was a training session and validation isn't disabled; validate.
    if not args.inference and not args.no_validation and not args.no_model_save and popdist_root(args):
        logger.info("Doing Validation")
        main(utils.get_validation_args(args))

    logger.info("Program Finished")
