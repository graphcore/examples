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

import time
import os
import sys
import math
import ctypes
import random
import datetime
from functools import reduce
from collections import deque
from collections import defaultdict
from itertools import chain
import logging
import socket

import popart
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from bert_model import ExecutionMode, get_model, BertConfig
from bert_data import get_pretraining_dataset, get_squad_dataset
from bert_tf_loader import load_initializers_from_tf
from bert_optimizer import ScheduledOptimizerFactory, BaseOptimizerFactory
from phased_execution.weight_mapping import get_phased_initializers_from_default
import utils
import utils.popvision as popvision
from utils.inference import (create_callback_stepio,
                             realtime_scheduling,
                             compute_latency_from_durations,
                             compute_latency_from_callbacks)


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
            "FLOAT16", [args.batch_size * args.sequence_length, args.hidden_size])
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
        "UINT32", [args.batch_size * args.sequence_length])
    indices, positions = bert_add_embedding_inputs(args, model, sequence_info)

    segments = model.builder.addInputTensor(sequence_info, "segments")
    labels = []
    masks = []
    mask_info = popart.TensorInfo("UINT32", [args.batch_size, 1])
    if args.task == "PRETRAINING":
        masks.append(
            model.builder.addInputTensor(mask_info, "mask_tokens_mask_idx"))
        masks.append(
            model.builder.addInputTensor(mask_info, "sequence_mask_idx"))
        mlm_info = popart.TensorInfo(
            "UINT32", [args.batch_size, args.mask_tokens])
        labels.append(model.builder.addInputTensor(mlm_info, "mask_labels"))
        nsp_info = popart.TensorInfo(
            "UINT32", [args.batch_size])
        labels.append(model.builder.addInputTensor(nsp_info, "nsp_labels"))
    elif args.task == "SQUAD":
        masks.append(model.builder.addInputTensor(mask_info, "seq_pad_idx"))
        if not args.inference:
            labels_info = popart.TensorInfo(
                "UINT32", [args.batch_size])
            labels.append(model.builder.addInputTensor(
                labels_info, "start_labels"))
            labels.append(model.builder.addInputTensor(
                labels_info, "end_labels"))
    return indices, positions, segments, masks, labels


def bert_logits_graph(model, indices, positions, segments, masks, mode):
    if mode == ExecutionMode.PHASED:
        logits = model(indices, positions, segments, masks)
    else:
        logits = model.build_graph(indices, positions, segments, masks)
    return logits


def bert_infer_graph(model, logits, include_probs=True):
    # NOTE: include_probs added as we don't need to calculate the softmax if we only care about accuracy and not
    # about loss
    probs = None
    if model.config.task == "SQUAD":
        scope = model.squad_scope
        if model.config.execution_mode == ExecutionMode.PHASED:
            scope = model.scope_provider(model.builder, scope)
        with scope:
            predictions = list(
                model.builder.aiOnnx.argmax([logit],
                                            axis=1,
                                            keepdims=0,
                                            debugPrefix=f"{logit}/ArgMax")
                for logit in logits)
            if include_probs:
                probs = list(
                    model.builder.aiOnnx.softmax(
                        [logit], axis=1, debugPrefix=f"{logit}/Softmax")
                    for logit in logits)

                for prob in probs:
                    model.builder.setInplacePreferences(
                        prob, {"SoftmaxInplace": -1})
    elif model.config.task == "PRETRAINING":
        nsp_scope = model.nsp_scope
        mlm_scope = model.mlm_scope
        if model.config.execution_mode == ExecutionMode.PHASED:
            nsp_scope = model.scope_provider(model.builder, nsp_scope)
            mlm_scope = model.scope_provider(model.builder, mlm_scope)
        with nsp_scope:
            nsp_predictions = model.builder.aiOnnx.argmax(
                [logits[1]], axis=1, keepdims=0, debugPrefix="ArgMax")
            if include_probs:
                nsp_probs = model.builder.aiOnnx.softmax([logits[1]],
                                                         axis=1,
                                                         debugPrefix="Softmax")
        with mlm_scope:
            mlm_predictions = model.builder.aiOnnx.argmax(
                [logits[0]], axis=2, keepdims=0, debugPrefix="ArgMax")
            if include_probs:
                mlm_probs = model.builder.aiOnnx.softmax([logits[0]],
                                                         axis=2,
                                                         debugPrefix="Softmax")
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
    if model.config.execution_mode == ExecutionMode.PHASED:
        scope = model.scope_provider(model.builder, scope)
    return scope


def bert_loss_graph(args, model, probs, labels):
    if args.gradient_reduction_type == "Sum":
        reduction_type = popart.ReductionType.Sum
    elif args.gradient_reduction_type == "Mean":
        reduction_type = popart.ReductionType.Mean
    else:
        raise RuntimeError(f"Unknown gradient_reduction_type {args.gradient_reduction_type}")

    def loss(prob, label):
        ignore_index = get_ignore_index(label)
        scope = get_loss_scope(model, label)
        with scope:
            if ignore_index is not None:
                nllloss = model.builder.aiGraphcore.nllloss(
                    [prob, label],
                    reduction=reduction_type,
                    ignoreIndex=ignore_index,
                    debugPrefix=f"{label}/loss")
            else:
                nllloss = model.builder.aiGraphcore.nllloss(
                    [prob, label],
                    reduction=reduction_type,
                    debugPrefix=f"{label}/loss")
        return nllloss

    losses = [loss(*p_l) for p_l in zip(probs, labels)]

    if len(losses) > 1:
        scope = model.final_loss_scope
        if model.config.execution_mode == ExecutionMode.PHASED:
            scope = model.scope_provider(model.builder, scope)
        with scope:
            final_loss = model.builder.aiOnnx.sum(losses, "FinalLoss")
    else:
        final_loss = losses[0]

    return losses, final_loss


def bert_perplexity_graph(args, model, logits, labels):
    with model.mlm_scope:
        mlm_probs = model.builder.aiOnnx.softmax(
            [logits[0]], axis=2, debugPrefix="Softmax")

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
                                                       debugPrefix="TotalAttempted")
        else:
            total_attempted = builder.aiOnnx.constant(np.array(np.prod(builder.getTensorShape(label))).astype(np.int32), f"{label}_total")
        total_correct = builder.aiOnnx.reducesum([results],
                                                 axes=range(len(builder.getTensorShape(label))),
                                                 keepdims=0,
                                                 debugPrefix="TotalCorrect")
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


def bert_add_logit_outputs(model, logits):
    outputs = {}
    for logit in logits:
        outputs[logit] = popart.AnchorReturnType("ALL")
    for out in outputs.keys():
        model.builder.addOutputTensor(out)
    return outputs


def bert_session_options(args, model):
    engine_options = {}
    options = popart.SessionOptions()
    options.virtualGraphMode = popart.VirtualGraphMode.Manual
    options.enableFloatingPointChecks = args.floating_point_exceptions
    options.enableStochasticRounding = args.stochastic_rounding
    options.enableGroupedMatmuls = False
    options.enablePrefetchDatastreams = not args.minimum_latency_inference
    options.enableOutlining = not args.no_outlining
    partials_type = "half" if args.enable_half_partials else "float"
    options.partialsTypeMatMuls = partials_type
    options.convolutionOptions = {'partialsType': partials_type}
    if args.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = args.replication_factor
        engine_options["target.syncReplicasIndependently"] = "true"
    # Increasing the outlineThreshold prevents creating subgraphs of cheap Ops
    # such as add or reshapeInplace.
    # Instead only reusing ops with a highSubgraphValue such as matmul or normalisation.
    options.outlineThreshold = 10.0
    if args.execution_mode == "PIPELINE":
        options.enablePipelining = True
        options.autoRecomputation = popart.RecomputationType.Pipeline
    elif args.execution_mode == "PHASED":
        options.virtualGraphMode = popart.VirtualGraphMode.ExecutionPhases
        options.enableOutliningCopyCostPruning = False
        options.outlineThreshold = -np.inf
        options.executionPhaseSettings.phases = model.total_execution_phases
        options.batchSerializationSettings.factor = args.batch_serialize
        options.autoRecomputation = popart.RecomputationType.Standard
        options.explicitRecomputation = True
        options.aliasZeroCopy = True

        options.activationTensorLocationSettings.location.storage = popart.TensorStorage.OffChip

        varLocation = popart.TensorLocation()
        varLocation.storage = popart.TensorStorage.OffChip
        varLocation.loadTileSet = popart.TileSet.IO
        varLocation.storageTileSet = popart.TileSet.IO
        varLocation.replicatedTensorSharding = (popart.ReplicatedTensorSharding.On
                                                if args.replicated_weight_sharding else
                                                popart.ReplicatedTensorSharding.Off)

        options.weightTensorLocationSettings.location = varLocation
        options.optimizerStateTensorLocationSettings.location = varLocation
        options.accumulatorTensorLocationSettings.location = varLocation

        options.numIOTiles = args.num_io_tiles
        options.timeLimitScheduler = -1
        options.swapLimitScheduler = -1
        engine_options["target.syncReplicasIndependently"] = "false"
        if args.activations_on_chip:
            options.activationTensorLocationSettings = popart.TensorLocationSettings(
                popart.TensorStorage.OnChip, 0)

    if args.optimizer_state_offchip:
        options.optimizerStateTensorLocationSettings.location.storage = popart.TensorStorage.OffChip
    if args.gradient_accumulation_factor > 1:
        options.enableGradientAccumulation = True
        options.accumulationFactor = args.gradient_accumulation_factor
        if args.gradient_reduction_type == "Mean":
            options.accumulationReductionType = popart.ReductionType.Mean

        # When not replicated SyncPattern.SinglePipeline will provide better overlap
        # than this option.
        if args.optimizer_state_offchip and args.replication_factor > 1:
            options.accumulateOuterFragmentSettings = popart.AccumulateOuterFragmentSettings(
                popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized, [0])
    if args.engine_cache is not None:
        options.enableEngineCaching = True
        options.cachePath = args.engine_cache
    if args.profile:
        options.enableEngineCaching = False
    options.instrumentWithHardwareCycleCounter = args.report_hw_cycle_count
    options.disableGradAccumulationTensorStreams = True
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
    if args.task != "SQUAD":
        patterns.enablePattern("DisableAttnDropoutBwdPattern", False)

    if args.execution_mode == ExecutionMode.PHASED:
        patterns.enablePattern("TiedGatherPattern", False)
        patterns.enablePattern("SparseAccumulatePattern", False)

    if args.execution_mode == ExecutionMode.PIPELINE and args.recompute_checkpoint_every_layer and any(map(lambda l: l > 1, args.layers_per_ipu)):
        patterns.enablePattern("AccumulatePriorityPattern", True)

    if args.task == "PRETRAINING" and args.execution_mode != ExecutionMode.PHASED and args.gradient_accumulation_factor <= 1 and not args.inference:
        patterns.enablePattern("TiedGatherPattern", False)
        logger.warning("Running Pretraining without Gradient Accumulation will disable optimisations "
                       "for the Word Embedding weight. This will increase memory usage. "
                       "Consider enabling Gradient Accumulation.")

    if args.optimizer == "SGD" and args.optimizer_state_offchip and args.execution_mode != ExecutionMode.PHASED:
        patterns.enablePattern("TiedGatherPattern", False)
        logger.warning("Remote Optimizer State with SGD/SGD+M is not a recommended configuration")

    return patterns


def calc_required_ipus(args, model):
    if args.execution_mode == "PHASED":
        if args.phased_execution_type == "DUAL":
            num_ipus = 2
        else:
            num_ipus = 1
    else:
        num_ipus = model.total_ipus
    num_ipus *= args.replication_factor
    request_ipus = pow(2, math.ceil(math.log2(num_ipus)))
    logger.info(f"Need {num_ipus} IPUs. Requesting {request_ipus}")
    return request_ipus, num_ipus


def compile_graph_checked(args, session):
    start_time = time.time()
    session.prepareDevice()
    end_time = time.time()
    compile_time = end_time - start_time
    logger.info(f"Compiled. Duration {compile_time} seconds")
    if args.profile:
        popvision.save_app_info({"compile_time": compile_time})


def bert_training_session(model, args, feed, loss, device,
                          optimizer_factory):
    options = bert_session_options(args, model)

    patterns = bert_session_patterns(args)

    proto = model.builder.getModelProto()

    optimizer = optimizer_factory.create()

    logger.info("Creating Session")
    session = popart.TrainingSession(fnModel=proto,
                                     loss=loss,
                                     deviceInfo=device,
                                     optimizer=optimizer,
                                     dataFlow=feed,
                                     patterns=patterns,
                                     userOptions=options)

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
    log_name = f"{os.path.basename(args.checkpoint_dir)}."\
               f"{datetime.datetime.now().isoformat()}"
    log_dir = os.path.join(
        args.log_dir, log_name)
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def get_bert_dataset(model, args, inputs, embedding_dict=None, positional_dict=None, merge_both_embeddings=False):
    config = model.config
    shapeOf = model.builder.getTensorShape
    # The inputs after the first three (ind, pos, seg) are always lists
    inputs = reduce(chain, inputs[3:], inputs[:3])
    tensor_shapes = [(tensorId, shapeOf(tensorId)) for tensorId in inputs]

    if config.task == "PRETRAINING":
        return get_pretraining_dataset(
            tensor_shapes,
            input_files=args.input_files,
            sequence_length=config.sequence_length,
            mask_tokens=config.mask_tokens,
            vocab_length=config.vocab_length,
            batch_size=config.batch_size,
            batches_per_step=args.batches_per_step,
            accumulation_factor=args.gradient_accumulation_factor,
            replication_factor=args.replication_factor,
            duplication_factor=args.duplication_factor,
            shuffle=args.shuffle,
            generated_data=args.generated_data or args.synthetic_data,
            epochs_to_cache=args.epochs_to_cache,
            start_data_at_epoch=args.continue_training_from_epoch)

    if config.task == "SQUAD":
        ds = get_squad_dataset(
            tensor_shapes,
            input_file=args.input_files[0],
            output_dir=args.squad_results_dir,
            sequence_length=config.sequence_length,
            vocab_file=args.vocab_file,
            vocab_length=config.vocab_length,
            batch_size=config.batch_size,
            batches_per_step=args.batches_per_step,
            embedding_dict=embedding_dict,
            positional_dict=positional_dict,
            merge_both_embeddings=merge_both_embeddings,
            accumulation_factor=args.gradient_accumulation_factor,
            replication_factor=args.replication_factor,
            shuffle=args.shuffle,
            is_training=not args.inference,
            overwrite_cache=args.overwrite_cache,
            no_drop_remainder=args.no_drop_remainder,
            evaluate_script=args.squad_evaluate_script,
            generated_data=args.generated_data or args.synthetic_data,
            do_lower_case=args.do_lower_case,
            max_pipeline_stage=model.total_pipeline_stages if args.execution_mode == "PIPELINE" else 1,
            seed=args.seed,
            mpi_size=args.mpi_size,
            mpi_rank=args.mpi_rank,
            is_distributed= args.mpi_size > 1)
        return ds


def bert_reduce_metric(args, anchors, metrics, mean=False):
    accumulated_stats = args.gradient_accumulation_factor * args.batches_per_step
    if len(metrics) > 1:
        metric = np.add(*[anchors[metric] for metric in metrics])
        if mean:
            accumulated_stats *= len(metrics)
    else:
        metric = anchors[metrics[0]]
    return np.mean(metric / accumulated_stats)


def bert_output_stats(args, anchors, losses, accuracies):
    return (bert_reduce_metric(args, anchors, losses),
            bert_reduce_metric(args, anchors, accuracies, mean=True))


def bert_pretraining_stats(args, anchors, losses, accuracies):
    losses = map(lambda loss: bert_reduce_metric(args, anchors, [loss]), losses)
    accuracies = map(lambda acc: bert_reduce_metric(args, anchors, [acc]), accuracies)
    return tuple(losses), tuple(accuracies)


def bert_pretraining_inference_stats(args, anchors, losses, accuracies):
    if args.inference_lm_perplexity:
        loss = bert_reduce_metric(args, anchors, [losses[0]])
    else:
        loss = None
    accuracies = map(lambda acc: bert_reduce_metric(args, anchors, [acc]), accuracies)
    return loss, tuple(accuracies)


def save_model_and_stats(args, session, writer, step, epoch=None, step_in_filename=False):
    if not args.no_model_save:
        save_file = "model"
        if epoch is not None:
            save_file += f"_{epoch}"
        if step_in_filename:
            save_file += f":{step}"
        save_file += '.onnx'
        save_path = os.path.join(args.checkpoint_dir, save_file)
        logger.info(f"Saving model to: {save_path}")
        session.modelToHost(save_path)
        utils.save_model_statistics(save_path, writer, step)


class Iteration:
    def __init__(self, args, batches_per_step, steps_per_epoch, writer, recording_steps=None):
        self.start_epoch = args.continue_training_from_epoch
        self.count = self.start_epoch * steps_per_epoch
        self.epoch = 0
        self.epochs = args.epochs
        self.epochs_per_save = args.epochs_per_save
        self.steps_per_log = args.steps_per_log
        self.samples_per_step = batches_per_step * \
            args.gradient_accumulation_factor * args.replication_factor * args.batch_size
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = self.steps_per_epoch * self.epochs
        self.writer = writer
        self.task = args.task
        self.calculate_perplexity = args.inference_lm_perplexity
        # This should get overridden but will ensure we can always write a scalar to TB.
        self.learning_rate = 0
        if recording_steps is None:
            recording_steps = self.steps_per_epoch
        self.durations = deque(maxlen=recording_steps)
        self.cycles = deque(maxlen=recording_steps)
        if self.task == "PRETRAINING":
            self.mlm_losses = deque(maxlen=recording_steps)
            self.nsp_losses = deque(maxlen=recording_steps)
            self.mlm_accuracies = deque(maxlen=recording_steps)
            self.nsp_accuracies = deque(maxlen=recording_steps)
            if args.inference:
                self.stats_fn = bert_pretraining_inference_stats
            else:
                self.stats_fn = bert_pretraining_stats
        else:
            self.losses = deque(maxlen=recording_steps)
            self.accuracies = deque(maxlen=recording_steps)
            self.stats_fn = bert_output_stats

    def add_stats(self, duration, hw_cycles, *args):
        self.durations.append(duration)
        if hw_cycles:
            self.cycles.append(hw_cycles)
        loss, accuracy = self.stats_fn(*args)
        self.writer.add_scalar("defaultLearningRate",
                               self.learning_rate,
                               self.count)
        self.writer.add_scalar("throughput",
                               np.average(self.throughput),
                               self.count)
        if self.task == "PRETRAINING":
            self.mlm_losses.append(loss[0])
            self.nsp_losses.append(loss[1])
            self.mlm_accuracies.append(accuracy[0])
            self.nsp_accuracies.append(accuracy[1])
            self.writer.add_scalar("loss/MLM",
                                   np.average(self.mlm_losses),
                                   self.count)
            self.writer.add_scalar("loss/NSP",
                                   np.average(self.nsp_losses),
                                   self.count)
            self.writer.add_scalar("accuracy/MLM",
                                   np.average(self.mlm_accuracies),
                                   self.count)
            self.writer.add_scalar("accuracy/NSP",
                                   np.average(self.nsp_accuracies),
                                   self.count)
        else:
            self.losses.append(loss)
            self.accuracies.append(accuracy)
            self.writer.add_scalar("loss",
                                   np.average(self.losses),
                                   self.count)
            self.writer.add_scalar("accuracy",
                                   np.average(self.accuracies),
                                   self.count)


    def add_inference_stats(self, duration, hw_cycles, *args):
        self.durations.append(duration)
        if hw_cycles:
            self.cycles.append(hw_cycles)

        if self.task == "PRETRAINING":
            loss, accuracy = self.stats_fn(*args)
            self.mlm_accuracies.append(accuracy[0])
            self.nsp_accuracies.append(accuracy[1])

            if loss is not None:
                self.mlm_losses.append(loss)

    @property
    def throughput(self):
        return np.divide(self.samples_per_step, self.durations)

    def report_stats(self):
        avg = np.average
        status_string = \
            f"Iteration: {self.count:6} " \
            f"Epoch: {self.count/self.steps_per_epoch:6.2f}/{self.epochs} "
        if self.task == "PRETRAINING":
            status_string += \
                f"Loss (MLM NSP): {avg(self.mlm_losses):5.3f} {avg(self.nsp_losses):5.3f} " \
                f"Accuracy (MLM NSP): {avg(self.mlm_accuracies):5.3f} {avg(self.nsp_accuracies):5.3f} "
        else:
            status_string += \
                f"Loss: {avg(self.losses):5.3f} " \
                f"Accuracy: {avg(self.accuracies):5.3f} "
        status_string += \
            f"Learning Rate: {self.learning_rate:.5f} "
        status_string += \
            f"Duration: {avg(self.durations):6.4f} s " \
            f"Throughput: {avg(self.throughput):6.1f} samples/s"
        if self.cycles:
            status_string += f" Cycles: {int(avg(self.cycles))}"
        logger.info(status_string)

    def report_inference_stats(self, mean_latency, min_latency, max_latency, hw_cycles):
        avg = np.average
        status_string = \
            f"Iteration: {self.count:6} " \
            f"Duration: {avg(self.durations):6.4f} s " \
            f"Throughput: {avg(self.throughput):6.1f} samples/s"

        if self.task == "PRETRAINING":
            status_string += \
                f" Accuracy (MLM NSP): {avg(self.mlm_accuracies):5.3f} {avg(self.nsp_accuracies):5.3f}"

            if self.calculate_perplexity:
                status_string += \
                    f" LM Perplexity: {np.exp(avg(self.mlm_losses)):5.3f}"

        if mean_latency is not None:
            status_string += f" Per-sample Latency: {mean_latency} {min_latency} {max_latency} seconds (mean min max)"
        if hw_cycles is not None:
            status_string += f" Cycles: {hw_cycles}"
        logger.info(status_string)


def bert_process_data(args,
                      session,
                      data,
                      anchors,
                      losses,
                      accuracies,
                      iteration: Iteration,
                      optimizer_factory: BaseOptimizerFactory):
    stepio = popart.PyStepIO(data, anchors)

    start = time.time()
    session.run(stepio)
    duration = time.time() - start
    hw_cycles = session.getCycleCount() if args.report_hw_cycle_count else None

    iteration.add_stats(duration,
                        hw_cycles,
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

    iteration.add_inference_stats(
        duration, hw_cycles, args, anchors, losses, accuracies)

    mean_latency, min_latency, max_latency = compute_latency(
        args, start_times, end_times, iteration.durations)

    if (iteration.count % iteration.steps_per_log) == 0:
        iteration.report_inference_stats(mean_latency, min_latency, max_latency, hw_cycles)

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
                    optimizer_factory: BaseOptimizerFactory):
    losses = [loss for loss in losses]

    save_model_and_stats(args, session, writer,
                         iteration.count, iteration.epoch)

    for iteration.epoch in range(iteration.start_epoch, args.epochs):
        for data in dataset:
            bert_process_data(args, session, data, anchors,
                              losses, accuracies, iteration, optimizer_factory)

            if args.steps_per_save > 0 and (iteration.count % args.steps_per_save) == 0:
                save_model_and_stats(args, session, writer,
                                     iteration.count, iteration.epoch, True)

        if args.epochs_per_save > 0 and ((iteration.epoch + 1) % iteration.epochs_per_save) == 0:
            save_model_and_stats(args, session, writer,
                                 iteration.count, iteration.epoch + 1)

    save_model_and_stats(args, session, writer, iteration.count)


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

    if not losses:
        losses = None

    # Create the stepio once outside of the inference loop:
    static_data = {}
    start_times = defaultdict(list)
    end_times = defaultdict(list)

    stepio = None
    if args.low_latency_inference and args.task == "SQUAD":
        stepio = create_callback_stepio(static_data, anchors, start_times, end_times,
                                        dataset.batches_per_step)

    with realtime_scheduling(args.realtime_scheduler):
        for iteration.epoch in range(args.epochs_inference):
            for data in dataset:
                static_data.update({t: data[t] for t in inputs})
                result = bert_process_infer_data(args, session, static_data, anchors,
                                                 logits, iteration,
                                                 start_times, end_times, stepio,
                                                 accuracies, losses)

                if result is not None and save_results and iteration.epoch == args.epochs_inference - 1:
                    dataset.add_results(data, result)
                start_times.clear()
                end_times.clear()

    # If SQuAD save the predictions and run the evaulation script
    if save_results:
        dataset.write_predictions()


def acquire_device(args, request_ipus):
    if args.use_ipu_model:
        model_opts = {"numIPUs": request_ipus}
        if args.ipu_model_version is not None:
            model_opts["ipuVersion"] = args.device_version
        device = popart.DeviceManager().createIpuModelDevice(model_opts)
    else:
        connection_type = popart.DeviceConnectionType.Always
        if args.device_connection_type == "ondemand":
            connection_type = popart.DeviceConnectionType.OnDemand
        if args.execution_mode == "PHASED":
            if args.phased_execution_type == "DUAL":
                sync_pattern = popart.SyncPattern.ReplicaAndLadder
            else:
                sync_pattern = popart.SyncPattern.Full
        elif args.execution_mode == "PIPELINE" and args.replication_factor <= 1:
            sync_pattern = popart.SyncPattern.SinglePipeline
        else:
            sync_pattern = popart.SyncPattern.Full

        manager = popart.DeviceManager()
        manager.setOnDemandAttachTimeout(args.device_ondemand_timeout)
        if args.device_connection_type == "offline":
            opts = dict()
            opts["numIPUs"] = request_ipus
            opts["syncPattern"] = str(sync_pattern)
            if args.device_tiles:
                opts["tilesPerIPU"] = args.device_tiles
            if args.device_version:
                opts["ipuVersion"] = args.device_version
            device = manager.createOfflineIPUDevice(opts)
        else:
            if args.device_id:
                device = manager.acquireDeviceById(
                    args.device_id,
                    pattern=sync_pattern,
                    connectionType=connection_type)
            else:
                device = manager.acquireAvailableDevice(
                    request_ipus,
                    pattern=sync_pattern,
                    connectionType=connection_type)
    if device is None:
        raise OSError("Failed to acquire IPU.")
    logger.info(f"Acquired device: {device}")
    return device


def bert_pretrained_initialisers(config, args):

    if args.synthetic_data:
        logger.info("Initialising from synthetic_data")
        return None

    if args.generated_data:
        logger.info("Initialising from generated_data")
        return None

    init = None
    if args.onnx_checkpoint:
        logger.info(f"Initialising from ONNX checkpoint: {args.onnx_checkpoint}")
        init = utils.load_initializers_from_onnx(args.onnx_checkpoint)

    if args.tf_checkpoint:
        logger.info(f"Initialising from TF checkpoint: {args.tf_checkpoint}")
        init = load_initializers_from_tf(args.tf_checkpoint, True, config, args.task)

    if init is not None:
        init.update(**get_phased_initializers_from_default(args, init))

    return init


def main(args):
    set_library_seeds(args.seed)

    config = bert_config_from_args(args)

    initializers = bert_pretrained_initialisers(config, args)

    logger.info("Building Model")
    # Specifying ai.onnx opset9 for the slice syntax
    model = get_model(config,
                      mode=args.execution_mode,
                      initializers=initializers,
                      block=None)

    # If config.host_embedding is enabled, indices and positions will have the matrices instead of the index vector.
    indices, positions, segments, masks, labels = bert_add_inputs(args, model)
    logits = bert_logits_graph(model, indices, positions, segments, masks,
                               args.execution_mode)

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
    else:
        predictions, probs = bert_infer_graph(model, logits)
        losses, final_loss = bert_loss_graph(args, model, probs, labels)
        outputs, accuracies, losses = bert_add_validation_outputs(args, model, predictions, labels, losses)
        writer = bert_writer(args)

    embedding_dict, positional_dict = model.get_model_embeddings()

    dataset = get_bert_dataset(model,
                               args,
                               [indices, positions, segments, masks, labels],
                               embedding_dict,
                               positional_dict,
                               config.host_embedding == "MERGE")
    logger.info(f"Dataset length: {len(dataset)}")

    data_flow = popart.DataFlow(dataset.batches_per_step, outputs)

    iteration = Iteration(
        args,
        batches_per_step=dataset.batches_per_step,
        steps_per_epoch=len(dataset),
        writer=writer,
        recording_steps=args.aggregate_metrics_over_steps)

    request_ipus, required_ipus = calc_required_ipus(args, model)

    device = acquire_device(args, request_ipus)

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
            optimizer_factory = ScheduledOptimizerFactory(args,
                                                          iteration,
                                                          args.optimizer,
                                                          model.tensors)

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

            device.detach()
            logger.info("Training Finished")

    return session, iteration


def setup_logger(log_level, handler=None):

    # Define a root config with a format which is simpler for console use
    root = logging.getLogger()
    root.setLevel(log_level)
    root_handler = logging.StreamHandler(sys.stdout)
    root_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s',
                                       '%Y-%m-%d %H:%M:%S')
    root_handler.setFormatter(root_formatter)
    root.handlers = [root_handler]
    if handler is not None:
        root.handlers += [handler]

    # Define a specific Handler for this file that removes the root name.
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if handler is not None:
        logger.addHandler(handler)
    logger.propagate = False


if __name__ == "__main__":

    args = utils.parse_bert_args()

    if args.profile:
        popvision.set_profiling_vars(args.profile_dir, args.profile_instrument)
        popvision.set_logging_vars()
        args_dict = vars(args)
        args_dict["hostname"] = socket.gethostname()
        args_dict["command"] = ' '.join(sys.argv)
        popvision.save_app_info(args_dict)
        logging_handler = popvision.get_profile_logging_handler()
    else:
        logging_handler = None

    setup_logger(logging.getLevelName(args.log_level), logging_handler)

    if args.wandb:
        import wandb
        wandb.init(project="popart-bert", sync_tensorboard=True)
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.batch_size * args.replication_factor * args.gradient_accumulation_factor
        wandb.config.update(args)

    logger.info("Program Start")
    logger.info("Hostname: " + socket.gethostname())
    logger.info("Command Executed: " + str(sys.argv))

    # Run the main inference/training session by default
    if args.inference or not args.no_training:
        main(args)

    # If this was a training session and validation isn't disabled; validate.
    if not args.inference and not args.no_validation and not args.no_model_save:
        logger.info("Doing Validation")
        args.remap = False
        main(utils.get_validation_args(args))

    logger.info("Program Finished")
