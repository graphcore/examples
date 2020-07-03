# Copyright 2019 Graphcore Ltd.
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

from bert_model import Bert, BertConfig
from bert_data import get_pretraining_dataset, get_squad_dataset
from bert_tf_loader import load_initializers_from_tf
from bert_optimizer import ScheduledOptimizerFactory
import utils

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


def bert_logits_graph(model, indices, positions, segments, masks):
    logits = model.build_graph(indices, positions, segments, masks)
    return logits


def bert_perplexity_graph(model, logits, labels):
    with model.mlm_scope:
        mlm_probs = model.builder.aiOnnx.softmax(
            [logits[0]], axis=2, debugPrefix="Softmax")

    losses = bert_loss_graph(model, [mlm_probs], [labels[0]])

    # Due to limitations on the way loss is handled with Popart, we'll anchor the loss and calculate
    # the perplexity on the host in the stats function.
    return losses


def bert_infer_graph(model, logits, include_probs=True):
    # NOTE: include_probs added as we don't need to calculate the softmax if we only care about accuracy and not
    # about loss
    probs = None
    if model.config.task == "SQUAD":
        with model.squad_scope:
            predictions = list(
                model.builder.aiOnnx.argmax(
                    [logit], axis=1, keepdims=0, debugPrefix=f"{logit}/ArgMax")
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
        with model.nsp_scope:
            nsp_predictions = model.builder.aiOnnx.argmax(
                [logits[1]], axis=1, keepdims=0, debugPrefix="ArgMax")

            if include_probs:
                nsp_probs = model.builder.aiOnnx.softmax(
                    [logits[1]], axis=1, debugPrefix="Softmax")
        with model.mlm_scope:
            mlm_predictions = model.builder.aiOnnx.argmax(
                [logits[0]], axis=2, keepdims=0, debugPrefix="ArgMax")

            if include_probs:
                mlm_probs = model.builder.aiOnnx.softmax(
                    [logits[0]], axis=2, debugPrefix="Softmax")

        predictions = [mlm_predictions, nsp_predictions]
        if include_probs:
            probs = [mlm_probs, nsp_probs]
    return predictions, probs


def bert_loss_graph(model, probs, labels):
    def loss(prob, label):
        if model.config.task == "SQUAD":
            with model.squad_scope:
                nllloss = model.builder.aiGraphcore.nllloss(
                    [prob, label],
                    reduction=popart.ReductionType.NoReduction,
                    debugPrefix=f"{label}/loss")
        elif 'nsp' in label:
            with model.nsp_scope:
                nllloss = model.builder.aiGraphcore.nllloss(
                    [prob, label],
                    reduction=popart.ReductionType.NoReduction,
                    ignoreIndex=2,
                    debugPrefix=f"{label}/loss")
        else:
            with model.mlm_scope:
                nllloss = model.builder.aiGraphcore.nllloss(
                    [prob, label],
                    reduction=popart.ReductionType.NoReduction,
                    ignoreIndex=0,
                    debugPrefix=f"{label}/loss")

        return nllloss

    return [loss(*p_l) for p_l in zip(probs, labels)]


def bert_add_logit_outputs(model, logits):
    outputs = {}
    for logit in logits:
        outputs[logit] = popart.AnchorReturnType("ALL")
    for out in outputs.keys():
        model.builder.addOutputTensor(out)
    return outputs


def bert_add_validation_outputs(model, predictions, losses):
    outputs = {}
    for pred in predictions:
        outputs[pred] = popart.AnchorReturnType("ALL")
    for loss in losses:
        outputs[loss] = popart.AnchorReturnType("ALL")
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
    options.enableOutlining = not args.no_outlining
    # Increasing the outlineThreshold prevents creating subgraphs of cheap Ops
    # such as add or reshapeInplace.
    # Instead only reusing ops with a highSubgraphValue such as matmul or normalisation.
    options.outlineThreshold = 10.0
    if args.execution_mode == "PIPELINE":
        options.enablePipelining = True
        options.autoRecomputation = popart.RecomputationType.Pipeline
    elif args.execution_mode == "PINGPONG":
        options.virtualGraphMode = popart.VirtualGraphMode.PingPong
        options.outlineThreshold = -np.inf
        options.enableOutliningCopyCostPruning = False
        options.pingPongPhases = model.total_ping_pong_phases
    if args.gradient_accumulation_factor > 1:
        options.enableGradientAccumulation = True
        options.accumulationFactor = args.gradient_accumulation_factor
    if args.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = args.replication_factor
    if args.engine_cache is not None:
        options.enableEngineCaching = True
        options.cachePath = args.engine_cache
    if args.gc_profile:
        options.enableEngineCaching = False
        options.reportOptions = {
            "showVarStorage": "true",
            "showPerIpuMemoryUsage": "true",
            "showExecutionSteps": "true"
        }
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

    options.engineOptions = engine_options

    # Set synthetic data mode (if active)
    if args.synthetic_data_type:
        if args.synthetic_data_type == "random_normal":
            options.syntheticDataMode = popart.SyntheticDataMode.RandomNormal
        elif args.synthetic_data_type == "zeros":
            options.syntheticDataMode = popart.SyntheticDataMode.Zeros
        logger.info(
            f"Running with Synthetic Data Type '{options.syntheticDataMode}'")
    return options


def bert_session_patterns(args):
    patterns = popart.Patterns()
    if args.task != "SQUAD":
        patterns.enablePattern("DisableAttnDropoutBwdPattern", False)
    return patterns


def calc_required_ipus(args, model):
    if args.execution_mode == "PINGPONG":
        num_ipus = 2
    else:
        num_ipus = math.ceil(model.config.num_layers /
                             model.config.layers_per_ipu) + model.layer_offset
    num_ipus *= args.replication_factor
    request_ipus = pow(2, math.ceil(math.log2(num_ipus)))
    logger.info(f"Need {num_ipus} IPUs. Requesting {request_ipus}")
    return request_ipus, num_ipus


def compile_graph_checked(args, session):
    try:
        start_time = time.time()
        session.prepareDevice()
        end_time = time.time()
        logger.info(f"Compiled. Duration {end_time - start_time} seconds")
    except popart.OutOfMemoryException as e:
        utils.fetch_reports(args, session=session, exception=e)
        raise e


def bert_training_session(model, args, feed, losses, device,
                          optimizer_factory):
    options = bert_session_options(args, model)

    patterns = bert_session_patterns(args)

    def final_loss(losses):
        with model.final_loss_scope:
            reduced_losses = []
            for loss in losses:
                reduced_losses.append(
                    model.builder.aiOnnx.reducesum([loss], [0, 1], 0, f"Reduce{loss}"))
            loss_sum = model.builder.aiOnnx.sum(reduced_losses, "FinalLoss")
            return loss_sum

    loss = final_loss(losses)
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

    utils.fetch_reports(args, session=session)

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

    utils.fetch_reports(args, session=session)

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
            synthetic=args.synthetic_data,
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
            synthetic=args.synthetic_data,
            do_lower_case=args.do_lower_case,
            max_pipeline_stage=model.total_pipeline_stages if args.execution_mode == "PIPELINE" else 1,
            seed=args.seed,
            mpi_size=args.mpi_size,
            mpi_rank=args.mpi_rank,
            is_distributed= args.mpi_size > 1)
        return ds


def bert_step_loss(losses, anchors, num_unmasked, padding_masks):
    master_mask = num_unmasked != 0
    for loss, mask in zip(losses, padding_masks):
        anchors[loss][np.logical_not(mask)] = 0
    combined_loss = reduce(np.add, map(lambda loss: anchors[loss], losses))
    # Mask both num_losses and combined_loss to remove entries where all labels are ignored
    num_unmasked = num_unmasked[master_mask]
    combined_loss = combined_loss[master_mask]
    # Calculate mean loss for each token
    combined_loss /= num_unmasked
    # Calculate mean loss for step
    step_loss = np.mean(combined_loss)
    return step_loss


def bert_step_accuracy(labels, anchors, predictions, num_unmasked, padding_masks):
    total_correct = 0
    for pred, label, mask in zip(map(lambda p: anchors[p], predictions),
                                 labels,
                                 padding_masks):
        equal = pred.reshape(label.shape) == label
        total_correct += np.sum(equal[mask])
    total_attempted = np.sum(num_unmasked)
    step_accuracy = total_correct / total_attempted
    return step_accuracy


def bert_output_stats(labels, anchors, losses, predictions, ignore_index=None):
    if ignore_index is not None:
        padding_masks = [label != ignore_index for label in labels]
    else:
        padding_masks = [np.ones(label.shape, np.bool) for label in labels]
    num_unmasked = np.sum(np.array(padding_masks, dtype=np.int8), axis=0)

    # In the case of inference w/ labels, we don't care about the loss, so it can be ignored
    if losses[0] is None:
        step_loss = None
    else:
        step_loss = bert_step_loss(losses, anchors, num_unmasked, padding_masks)

    step_accuracy = bert_step_accuracy(labels, anchors, predictions, num_unmasked, padding_masks)
    return step_loss, step_accuracy


def bert_pretraining_stats(labels, anchors, losses, predictions):
    if losses is None:
        losses = [None, None]

    # For pretraining inference with perplexity, we'll only have a single loss
    if len(losses) == 1:
        losses.append(None)

    mlm_loss, mlm_acc = bert_output_stats(
        [labels[0]], anchors, [losses[0]], [predictions[0]], 0)
    nsp_loss, nsp_acc = bert_output_stats(
        [labels[1]], anchors, [losses[1]], [predictions[1]], 2)

    return [mlm_loss, nsp_loss], [mlm_acc, nsp_acc]


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


    def add_inference_stats(self, duration, hw_cycles, labels_data, anchors, predictions, losses):
        self.durations.append(duration)
        if hw_cycles:
            self.cycles.append(hw_cycles)

        if labels_data:

            loss, accuracy = self.stats_fn(labels_data, anchors, losses, predictions)

            if self.task == "PRETRAINING":
                self.mlm_accuracies.append(accuracy[0])
                self.nsp_accuracies.append(accuracy[1])

                if self.calculate_perplexity:
                    self.mlm_losses.append(loss[0])
            else:
                self.accuracies.append(accuracy)

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



def bert_process_data(args, session, labels, data, anchors,
                      losses, predictions, iteration: Iteration,
                      optimizer_factory: ScheduledOptimizerFactory):
    labels_data = [data[label] for label in labels]
    if not np.any([np.any(label) for label in labels_data]):
        # Label may be all padding due to args.vocab_length being smaller than when the data was generated
        return

    stepio = popart.PyStepIO(data, anchors)

    start = time.time()
    session.run(stepio)
    duration = time.time() - start
    hw_cycles = session.getCycleCount() if args.report_hw_cycle_count else None

    iteration.add_stats(duration, hw_cycles, labels_data,
                        anchors, losses, predictions)

    if (iteration.count % iteration.steps_per_log) == 0:
        iteration.report_stats()

    utils.fetch_reports(args, session=session, execution=True)

    # The following will only be true if:
    #   Learning rate mode is STEP and the current total step counter is in the schedule
    #   Learning rate mode is EPOCH and the current epoch has just changed to one in the schedule
    if optimizer_factory.should_update(iteration):
        optimizer = optimizer_factory.update_and_create(iteration)
        session.updateOptimizerFromHost(optimizer)

    iteration.count += 1


def get_timing_start_anchor(start_times):
    # Return the ID of the first input that is sent from the host.
    # Order is repeateable so we can just check the time for one entry:
    return min(start_times, key=lambda k: start_times[k][-1])


def get_timing_end_anchor(end_times):
    # Return the ID of the last anchor that is returned to the host.
    # Order is repeateable so we can just check the time for one entry:
    return max(end_times, key=lambda k: end_times[k][-1])


def create_callback_stepio(data, anchors, start_times, end_times, batches_per_step):

    micro_batch_indices = defaultdict(int)

    # Input callback is called when the data is needed:
    def input_callback(id, is_prefetch: bool):
        if is_prefetch:
            input_time = time.perf_counter()
            start_times[id].append(input_time)

        return data[id][micro_batch_indices[id]]

    # Called after the input buffer has been consumed by the device:
    def input_complete_callback(id):
        micro_batch_indices[id] = \
            (micro_batch_indices[id] + 1) % batches_per_step
        return

    # Output callback is called when a buffer is needed for the result:
    def output_callback(id):
        return anchors[id][micro_batch_indices[id]]

    # Complete callback is called when the output buffer has
    # been filled (result is ready to be consumed by the host):
    def output_complete_callback(id):
        output_time = time.perf_counter()
        end_times[id].append(output_time)
        micro_batch_indices[id] = \
            (micro_batch_indices[id] + 1) % batches_per_step

    stepio = popart.PyStepIOCallback(input_callback,
                                     input_complete_callback,
                                     output_callback,
                                     output_complete_callback)
    return stepio


def compute_latency(args, start_times, end_times, durations):
    if start_times and args.low_latency_inference and args.task == "SQUAD":
        # Compute latency stats using time between the
        # two anchors most separated in time:
        start_id = get_timing_start_anchor(start_times)
        end_id = get_timing_end_anchor(end_times)
        rtts = list(
            map(lambda v: v[1] - v[0], zip(start_times[start_id], end_times[end_id])))
        if len(rtts) != args.batches_per_step:
            raise RuntimeError(
                "Number of timings doesn't match items in the batch. Something is wrong.")
        mean_latency = (sum(rtts)) / args.batches_per_step
        min_latency = min(rtts)
        max_latency = max(rtts)
        if (logging.getLogger().isEnabledFor(logging.DEBUG)):
            for i, v in enumerate(rtts):
                logging.debug(f"LATENCY: {i} {v}")
    else:
        mean_latency = np.average(durations)
        min_latency = min(durations)
        max_latency = max(durations)
    return mean_latency, min_latency, max_latency


def bert_process_infer_data(args, session, data, anchors,
                            logits, iteration: Iteration,
                            start_times, end_times, stepio,
                            labels=None, predictions=None, losses=None):
    if stepio is None:
        stepio = popart.PyStepIO(data, anchors)

    start = time.perf_counter()
    session.run(stepio)
    duration = time.perf_counter() - start
    hw_cycles = session.getCycleCount() if args.report_hw_cycle_count else None

    labels_data = None
    if labels is not None:
        labels_data = [data[label] for label in labels]

        if not np.any([np.any(label) for label in labels_data]):
            labels_data = None

    iteration.add_inference_stats(duration, hw_cycles, labels_data, anchors, predictions, losses)

    mean_latency, min_latency, max_latency = compute_latency(
        args, start_times, end_times, iteration.durations)

    if (iteration.count % iteration.steps_per_log) == 0:
        iteration.report_inference_stats(mean_latency, min_latency, max_latency, hw_cycles)

    utils.fetch_reports(args, session=session, execution=True)

    iteration.count += 1

    if args.task == "PRETRAINING":
        return None

    return [anchors[logit] for logit in logits]


def bert_train_loop(args, session, writer,
                    dataset, labels, predictions, losses, anchors,
                    iteration, optimizer_factory):
    losses = [loss for loss in losses]

    save_model_and_stats(args, session, writer,
                         iteration.count, iteration.epoch)

    for iteration.epoch in range(iteration.start_epoch, args.epochs):
        for data in dataset:
            bert_process_data(args, session, labels, data, anchors,
                              losses, predictions, iteration, optimizer_factory)

            if args.steps_per_save > 0 and (iteration.count % args.steps_per_save) == 0:
                save_model_and_stats(args, session, writer,
                                     iteration.count, iteration.epoch, True)

        if args.epochs_per_save > 0 and ((iteration.epoch + 1) % iteration.epochs_per_save) == 0:
            save_model_and_stats(args, session, writer,
                                 iteration.count, iteration.epoch + 1)

    save_model_and_stats(args, session, writer, iteration.count)


def enable_realtime_scheduling(args):
    if args.realtime_scheduler:
        # Use a system call to enable real-time scheduling
        # for the whole process:
        pid = os.getpid()
        logger.info(f"Enabling real-time scheduler for process: PID {pid}")
        os.system(f"sudo -n chrt --rr -p 99 {pid}")


def disable_realtime_scheduling(args):
    if args.realtime_scheduler:
        # Use a system call to reset to default scheduling
        # for the whole process:
        pid = os.getpid()
        logger.info(f"Disabling real-time scheduler for process: PID {pid}")
        os.system(f"sudo -n chrt --other -p 0 {pid}")


def bert_infer_loop(args, session,
                    dataset, inputs, logits, anchors,
                    labels, predictions, losses,
                    iteration):
    save_results = args.task == "SQUAD" and not args.synthetic_data

    if not losses:
        losses = None
    else:
        losses = [loss.output(0) for loss in losses]

    # Create the stepio once outside of the inference loop:
    static_data = {}
    start_times = defaultdict(list)
    end_times = defaultdict(list)
    if args.low_latency_inference and args.task == "SQUAD":
        stepio = create_callback_stepio(static_data, anchors, start_times, end_times,
                                        dataset.batches_per_step)
    else:
        stepio = None

    enable_realtime_scheduling(args)

    for iteration.epoch in range(args.epochs_inference):
        for data in dataset:
            static_data.update({t: data[t] for t in inputs})
            result = bert_process_infer_data(args, session, static_data, anchors,
                                             logits, iteration,
                                             start_times, end_times, stepio,
                                             labels, predictions, losses)

            if result is not None and save_results and iteration.epoch == args.epochs_inference - 1:
                dataset.add_results(data, result)
            start_times.clear()
            end_times.clear()

    disable_realtime_scheduling(args)

    # If SQuAD save the predictions and run the evaulation script
    if save_results:
        dataset.write_predictions()


def acquire_device(args, request_ipus):
    if args.use_ipu_model:
        model_opts = {"numIPUs": request_ipus}
        if args.ipu_model_version is not None:
            model_opts["ipuVersion"] = args.ipu_model_version
        device = popart.DeviceManager().createIpuModelDevice(model_opts)
    else:
        if args.execution_mode == "PINGPONG":
            sync_pattern = popart.SyncPattern.PingPong
        elif args.execution_mode == "PIPELINE" and args.replication_factor <= 1:
            sync_pattern = popart.SyncPattern.SinglePipeline
        else:
            sync_pattern = popart.SyncPattern.Full
        device = popart.DeviceManager().acquireAvailableDevice(
            request_ipus,
            pattern=sync_pattern)
    if device is None:
        raise OSError("Failed to acquire IPU.")
    logger.info(f"Acquired device: {device}")
    return device


def bert_pretrained_initialisers(config, args):
    if args.synthetic_data:
        logger.info("Initialising from synthetic_data")
        return None
    if args.onnx_checkpoint:
        logger.info(
            f"Initialising from ONNX checkpoint: {args.onnx_checkpoint}")
        return utils.load_initializers_from_onnx(args.onnx_checkpoint)
    if args.tf_checkpoint:
        logger.info(f"Initialising from TF checkpoint: {args.tf_checkpoint}")
        return load_initializers_from_tf(args.tf_checkpoint, True, config, args.task)
    return None


def main(args):
    set_library_seeds(args.seed)

    config = bert_config_from_args(args)

    initializers = bert_pretrained_initialisers(config, args)

    logger.info("Building Model")
    # Specifying ai.onnx opset9 for the slice syntax
    model = Bert(config,
                 builder=popart.Builder(
                     opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}),
                 initializers=initializers,
                 execution_mode=args.execution_mode)

    # If config.host_embedding is enabled, indices and positions will have the matrices instead of the index vector.
    indices, positions, segments, masks, labels = bert_add_inputs(args, model)
    logits = bert_logits_graph(model, indices, positions, segments, masks)

    if args.inference:

        predictions = None
        losses = []
        if args.task == "PRETRAINING":
            # If this is a pretraining session, labels for NSP and MLM are already within the dataset,
            # so we can always calculate prediction performance
            predictions, _ = bert_infer_graph(model, logits, include_probs=False)

            if args.inference_lm_perplexity:
                losses = bert_perplexity_graph(model, logits, labels)

            outputs = bert_add_validation_outputs(model, predictions, losses)
        else:
            if args.inference_lm_perplexity:
                raise RuntimeError("Masked LM perplexity is only supported in pretraining.")

            outputs = bert_add_logit_outputs(model, logits)

        writer = None
    else:
        predictions, probs = bert_infer_graph(model, logits)
        losses = bert_loss_graph(model, probs, labels)
        outputs = bert_add_validation_outputs(model, predictions, losses)
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
                        labels, predictions, losses, iteration)
        device.detach()
    else:
        if not args.no_training:
            optimizer_factory = ScheduledOptimizerFactory(args,
                                                          iteration,
                                                          model.tensors)

            session, anchors = bert_training_session(model,
                                                     args,
                                                     data_flow,
                                                     losses,
                                                     device,
                                                     optimizer_factory)
            logger.info("Training Started")
            bert_train_loop(args, session, writer,
                            dataset, labels, predictions, losses, anchors,
                            iteration, optimizer_factory)

            device.detach()
            logger.info("Training Finished")

    return session, iteration


def setup_logger(log_level):
    # Define a root config with a format which is simpler for console use
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    # Define a specific Handler for this file that removes the root name.
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False


if __name__ == "__main__":

    args = utils.parse_bert_args()

    setup_logger(logging.getLevelName(args.log_level))

    logger.info("Program Start")
    logger.info("Hostname: " + socket.gethostname())
    logger.info("Command Executed: " + str(sys.argv))

    # Run the main inference/training session by default
    if args.inference or not args.no_training:
        main(args)

    # If this was a training session and validation isn't disabled; validate.
    if not args.inference and not args.no_validation and not args.no_model_save:
        logger.info("Doing Validation")
        main(utils.get_validation_args(args))

    logger.info("Program Finished")
