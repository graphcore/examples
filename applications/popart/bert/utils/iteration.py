# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import math
import logging
from functools import partial
from collections import deque
import numpy as np

from .distributed import average_distributed_deques

logger = logging.getLogger('BERT')


def reduce_metric(args, anchors, metrics, mean=False):
    accumulated_stats = args.gradient_accumulation_factor * args.batches_per_step
    if len(metrics) > 1:
        metric = np.add(*[anchors[metric] for metric in metrics])
        if mean:
            accumulated_stats *= len(metrics)
    else:
        metric = anchors[metrics[0]]
    return np.mean(metric / accumulated_stats)


def output_stats(args, anchors, losses, accuracies):
    return (reduce_metric(args, anchors, losses),
            reduce_metric(args, anchors, accuracies, mean=True))


class Iteration:
    def __init__(self, args, steps_per_epoch, writer, recording_steps=None):
        self.epoch = args.continue_training_from_epoch
        self.count = self.epoch * steps_per_epoch
        self.micro_batch_size = args.micro_batch_size
        self.batches_per_step = args.batches_per_step
        self.gradient_accumulation_factor = args.gradient_accumulation_factor
        self.replication_factor = args.replication_factor
        self.training = not args.inference
        self.epochs = args.epochs if self.training else args.epochs_inference
        self.training_steps = args.training_steps
        if self.epochs is None:
            if args.training_steps is None:
                RuntimeError("Either epochs or training_steps need to be specified.")
            self.epochs = math.ceil(args.training_steps / steps_per_epoch)
            self.total_steps = args.training_steps
        else:
            self.total_steps = steps_per_epoch * self.epochs
        self.epochs_per_save = args.epochs_per_save
        self.steps_per_log = args.steps_per_log
        self.steps_per_epoch = steps_per_epoch
        self.recording_steps = self.steps_per_epoch if recording_steps is None else recording_steps
        self.writer = writer
        self.use_packed_sequence_format = args.use_packed_sequence_format
        self.use_popdist = args.use_popdist
        self.popdist_size = args.popdist_size

        # This should get overridden but will ensure we can always write a scalar to TB.
        self.learning_rate = 0
        self.total_sequences_so_far = 0
        self.sequences_per_step = deque(maxlen=self.recording_steps)
        self.durations = deque(maxlen=self.recording_steps)
        self.cycles = deque(maxlen=self.recording_steps)
        self.losses = deque(maxlen=self.recording_steps)
        self.accuracies = deque(maxlen=self.recording_steps)
        self.packing_ratio = deque(maxlen=self.recording_steps)
        self.stats_fn = output_stats

        if args.use_popdist:
            self.distributed = True
            self.steps_per_distributed_reduce = 1
        else:
            self.distributed = False

    @property
    def throughput(self):
        return np.divide(self.sequences_per_step, self.durations)

    def add_scalar(self, name, scalar):
        if self.writer is not None:
            if self.use_packed_sequence_format:
                self.writer.add_scalar(name, scalar, self.total_sequences_so_far)
            else:
                self.writer.add_scalar(name, scalar, self.count)

    def add_stats(self, duration, hw_cycles, data, *args):
        self.durations.append(duration)

        if self.use_packed_sequence_format:
            # To count the number of samples in each batch first
            # expand the micro-batch dimension (flattened on device)
            input_mask = data["input_mask"]
            mlm_weights = data["masked_lm_weights"]
            new_shape = list(input_mask.shape[:-1]) + [self.micro_batch_size, -1]
            input_mask = input_mask.reshape(new_shape)
            sequences_in_step = int(input_mask.max(-1).sum())
            tokens_in_step = int((mlm_weights > 0).sum())
            args = tokens_in_step, sequences_in_step, *args
        else:
            sequences_in_step = self.batches_per_step * self.gradient_accumulation_factor * \
                                self.replication_factor * self.micro_batch_size  # noqa
        if self.use_popdist:
            sequences_in_step = sequences_in_step * self.popdist_size
        self.total_sequences_so_far += sequences_in_step
        self.sequences_per_step.append(sequences_in_step)

        if hw_cycles:
            self.cycles.append(hw_cycles)
        if self.training or self.use_packed_sequence_format:
            self.add_training_stats(*args)

            if self.distributed and (self.count % self.steps_per_distributed_reduce) == 0:
                self.average_distributed_stats()

            self.add_scalar("defaultLearningRate", self.learning_rate)
            self.add_scalar("throughput", np.average(self.throughput))

            if self.use_packed_sequence_format:
                self.add_scalar("update_steps", self.count)
                self.packing_ratio.append(data['nsp_weights'].sum()/len(data['nsp_weights']))
                self.add_scalar("packing_ratio", np.average(self.packing_ratio))
            self.write_training_stats()
        else:
            self.add_inference_stats(*args)

    def add_training_stats(self, *args):
        loss, accuracy = self.stats_fn(*args)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def write_training_stats(self):
        self.add_scalar("loss", np.average(self.losses))
        self.add_scalar("accuracy", np.average(self.accuracies))

    def add_inference_stats(self, *args):
        pass

    def epoch_string(self):
        if self.training_steps is not None:
            status_string = f"Iteration: {self.count:5}/{int(self.training_steps)} "
        else:
            status_string = \
                f"Iteration: {self.count:6} " \
                f"Epoch: {self.count/self.steps_per_epoch:6.2f}/{self.epochs} "
        if self.use_packed_sequence_format:
            status_string +=  \
                f"Sequences processed: {self.total_sequences_so_far/1000.0:6.1f}k "
        return status_string

    def training_metrics_string(self):
        avg = np.average
        status_string = \
            f"Loss: {avg(self.losses):5.3f} " \
            f"Accuracy: {avg(self.accuracies):5.3f} "
        return status_string

    def optimizer_string(self):
        return f"Learning Rate: {self.learning_rate:.5f} "

    def throughput_string(self):
        avg = np.average
        status_string = \
            f"Duration: {avg(self.durations):6.4f} s " \
            f"Throughput: {avg(self.throughput):6.1f} sequences/s "
        if self.cycles:
            status_string += f"Cycles: {int(avg(self.cycles))} "
        return status_string

    def average_distributed_stats(self):
        replica_avg = partial(average_distributed_deques, N=self.steps_per_distributed_reduce)
        self.durations = replica_avg(self.durations)
        if self.cycles:
            self.cycles = replica_avg(self.cycles)
        self.losses = replica_avg(self.losses)
        self.accuracies = replica_avg(self.accuracies)

    def report_stats(self):
        status_string = self.epoch_string()
        status_string += self.training_metrics_string()
        status_string += self.optimizer_string()
        status_string += self.throughput_string()
        logger.info(status_string)

    def inference_metrics_string(self):
        return ""

    def report_inference_stats(self, mean_latency, min_latency, max_latency, p99_latency, p999_latency):
        avg = np.average
        status_string = f"Iteration: {self.count:6} "
        status_string += self.inference_metrics_string()
        status_string += self.throughput_string()
        if mean_latency is not None:
            status_string += f"Per-sample: Mean Latency={mean_latency} Min Latency={min_latency} Max Latency={max_latency} p99 Latency={p99_latency} p999 Latency={p999_latency} seconds"
        logger.info(status_string)


def pretraining_stats(args, anchors, losses, accuracies):
    losses = map(lambda loss: reduce_metric(args, anchors, [loss]), losses)
    accuracies = map(lambda acc: reduce_metric(args, anchors, [acc]), accuracies)
    return tuple(losses), tuple(accuracies)


def packed_pretraining_stats(tokens_in_step, sequences_in_step, args, anchors, losses, accuracies):
    """
    In packedBERT each step contains a different number of sequences and masked tokens,
    """
    mlm_loss = anchors[losses[0]].sum()/tokens_in_step
    mlm_accuracy = anchors[accuracies[0]].sum()/tokens_in_step

    nsp_loss = anchors[losses[1]].sum()/sequences_in_step
    nsp_accuracy = anchors[accuracies[1]].sum()/sequences_in_step
    return (mlm_loss, nsp_loss), (mlm_accuracy, nsp_accuracy)


def pretraining_inference_stats(args, anchors, losses, accuracies):
    if args.inference_lm_perplexity:
        loss = reduce_metric(args, anchors, [losses[0]])
    else:
        loss = None
    accuracies = map(lambda acc: reduce_metric(args, anchors, [acc]), accuracies)
    return loss, tuple(accuracies)


class PretrainingIteration(Iteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.calculate_perplexity = args[0].inference_lm_perplexity
        self.losses = [deque(maxlen=self.recording_steps), deque(maxlen=self.recording_steps)]
        self.accuracies = [deque(maxlen=self.recording_steps), deque(maxlen=self.recording_steps)]

        if self.use_packed_sequence_format:
            self.stats_fn = packed_pretraining_stats
        else:
            if self.training:
                self.stats_fn = pretraining_stats
            else:
                self.stats_fn = pretraining_inference_stats

    def average_distributed_stats(self):
        replica_avg = partial(average_distributed_deques, N=self.steps_per_distributed_reduce)
        self.durations = replica_avg(self.durations)
        if self.cycles:
            self.cycles = replica_avg(self.cycles)
        self.losses = [replica_avg(self.losses[0]), replica_avg(self.losses[1])]
        self.accuracies = [replica_avg(self.accuracies[0]), replica_avg(self.accuracies[1])]

    def add_training_stats(self, *args):
        loss, accuracy = self.stats_fn(*args)
        self.losses[0].append(loss[0])
        self.losses[1].append(loss[1])
        self.accuracies[0].append(accuracy[0])
        self.accuracies[1].append(accuracy[1])

    def write_training_stats(self):
        self.add_scalar("loss/MLM", np.average(self.losses[0]))
        self.add_scalar("loss/NSP", np.average(self.losses[1]))
        self.add_scalar("accuracy/MLM", np.average(self.accuracies[0]))
        self.add_scalar("accuracy/NSP", np.average(self.accuracies[1]))

    def training_metrics_string(self):
        avg = np.average
        status_string = \
            f"Loss (MLM NSP): {avg(self.losses[0]):5.3f} {avg(self.losses[1]):5.3f} " \
            f"Accuracy (MLM NSP): {avg(self.accuracies[0]):5.3f} {avg(self.accuracies[1]):5.3f} "
        return status_string

    def add_inference_stats(self, *args):
        loss, accuracy = self.stats_fn(*args)
        self.accuracies[0].append(accuracy[0])
        self.accuracies[1].append(accuracy[1])

        if loss is not None:
            self.losses[0].append(loss)

    def inference_metrics_string(self):
        avg = np.average
        status_string = \
            f"Accuracy (MLM NSP): {avg(self.accuracies[0]):5.3f} {avg(self.accuracies[1]):5.3f} "
        if self.calculate_perplexity:
            status_string += \
                f"LM Perplexity: {np.exp(avg(self.losses[0])):5.3f} "
        return status_string
