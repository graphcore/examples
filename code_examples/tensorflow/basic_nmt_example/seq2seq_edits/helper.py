# Copyright 2019 Graphcore Ltd.
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


class GreedyEmbeddingHelperNoCond(tf.contrib.seq2seq.GreedyEmbeddingHelper):
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        # Removed a cond that stopped cycle counts due to dynamic control flow
        next_inputs = self._embedding_fn(sample_ids)
        return (finished, next_inputs, state)


class TrainingHelperNoCond(tf.contrib.seq2seq.TrainingHelper):
    def initialize(self, name=None):
        with ops.name_scope(name, "TrainingHelperInitialize"):
            finished = math_ops.equal(0, self._sequence_length)
            # Removed a cond that a) stopping cycle counts due to dynamic control flow
            # b) causes a switch depth bug in xla.
            next_inputs = nest.map_structure(lambda inp: inp.read(0), self._input_tas)
            return (finished, next_inputs)

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """next_inputs_fn for TrainingHelper."""
        with ops.name_scope(name, "TrainingHelperNextInputs", [time, outputs, state]):
            next_time = time + 1
            finished = next_time >= self._sequence_length

            def read_from_ta(inp):
                return inp.read(next_time)

            # Removed a cond that stopped cycle counts due to dynamic control flow
            next_inputs = nest.map_structure(read_from_ta, self._input_tas)
            return (finished, next_inputs, state)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with ops.name_scope(name, "TrainingHelperSample", [time, outputs]):
            sample_ids = math_ops.argmax(outputs, axis=-1, output_type=tf.int32)
        return sample_ids
