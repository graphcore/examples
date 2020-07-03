# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Seq2seq layer operations for use in neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import Decoder


__all__ = ["Decoder", "dynamic_decode"]

_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access
_zero_state_tensors = (
    rnn_cell_impl._zero_state_tensors
)  # pylint: disable=protected-access



def _create_zero_outputs(size, dtype, batch_size):
    """Create a zero outputs Tensor structure."""

    def _create(s, d):
        return _zero_state_tensors(s, batch_size, d)

    return nest.map_structure(_create, size, dtype)


def dynamic_decode(
    decoder,
    output_time_major=False,
    impute_finished=False,
    maximum_iterations=None,
    parallel_iterations=32,
    swap_memory=False,
    scope=None,
):
    """Perform dynamic decoding with `decoder`.

  Calls initialize() once and step() repeatedly on the Decoder object.

  Args:
    decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.

  Returns:
    `(final_outputs, final_state, final_sequence_lengths)`.

  Raises:
    TypeError: if `decoder` is not an instance of `Decoder`.
    ValueError: if `maximum_iterations` is provided but is not a scalar.
  """
    if not isinstance(decoder, Decoder):
        raise TypeError(
            "Expected decoder to be type Decoder, but saw: %s" % type(decoder)
        )

    with variable_scope.variable_scope(scope, "decoder") as varscope:
        # Determine context types.
        ctxt = (
            ops.get_default_graph()._get_control_flow_context()
        )  # pylint: disable=protected-access
        is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
        is_xla = True  # XLA detection does not work
        in_while_loop = control_flow_util.GetContainingWhileContext(ctxt) is not None
        # Properly cache variable values inside the while_loop.
        # Don't set a caching device when running in a loop, since it is possible
        # that train steps could be wrapped in a tf.while_loop. In that scenario
        # caching prevents forward computations in loop iterations from re-reading
        # the updated weights.
        if not context.executing_eagerly() and not in_while_loop:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = ops.convert_to_tensor(
                maximum_iterations, dtype=dtypes.int32, name="maximum_iterations"
            )
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state = decoder.initialize()

        zero_outputs = _create_zero_outputs(
            decoder.output_size, decoder.output_dtype, decoder.batch_size
        )

        if is_xla and maximum_iterations is None:
            raise ValueError("maximum_iterations is required for XLA compilation.")
        if maximum_iterations is not None:
            initial_finished = math_ops.logical_or(
                initial_finished, 0 >= maximum_iterations
            )
        initial_sequence_lengths = array_ops.zeros_like(
            initial_finished, dtype=dtypes.int32
        )
        initial_time = constant_op.constant(0, dtype=dtypes.int32)

        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0):
                return tensor_shape.TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(
                    ops.convert_to_tensor(batch_size, name="batch_size")
                )
                return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

        dynamic_size = maximum_iterations is None or not is_xla

        def _create_ta(s, d):
            return tensor_array_ops.TensorArray(
                dtype=d,
                size=0 if dynamic_size else maximum_iterations,
                dynamic_size=dynamic_size,
                element_shape=_shape(decoder.batch_size, s),
            )

        initial_outputs_ta = nest.map_structure(
            _create_ta, decoder.output_size, decoder.output_dtype
        )

        def condition(
            unused_time,
            unused_outputs_ta,
            unused_state,
            unused_inputs,
            finished,
            unused_sequence_lengths,
        ):
            # return math_ops.logical_not(math_ops.reduce_all(finished))                                       #Remove this cond
            return True

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths):
            """Internal while_loop body.

      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: bool tensor (keeping track of what's finished).
        sequence_lengths: int32 tensor (keeping track of time of finish).

      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
          next_sequence_lengths)`.
        ```
      """
            (next_outputs, decoder_state, next_inputs, decoder_finished) = decoder.step(
                time, inputs, state
            )
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
            else:
                next_finished = math_ops.logical_or(decoder_finished, finished)
            next_sequence_lengths = array_ops.where(
                math_ops.logical_not(finished),
                array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
                sequence_lengths,
            )

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(
                    lambda out, zero: array_ops.where(finished, zero, out),
                    next_outputs,
                    zero_outputs,
                )
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = new.shape.ndims == 0
                return new if pass_through else array_ops.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(_maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            outputs_ta = nest.map_structure(
                lambda ta, out: ta.write(time, out), outputs_ta, emit
            )
            return (
                time + 1,
                outputs_ta,
                next_state,
                next_inputs,
                next_finished,
                next_sequence_lengths,
            )

        res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=(
                initial_time,
                initial_outputs_ta,
                initial_state,
                initial_inputs,
                initial_finished,
                initial_sequence_lengths,
            ),
            parallel_iterations=parallel_iterations,
            maximum_iterations=maximum_iterations,
            swap_memory=swap_memory,
        )

        final_outputs_ta = res[1]
        final_state = res[2]
        final_sequence_lengths = res[5]

        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

        try:
            final_outputs, final_state = decoder.finalize(
                final_outputs, final_state, final_sequence_lengths
            )
        except NotImplementedError:
            pass

        if not output_time_major:
            final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

    return final_outputs, final_state, final_sequence_lengths
