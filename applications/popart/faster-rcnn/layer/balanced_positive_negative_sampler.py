# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# This file has been modified by Graphcore Ltd.
import numpy as np
from config import cfg
from models.base_model import BaseModel
from IPU.ipu_tensor import gcop
"""Class to subsample minibatches by balancing positives and negatives.
Subsamples minibatches based on a pre-specified positive fraction in range
[0,1]. The class presumes there are many more negatives than positive examples:
if the desired batch_size cannot be achieved with the pre-specified positive
fraction, it fills the rest with negative examples. If this is not sufficient
for obtaining the desired batch_size, it returns fewer examples.
The main function to call is Subsample(self, indicator, labels). For convenience
one can also call SubsampleWeights(self, weights, labels) which is defined in
the minibatch_sampler base class.
When is_static is True, it implements a method that guarantees static shapes.
It also ensures the length of output of the subsample is always batch_size, even
when number of examples set to True in indicator is less than batch_size.
This is originally implemented in TensorFlow Object Detection API.
"""


class BalancedPositiveNegativeSampler(BaseModel):
    """Subsamples minibatches to a desired balance of positives and negatives."""

    def __init__(self, fp16_on=False, training=True, positive_fraction=0.5):
        """Constructs a minibatch sampler.
    Args:
      positive_fraction: desired fraction of positive examples (scalar in [0,1])
        in the batch.
      is_static: If True, uses an implementation with static shape guarantees.
    Raises:
      ValueError: if positive_fraction < 0, or positive_fraction > 1
    """
        super().__init__(fp16_on=fp16_on, training=training)
        if positive_fraction < 0 or positive_fraction > 1:
            raise ValueError('positive_fraction should be in range [0,1]. '
                             'Received: %s.' % positive_fraction)
        self._positive_fraction = positive_fraction
        self._is_static = True

    def _get_num_pos_neg_samples(self, sorted_indices_tensor,
                                 sample_size):
        """Counts the number of positives and negatives numbers to be sampled.
    Args:
      sorted_indices_tensor: A sorted int32 tensor of shape [N] which contains
        the signed indices of the examples where the sign is based on the label
        value. The examples that cannot be sampled are set to 0. It samples
        atmost sample_size*positive_fraction positive examples and remaining
        from negative examples.
      sample_size: Size of subsamples.
    Returns:
      A tuple containing the number of positive and negative labels in the
      subsample.
    """
        input_length = sorted_indices_tensor.shape.as_list()[0]
        valid_positive_index = gcop.greater(
            sorted_indices_tensor, gcop.zeros(input_length, gcop.int32))
        valid_negative_index = gcop.less(sorted_indices_tensor,
                                         gcop.zeros(input_length, gcop.int32))

        # check if negative samples less than (1-self._positive_fraction)
        num_sampled_neg = gcop.reduce_sum(
            gcop.cast(valid_negative_index, gcop.int32))
        negative_fraction = 1 - self._positive_fraction
        min_num_negative_samples = gcop.constant(
            np.array(sample_size * negative_fraction, np.int32))
        insufficient_negatives = gcop.maximum(
            min_num_negative_samples - num_sampled_neg,
            gcop.constant(np.array(0, np.int32)))

        num_sampled_pos = gcop.reduce_sum(
            gcop.cast(valid_positive_index, gcop.int32))
        max_num_positive_samples = gcop.constant(
            np.array(sample_size * self._positive_fraction, np.int32))
        num_positive_samples = gcop.minimum(
            max_num_positive_samples + insufficient_negatives, num_sampled_pos)
        num_negative_samples = gcop.constant(np.array(
            sample_size, np.int32)) - num_positive_samples

        return num_positive_samples, num_negative_samples

    def _get_values_from_start_and_end(self, input_tensor, num_start_samples,
                                       num_end_samples, total_num_samples):
        """slices num_start_samples and last num_end_samples from input_tensor.
    Args:
      input_tensor: An int32 tensor of shape [N] to be sliced.
      num_start_samples: Number of examples to be sliced from the beginning
        of the input tensor.
      num_end_samples: Number of examples to be sliced from the end of the
        input tensor.
      total_num_samples: Sum of is num_start_samples and num_end_samples. This
        should be a scalar.
    Returns:
      A tensor containing the first num_start_samples and last num_end_samples
      from input_tensor.
    """
        input_length = input_tensor.shape.as_list()[0]
        start_positions = gcop.less(gcop.range(input_length),
                                    num_start_samples)
        end_positions = gcop.math.greater_equal(gcop.range(input_length),
                                                input_length - num_end_samples)
        selected_positions = gcop.math.logical_or(start_positions,
                                                  end_positions)
        selected_positions = gcop.cast(selected_positions, 'FLOAT')
        indexed_positions = gcop.multiply(gcop.math.cumsum(selected_positions),
                                          selected_positions)

        if cfg.TRAIN.ROI_SAMPLER.one_hot_opti:
            one_hot_selector = gcop.one_hot(
                gcop.cast(indexed_positions, gcop.int32),
                total_num_samples + 1)
            one_hot_selector = one_hot_selector[:, 1:]
        else:
            one_hot_selector = gcop.one_hot(
                (gcop.cast(indexed_positions, gcop.int32) - 1).cast(gcop.int32),
                total_num_samples)

        result = gcop.matmul(
            input_tensor.cast(gcop.float32).unsqueeze(0),
            one_hot_selector.cast(gcop.float32))
        return result[0]

    def _static_subsample(self,
                          indicator,
                          batch_size,
                          positive_flags,
                          negative_flags,
                          boxes_keep_arr):
        """Returns subsampled minibatch.
    Args:
      indicator: boolean tensor of shape [N] whose True entries can be sampled.
        N should be a complie time constant.
      boxes_keep_arr: some box's area is zero, this boolen array: False for zero area box, True for non-zero area box
      batch_size: desired batch size. This scalar cannot be None.
      positive_flags: boolean tensor of shape [N] denoting positive(=True) and negative
        (=False) examples. N should be a complie time constant.
    Returns:
      sampled_idx_indicator: boolean tensor of shape [N], True for entries which
        are sampled. It ensures the length of output of the subsample is always
        batch_size, even when number of examples set to True in indicator is
        less than batch_size.
    Raises:
      ValueError: if labels and indicator are not 1D boolean tensors.
    """
        # Check if indicator and labels have a static size.
        if not isinstance(batch_size, int):
            raise ValueError(
                'batch_size has to be an integer when is_static is'
                'True.')
        input_length = indicator.shape.as_list()[0]

        # Set the number of examples set True in indicator to be at least
        # batch_size.
        num_true_sampled = gcop.reduce_sum(gcop.cast(indicator, gcop.float32))
        additional_false_sample = gcop.math.less_equal(
            gcop.math.cumsum(
                gcop.cast(gcop.math.logical_not(indicator), 'FLOAT')),
            batch_size - num_true_sampled)
        indicator = gcop.math.logical_or(indicator, additional_false_sample)

        # Shuffle indicator and label. Need to store the permutation to restore the
        # order post sampling.
        permutation = gcop.random.shuffle(gcop.range(input_length))

        indicator = self.matmul_gather_on_zeroth_axis(
            gcop.cast(indicator, gcop.float32), permutation)
        positive_flags = self.matmul_gather_on_zeroth_axis(
            gcop.cast(positive_flags, gcop.float32), permutation)
        boxes_keep_arr = self.matmul_gather_on_zeroth_axis(
            gcop.cast(boxes_keep_arr, gcop.float32),
            permutation).cast(gcop.int32)

        # index (starting from 1) when indicator is True, 0 when False
        indicator_idx = gcop.where(
            gcop.cast(indicator, gcop.bool),
            gcop.range(2, input_length + 2).cast(gcop.int32),
            gcop.ones(input_length, gcop.int32).cast(gcop.int32))

        # Replace -1 for negative, +1 for positive labels
        signed_label = gcop.where(gcop.cast(positive_flags, gcop.bool),
                                  gcop.ones(input_length, gcop.int32),
                                  -1 * gcop.ones(input_length, gcop.int32))
        # negative of index for negative label, positive index for positive label,
        # 0 when indicator is False.
        signed_indicator_idx = gcop.multiply(indicator_idx, signed_label)
        signed_indicator_idx_filter_zeroAreas = gcop.multiply(
            signed_indicator_idx, boxes_keep_arr)
        sorted_signed_indicator_idx, _ = gcop.nn.top_k(
            signed_indicator_idx_filter_zeroAreas, input_length, sorted=True
        )  # TODO,return values, order, I guess it values represents the sorted of signed_indicator_idx

        [num_positive_samples,
         num_negative_samples] = self._get_num_pos_neg_samples(
             sorted_signed_indicator_idx, batch_size)

        sampled_idx = self._get_values_from_start_and_end(
            sorted_signed_indicator_idx, num_positive_samples,
            num_negative_samples, batch_size)

        # Shift the indices to start from 0 and remove any samples that are set as
        # False.
        sampled_idx = gcop.abs(
            sampled_idx) - 2
        sampled_idx = gcop.multiply(
            gcop.cast(
                gcop.math.greater_equal(sampled_idx,
                                        gcop.constant(np.array(0,
                                                               np.float32))),
                gcop.float32), sampled_idx)

        sampled_idx_indicator = gcop.cast(
            gcop.reduce_sum(gcop.one_hot(sampled_idx.cast(gcop.int32),
                                         depth=input_length),
                            axis=[0]), gcop.bool)

        # project back the order based on stored permutations
        reprojections = gcop.one_hot(permutation.cast(gcop.int32),
                                     depth=input_length)
        result = gcop.matmul(
            sampled_idx_indicator.cast(gcop.float32).unsqueeze(0),
            reprojections.cast(gcop.float32))
        return result[0]

    def subsample(self,
                  indicator,
                  batch_size,
                  positive_flags,
                  negative_flags,
                  boxes_keep_arr):
        """Returns subsampled minibatch.
    Args:
      indicator: boolean tensor of shape [N] whose True entries can be sampled.
      boxes_keep_arr: some box's area is zero, this boolen array: False for zero area box, True for non-zero area box
      batch_size: desired batch size. If None, keeps all positive samples and
        randomly selects negative samples so that the positive sample fraction
        matches self._positive_fraction. It cannot be None is is_static is True.
      positive_flags: boolean tensor of shape [N] denoting positive(=True) and negative and unsampled (=False) examples.
      negative_flags: boolean tensor of shape [N] denoting negative(=True) and positive and unsampled (=False) examples.
      scope: name scope.
    Returns:
      sampled_idx_indicator: boolean tensor of shape [N], True for entries which
        are sampled.
    Raises:
      ValueError: if labels and indicator are not 1D boolean tensors.
    """
        if len(indicator.shape) != 1:
            raise ValueError(
                'indicator must be 1 dimensional, got a tensor of '
                'shape %s' % indicator.shape)
        if len(positive_flags.shape) != 1:
            raise ValueError(
                'positive_flags must be 1 dimensional, got a tensor of '
                'shape %s' % positive_flags.shape)
        if len(negative_flags.shape) != 1:
            raise ValueError(
                'negative_flags must be 1 dimensional, got a tensor of '
                'shape %s' % negative_flags.shape)
        if positive_flags.dtype != gcop.bool:
            raise ValueError(
                'positive_flags should be of type bool. Received: %s' %
                positive_flags.dtype)
        if negative_flags.dtype != gcop.bool:
            raise ValueError(
                'negative_flags should be of type bool. Received: %s' %
                negative_flags.dtype)
        if indicator.dtype != gcop.bool:
            raise ValueError('indicator should be of type bool. Received: %s' %
                             indicator.dtype)
        with gcop.variable_scope('BalancedPositiveNegativeSampler'):
            if self._is_static:
                return self._static_subsample(indicator, batch_size,
                                              positive_flags,
                                              negative_flags,
                                              boxes_keep_arr)

            else:
                raise RuntimeError('only static sampler can be use')

    def matmul_gather_on_zeroth_axis(self, params, indices, scope=None):
        """Matrix multiplication based implementation of self.gather on zeroth axis.
      TODO(rathodv, jonathanhuang): enable sparse matmul option.
      Args:
          params: A float32 Tensor. The tensor from which to gather values.
          Must be at least rank 1.
          indices: A Tensor. Must be one of the following types: int32, int64.
          Must be in range [0, params.shape[0])
          scope: A name for the operation (optional).
      Returns:
          A Tensor. Has the same type as params. Values from params gathered
          from indices given by indices, with shape indices.shape + params.shape[1:].
      """
        with gcop.variable_scope('MatMulGather'):
            params_shape = self.combined_static_and_dynamic_shape(params)
            indices_shape = self.combined_static_and_dynamic_shape(indices)
            params2d = gcop.reshape(params, [params_shape[0], -1])
            indicator_matrix = gcop.one_hot(indices.cast(gcop.int32),
                                            params_shape[0])
            gathered_result_flattened = gcop.matmul(
                indicator_matrix.cast(gcop.float32),
                params2d.cast(gcop.float32))
            return gcop.reshape(gathered_result_flattened,
                                indices_shape + params_shape[1:])

    def combined_static_and_dynamic_shape(self, tensor):
        """Returns a list containing static and dynamic values for the dimensions.
      Returns a list of static and dynamic values for shape dimensions. This is
      useful to preserve static shapes when available in reshape operation.
      Args:
          tensor: A tensor of any type.
      Returns:
          A list of size tensor.shape.ndims containing integers or a scalar tensor.
      """
        static_tensor_shape = tensor.shape.as_list()
        dynamic_tensor_shape = tensor.shape.as_list()
        combined_shape = []
        for index, dim in enumerate(static_tensor_shape):
            if dim is not None:
                combined_shape.append(dim)
            else:
                combined_shape.append(dynamic_tensor_shape[index])
        return combined_shape
