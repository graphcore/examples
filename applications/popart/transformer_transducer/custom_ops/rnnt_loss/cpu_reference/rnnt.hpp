// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include "utils.hpp"

// for 1 batch -> <losses, alphas>
std::pair<std::vector<float>, NDArray<float>>
alpha(const std::vector<float> &log_probs, const Shape &shape,
      const std::vector<size_t> &input_lengths,
      const std::vector<int32_t> &labels,
      const std::vector<size_t> &label_lengths);
std::pair<std::vector<float>, NDArray<float>>
beta(const std::vector<float> &log_probs, const Shape &shape,
     const std::vector<size_t> &input_lengths,
     const std::vector<int32_t> &labels,
     const std::vector<size_t> &label_lengths);
NDArray<float> grads(const std::vector<float> &log_probs, const Shape &shape,
                     const std::vector<size_t> &input_lengths,
                     const std::vector<int32_t> &labels,
                     const std::vector<size_t> &label_lengths,
                     const NDArray<float> &alphas, const NDArray<float> &betas,
                     const std::vector<float> &losses, bool sparse = false);
std::pair<std::vector<float>, NDArray<float>>
alpha_static(const std::vector<float> &log_probs, const Shape &shape,
             const std::vector<size_t> &input_lengths,
             const std::vector<int32_t> &labels,
             const std::vector<size_t> &label_lengths);
std::pair<std::vector<float>, NDArray<float>>
beta_static(const std::vector<float> &log_probs, const Shape &shape,
            const std::vector<size_t> &input_lengths,
            const std::vector<int32_t> &labels,
            const std::vector<size_t> &label_lengths);
NDArray<float> grads_static(const std::vector<float> &log_probs,
                            const Shape &shape,
                            const std::vector<size_t> &input_lengths,
                            const std::vector<int32_t> &labels,
                            const std::vector<size_t> &label_lengths,
                            const NDArray<float> &alphas,
                            const NDArray<float> &betas,
                            const std::vector<float> &losses);

NDArray<float> compact_cpu(const std::vector<float> &t, const Shape &shape,
                           const std::vector<int> &labels,
                           const std::vector<int> &labelLengths);

NDArray<float> expand_cpu(const std::vector<float> &compactGrads,
                          const Shape &shape, uint32_t alphabet,
                          const std::vector<int32_t> &labels,
                          const std::vector<int32_t> &labelLengths);
