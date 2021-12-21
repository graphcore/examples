// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once
#include <poplar/Graph.hpp>

void compactProbs(poplar::Graph &graph, poplar::program::Sequence &program,
                  const poplar::Tensor &log_probs,
                  poplar::Tensor &compacted_probs, const poplar::Tensor &labels,
                  const poplar::Tensor &label_lengths);
poplar::Tensor expandGradients(poplar::Graph &graph,
                               poplar::program::Sequence &program,
                               const poplar::Tensor &compacted_gradients,
                               const poplar::Tensor &log_probs,
                               const poplar::Tensor &labels,
                               const poplar::Tensor &label_lengths);
void alpha(poplar::Graph &graph, poplar::program::Sequence &program,
           const poplar::Tensor &log_probs, const poplar::Tensor &input_lengths,
           const poplar::Tensor &label_lengths, poplar::Tensor &log_alpha);
poplar::Tensor losses(poplar::Graph &graph, poplar::program::Sequence &program,
                      const poplar::Tensor &log_beta);
void maplosses(poplar::Graph &graph, poplar::program::Sequence &program,
               const poplar::Tensor &log_probs, const poplar::Tensor &log_alpha,
               const poplar::Tensor &input_lengths,
               const poplar::Tensor &label_lengths, poplar::Tensor &losses);
void beta(poplar::Graph &graph, poplar::program::Sequence &program,
          const poplar::Tensor &log_probs, const poplar::Tensor &input_lengths,
          const poplar::Tensor &label_lengths, poplar::Tensor &log_beta);
void grads(poplar::Graph &graph, poplar::program::Sequence &program,
           const poplar::Tensor &log_probs, const poplar::Tensor &input_lengths,
           const poplar::Tensor &label_lengths, const poplar::Tensor &alphas,
           const poplar::Tensor &betas, const poplar::Tensor &losses,
           poplar::Tensor &grads);
void alpha_beta(poplar::Graph &graph, poplar::program::Sequence &program,
                const poplar::Tensor &log_probs,
                const poplar::Tensor &input_lengths,
                const poplar::Tensor &label_lengths, poplar::Tensor &log_alpha,
                poplar::Tensor &log_beta);
