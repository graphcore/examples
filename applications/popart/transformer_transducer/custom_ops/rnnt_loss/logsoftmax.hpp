// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

poplar::Tensor logSoftmaxRnnt(poplar::Graph &graph, const poplar::Tensor &probs,
                              const poplar::Tensor &labels,
                              const poplar::Tensor &labelLengths,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext = {});

poplar::Tensor logSoftmaxRnntGrad(
    poplar::Graph &graph, poplar::Tensor &probs,
    const poplar::Tensor &outerGrads, const poplar::Tensor &labels,
    const poplar::Tensor &labelLengths, poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {});
