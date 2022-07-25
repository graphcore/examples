// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poplar/Tensor.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

poplar::Tensor compactLogProbs(poplar::Graph &graph, const poplar::Tensor &logProbs, const poplar::Tensor &labels, const poplar::Tensor &labelLengths, poplar::program::Sequence &program);
poplar::Tensor shiftLogProbs(poplar::Graph &graph, const poplar::Tensor &logProbs, poplar::program::Sequence &prog, bool shiftAlsoInU);
poplar::Tensor computeAlpha(poplar::Graph &graph, const poplar::Tensor &logProbs, const poplar::Tensor &inputLengths, poplar::program::Sequence &prog);
poplar::Tensor computeBeta(poplar::Graph &graph, const poplar::Tensor &logProbs, const poplar::Tensor &inputLengths, const poplar::Tensor &labelLengths, poplar::program::Sequence &prog);
poplar::Tensor computeLoss(poplar::Graph &graph, const poplar::Tensor &logBeta, poplar::program::Sequence &prog);
poplar::Tensor computeGrads(poplar::Graph &graph, const poplar::Tensor &logProbs,
                            const poplar::Tensor &logAlpha, const poplar::Tensor &logBeta, const poplar::Tensor &logLoss,
                            const poplar::Tensor &inputLengths, const poplar::Tensor &labelLengths,
                            poplar::program::Sequence &program);
void expandAndSub(poplar::Graph &graph, const poplar::Tensor &compactedT, poplar::Tensor subT, const poplar::Tensor &labels, const poplar::Tensor &labelLengths, poplar::program::Sequence &program);
