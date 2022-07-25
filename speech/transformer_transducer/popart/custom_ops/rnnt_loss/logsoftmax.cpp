// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "logsoftmax.hpp"

#include "popnn/NonLinearity.hpp"
#include "popops/ElementWise.hpp"
#include "popops/ElementWiseUtil.hpp"
#include "popops/Reduce.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include "ipu_utils.hpp"
#include "rnnt_utils.hpp"
#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;

void mapVertex(poplar::Graph &graph, const poplar::Tensor &max,
               const poplar::Tensor &probs, poplar::Tensor &out,
               poplar::ComputeSet &computeSet) {
  auto outF = out.flatten();
  const auto probsF = probs.flatten();
  const auto maxF = max.flatten();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getTilesPerIPU();
  const auto mappingProbs = graph.getTileMapping(probs);
  const auto mappingOut = graph.getTileMapping(out);
  const size_t alphabet = probs.dim(3);
  const std::string vertexName =
      poputil::templateVertex("SoftmaxMapVertex", probs.elementType());
  for (size_t i = 0; i < numTiles; ++i) {
    auto &regions = mappingOut[i];
    if (regions.size() > 0) {
      auto vertexRegions =
          poputil::splitRegionsBetweenWorkers(graph.getTarget(), regions, 1, 1);
      auto vertexRegionsProbs = poputil::splitRegionsBetweenWorkers(
          graph.getTarget(), mappingProbs[i], alphabet, 1);
      assert(vertexRegionsProbs.size() == vertexRegions.size());
      size_t j = 0;
      for (auto &r : vertexRegions) {
        auto &probsR = vertexRegionsProbs[j];
        assert(r.size() == probsR.size());
        poplar::VertexRef vtx = graph.addVertex(computeSet, vertexName);
        graph.connect(vtx["max"], poplar::concat(maxF.slices(r)));
        graph.connect(vtx["out"], poplar::concat(outF.slices(r)));
        graph.connect(vtx["probs"], poplar::concat(probsF.slices(probsR)));
        graph.setPerfEstimate(vtx, alphabet); // wrong ...
        graph.setTileMapping(vtx, i);
        graph.setInitialValue(vtx["alphabet"], alphabet);
        ++j;
      }
    }
  }
}

//
// j - softmax dimension, that is - dimension along which a sum(exp) is
// performed
// i - all other dimensions indicated as one i index for simplicity
//
// logsoftmax_j(i) = x_j(i) - max_k(x_k(i)) -
// log(sum_k(exp(x_k(i)-max_l(x_l(i)))))
//
Tensor logSoftmaxRnnt(Graph &graph, const Tensor &probs, const Tensor &labels,
                      const Tensor &labelLengths, Sequence &prog,
                      const DebugContext &debugContext) {
  const std::string fnStr = "LogSoftmaxRnnt";
  const auto dType = probs.elementType();
  const auto rank = probs.rank();
  assert(rank == 4);
  unsigned batch_size = probs.dim(0);
  unsigned T = probs.dim(1);
  unsigned U = probs.dim(2);
  unsigned alphabet = probs.dim(3);

  poputil::mapTensorLinearly(graph, probs, 1, alphabet);
  // Switch innermost dimension to outer as softmax is done over it
  Tensor probsShuf = probs.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(probsShuf.shape() ==
         std::vector<std::size_t>({alphabet, T, U, batch_size}));

  poplar::Tensor probsCompacted =
      graph.addVariable(dType, {batch_size, T, U, 2}, "compactedLogProbs");
  poputil::mapTensorLinearly(graph, probsCompacted, 1, 2);
  compactProbs(graph, prog, probs, probsCompacted, labels, labelLengths);
  assert(probsCompacted.shape() ==
         std::vector<std::size_t>({batch_size, T, U, 2}));
  Tensor probsCompactedShuf =
      probsCompacted.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(probsCompactedShuf.shape() ==
         std::vector<std::size_t>({2, T, U, batch_size}));

  poputil::PoplibsOpDebugInfo dnai(debugContext, DI_ARGS(probs));
  Tensor max = popops::reduce(graph, probsShuf, {0}, popops::Operation::MAX,
                              prog, {dnai, fnStr});
  assert(max.elementType() == dType);
  Tensor max_2 = max.expand({0}).broadcast(2, 0);
  assert(max_2.shape() == std::vector<std::size_t>({2, T, U, batch_size}));

  poplar::Tensor sum = graph.addVariable(dType, {batch_size, T, U}, "sum");
  poputil::mapTensorLinearly(graph, sum, 1, 1);
  auto maxShuffle = max.dimShuffle({2, 0, 1});
  poplar::ComputeSet computeSet = graph.addComputeSet("LogSoftmaxSum");
  mapVertex(graph, maxShuffle, probs, sum, computeSet);
  prog.add(program::Execute(computeSet));
  // Do not fuse with above expression to allow efficient use of broadcast
  // vertices.
  assert(max_2.elementType() == dType);
  mapInPlace(
      graph,
      expr::Sub(expr::Sub(expr::_1, expr::_2), expr::Cast(expr::_3, dType)),
      {probsCompactedShuf, max_2, sum.dimShuffle({1, 2, 0})}, prog,
      {dnai, fnStr});

  // Shuffle dimensions back to original ordering and return.
  // If inPlace == true then this is the same as the original tensor.
  probsCompacted =
      probsCompactedShuf.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(probsCompacted.shape() ==
         std::vector<std::size_t>({batch_size, T, U, 2}));
  return probsCompacted;
}

void expandGradientsInPlace(poplar::Graph &graph,
                            poplar::program::Sequence &program,
                            const poplar::Tensor &compacted_gradients,
                            poplar::Tensor &log_probs,
                            const poplar::Tensor &labels,
                            const poplar::Tensor &label_lengths) {
  size_t batch_size = log_probs.dim(0);
  size_t maxT = log_probs.dim(1);
  size_t maxU = log_probs.dim(2);
  size_t alphabet = log_probs.dim(3);
  poplar::Tensor compactedGradientsFinal =
      graph.addVariable(compacted_gradients.elementType(),
                        {batch_size, maxT, maxU, 2}, "compactedGradients_");
  poputil::mapTensorLinearly(graph, compactedGradientsFinal, 1, 2);
  program.add(
      poplar::program::Copy(compacted_gradients, compactedGradientsFinal));
  program.add(poplar::program::WriteUndef(compacted_gradients));
  poplar::ComputeSet computeSet = graph.addComputeSet("expansionAdd");
  for (size_t t = 0; t < maxT; ++t) {
    for (size_t u = 0; u < maxU; ++u) {
      for (size_t i = 0; i < batch_size; ++i) {
        const size_t tile = getTile(graph, log_probs[i][t][u]);
        connectVertex(graph, computeSet,
                      poputil::templateVertex(
                          "AddVertex", compactedGradientsFinal.elementType(),
                          log_probs.elementType()),
                      {{"out", log_probs[i][t][u][0]},
                       {"in", compactedGradientsFinal[i][t][u][0]}},
                      tile);
        if (u < maxU - 1) {
          auto vtx = connectVertex(
              graph, computeSet,
              poputil::templateVertex("AddExpandVertex",
                                      compactedGradientsFinal.elementType(),
                                      log_probs.elementType()),
              {{"out", log_probs[i][t][u].slice({1, alphabet})},
               {"in", compactedGradientsFinal[i][t][u][1]},
               {"label_len", label_lengths[i]},
               {"label", labels[i][u]}},
              tile);
          graph.setInitialValue(vtx["u"], u);
        }
      }
    }
  }
  program.add(poplar::program::Execute(computeSet));
}

//
// j - softmax dimension, that is - dimension along which a sum(exp) is
// performed i - all other dimensions indicated as one i index for simplicity
//
// dL/dx_j(i) = og_j(i) - sum_m(og_m(i)) * softmax_j(i)
//
Tensor logSoftmaxRnntGrad(Graph &graph, Tensor &probs, const Tensor &outerGrads,
                          const Tensor &labels, const Tensor &labelLengths,
                          Sequence &prog, const DebugContext &debugContext) {
  const std::string fnStr = "LogSoftmaxRnntGrad";
  const auto dType = probs.elementType();
  const auto rank = probs.rank();
  if (rank != 4) {
    throw poplibs_error(
        "probabilities tensor to LogSoftmaxRnntGrad must have 4 dimensions");
  }
  if (outerGrads.rank() != 4) {
    throw poplibs_error(
        "outer gradients tensor to LogSoftmaxRnntGrad must have 4 dimensions");
  }

  unsigned batch_size = probs.dim(0);
  unsigned T = probs.dim(1);
  unsigned U = probs.dim(2);
  unsigned alphabet = probs.dim(3);

  if (outerGrads.dim(0) != batch_size) {
    throw poplibs_error(
        std::string("outer gradients tensor dimension 0 must be ") +
        std::to_string(batch_size));
  }
  if (outerGrads.dim(1) != T) {
    throw poplibs_error(
        std::string("outer gradients tensor dimension 1 must be ") +
        std::to_string(T));
  }
  if (outerGrads.dim(2) != U) {
    throw poplibs_error(
        std::string("outer gradients tensor dimension 2 must be ") +
        std::to_string(U));
  }
  if (outerGrads.dim(3) != 2) {
    throw poplibs_error("outer gradients tensor dimension 3 must be 2");
  }

  poputil::PoplibsOpDebugInfo dnai(debugContext, DI_ARGS(probs, outerGrads));
  popnn::softmaxStableInPlace(graph, probs, prog, {dnai, fnStr});
  Tensor softMax = probs;

  assert(softMax.shape() ==
         std::vector<std::size_t>({batch_size, T, U, alphabet}));
  Tensor softMaxShuf = softMax.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(softMaxShuf.shape() ==
         std::vector<std::size_t>({alphabet, T, U, batch_size}));

  Tensor outerGradsShuf =
      outerGrads.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(outerGradsShuf.shape() ==
         std::vector<std::size_t>({2, T, U, batch_size}));

  Tensor sumOuterGrads =
      popops::reduce(graph, outerGradsShuf, {0}, {popops::Operation::ADD}, prog,
                     fnStr + "/ReduceSumOuterGrad");
  assert(sumOuterGrads.shape() == std::vector<std::size_t>({T, U, batch_size}));
  popops::mapInPlace(
      graph, expr::Neg(expr::Mul(expr::_1, expr::Cast(expr::_2, dType))),
      {softMaxShuf, sumOuterGrads}, prog, {dnai, fnStr + "/MapInPlace"});
  expandGradientsInPlace(graph, prog, outerGrads, softMax, labels,
                         labelLengths);

  return softMax;
}
