// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "rnnt_ipu_dynamic.hpp"
#include "ipu_utils.hpp"

#include <poplar/Tensor.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include <vector>
#include <climits>
#include <numeric>

using namespace poplar;

// Compact log probabilities from [B, T, U, A] to [B, T, U, 2]
Tensor compactLogProbs(Graph &graph, const Tensor &logProbs, const Tensor &labels, const Tensor &labelLengths, ComputeSet &cs) {
  assert(logProbs.rank() == 4);
  unsigned B = logProbs.dim(0);
  unsigned T = logProbs.dim(1);
  unsigned U = logProbs.dim(2);
  assert(U > 1);

  assert(labels.shape() == std::vector<std::size_t>({B, U - 1}));
  assert(labels.elementType() == INT);

  assert(labelLengths.shape() == std::vector<std::size_t>({B}));
  assert(labelLengths.elementType() == INT);

  Type dataType = logProbs.elementType();

  Tensor logProbsCompacted0_B_T_U = graph.addVariable(FLOAT, {B, T, U}, "logProbsCompacted_0_BTU");
  poputil::mapTensorLinearly(graph, logProbsCompacted0_B_T_U, 1, 1);
  const std::vector<std::vector<Interval>> mapping_BTU = graph.getTileMapping(logProbsCompacted0_B_T_U);
  Tensor logProbsCompacted0_BTU = logProbsCompacted0_B_T_U.flatten();

  Tensor logProbsCompacted1_B_T_U = graph.addVariable(FLOAT, {B, T, U}, "logProbsCompacted1_B_T_U");
  graph.setTileMapping(logProbsCompacted1_B_T_U, mapping_BTU);
  Tensor logProbsCompacted1_BTU = logProbsCompacted1_B_T_U.flatten();

  Tensor logProbs_BTU_A = logProbs.flatten(0, 3);
  // The labels at u >=  U - 1 do not exist
  Tensor labels_padUm1 = graph.addConstant(labels.elementType(), {B, 1}, std::vector<int>(B, 0).data(), "labels_padUm1");
  poputil::mapTensorLinearly(graph, labels_padUm1, 1, 1);
  Tensor labels_B_U = concat(labels, labels_padUm1, 1);
  assert(labels_B_U.shape() == std::vector<std::size_t>({B, U}));

  Tensor labels_BTU = labels_B_U.expand({1}).broadcast(T, 1).flatten();
  assert(labels_BTU.shape() == std::vector<std::size_t>({B * T * U}));

  Tensor labelLengths_BTU = labelLengths.expand({1, 1}).broadcast(T, 1).broadcast(U, 2).flatten();
  assert(labelLengths_BTU.shape() == std::vector<std::size_t>({B * T * U}));

  std::vector<int> uMapHost(U);
  for (int i = 0; i < static_cast<int>(U); ++i) {
    uMapHost[i] = i;
  }
  Tensor uMap = graph.addConstant(INT, {U}, uMapHost.data(), "uMap");
  poputil::mapTensorLinearly(graph, uMap, 1, 1);
  Tensor uMap_BTU = uMap.expand({0, 0}).broadcast(B, 0).broadcast(T, 1).flatten();
  assert(uMap_BTU.shape() == std::vector<std::size_t>({B * T * U}));

  const auto &target = graph.getTarget();
  const auto numTiles = target.getTilesPerIPU();
  for (size_t idxTile = 0; idxTile < numTiles; ++idxTile) {
    const auto &regions = mapping_BTU[idxTile];
    if (!regions.empty()) {
      auto vertexRegions = poputil::splitRegionsBetweenWorkers(target, regions, 1, 1);
      for (auto &vertexRegion : vertexRegions) {
        VertexRef vtx = graph.addVertex(cs, poputil::templateVertex("CompactLogProbsVertex", dataType));
        graph.setTileMapping(vtx, idxTile);
        Tensor logProbs_BTU_A_Slices = concat(logProbs_BTU_A.slices(vertexRegion));
        graph.connect(vtx["logProbs_BTU_A"], logProbs_BTU_A_Slices);
        graph.connect(vtx["labels_BTU"], concat(labels_BTU.slices(vertexRegion)));
        graph.connect(vtx["uMap_BTU"], concat(uMap_BTU.slices(vertexRegion)));
        graph.connect(vtx["labelLengths_BTU"], concat(labelLengths_BTU.slices(vertexRegion)));
        graph.connect(vtx["logProbsCompacted0_BTU"], concat(logProbsCompacted0_BTU.slices(vertexRegion)));
        graph.connect(vtx["logProbsCompacted1_BTU"], concat(logProbsCompacted1_BTU.slices(vertexRegion)));

        graph.setPerfEstimate(vtx, logProbs_BTU_A_Slices.dim(0));
      }
    }
  }
  Tensor logProbsCompacted = concat(logProbsCompacted0_B_T_U.expand({3}), logProbsCompacted1_B_T_U.expand({3}), 3);
  assert(logProbsCompacted.shape() == std::vector<std::size_t>({B, T, U, 2}));

  return logProbsCompacted;
}

Tensor compactLogProbs(Graph &graph, const Tensor &logProbs, const Tensor &labels, const Tensor &labelLengths, program::Sequence &program) {
  ComputeSet cs = graph.addComputeSet("compactLogProbs");
  Tensor logProbsCompacted = compactLogProbs(graph, logProbs, labels, labelLengths, cs);
  program.add(program::Execute(cs));
  return logProbsCompacted;
}

void expandAndSub(Graph &graph, const Tensor &compactedT, Tensor subT, const Tensor &labels, const Tensor &labelLengths, program::Sequence &program) {
  assert(subT.rank() == 4);
  unsigned B = subT.dim(0);
  unsigned T = subT.dim(1);
  unsigned U = subT.dim(2);
  unsigned A = subT.dim(3);
  assert(U > 1);
  unsigned A2 = 2;

  assert(compactedT.shape() == std::vector<std::size_t>({B, T, U, A2}));

  assert(labels.shape() == std::vector<std::size_t>({B, U - 1}));
  assert(labels.elementType() == INT);

  assert(labelLengths.shape() == std::vector<std::size_t>({B}));
  assert(labelLengths.elementType() == INT);

  Type dataType = subT.elementType();

  Tensor compactedT0 = graph.addVariable(compactedT.elementType(), {B, T, U}, "compactedT0");
  poputil::mapTensorLinearly(graph, compactedT0, 1, 1);
  const std::vector<std::vector<Interval>> mapping_BTU = graph.getTileMapping(compactedT0);
  program.add(program::Copy(compactedT.slice(0, 1, 3), compactedT0));
  Tensor compactedT0_BTU = compactedT0.flatten();

  Tensor compactedT1 = graph.addVariable(compactedT.elementType(), {B, T, U}, "compactedT1");
  graph.setTileMapping(compactedT1, mapping_BTU);
  program.add(program::Copy(compactedT.slice(1, A2, 3), compactedT1));
  Tensor compactedT1_BTU = compactedT1.flatten();

  Tensor subT_BTU_A = subT.flatten(0, 3);

  // The labels at u >=  U - 1 do not exist
  Tensor labels_padUm1 = graph.addConstant(labels.elementType(), {B, 1}, std::vector<int>(B, 0).data(), "labels_padUm1");
  poputil::mapTensorLinearly(graph, labels_padUm1, 1, 1);
  Tensor labels_B_U = concat(labels, labels_padUm1, 1);
  assert(labels_B_U.shape() == std::vector<std::size_t>({B, U}));

  Tensor labels_B_T_U = labels_B_U.expand({1}).broadcast(T, 1);
  assert(labels_B_T_U.shape() == std::vector<std::size_t>({B, T, U}));
  Tensor labels_BTU = labels_B_T_U.flatten();
  
  ComputeSet cs = graph.addComputeSet("expandAndSub");

  Tensor dimsT = graph.addConstant(UNSIGNED_INT, {4}, ArrayRef<unsigned>({B, T, U, A}));
  graph.setTileMapping(dimsT, 0);

  const auto &target = graph.getTarget();
  const auto numTiles = target.getTilesPerIPU();
  for (size_t idxTile = 0; idxTile < numTiles; ++idxTile) {
    const auto &regions = mapping_BTU[idxTile];
    if (!regions.empty()) {
      auto vertexRegions = poputil::splitRegionsBetweenWorkers(target, regions, 1, 1);
      for (auto &vertexRegion : vertexRegions) {
        assert(vertexRegion.size() == 1);
        unsigned intervalBegin = static_cast<unsigned>(vertexRegion[0].begin());
        Tensor intervalStart = graph.addConstant(UNSIGNED_INT, {}, ArrayRef<unsigned>({intervalBegin}));
        graph.setTileMapping(intervalStart, idxTile);

        VertexRef vtx = graph.addVertex(cs, poputil::templateVertex("expandAndSubVertex", dataType));
        graph.setTileMapping(vtx, idxTile);
        Tensor subT_Slices = concat(subT_BTU_A.slices(vertexRegion));
        graph.connect(vtx["subT"], subT_Slices);
        graph.connect(vtx["compactedT0"], concat(compactedT0_BTU.slices(vertexRegion)));
        graph.connect(vtx["compactedT1"], concat(compactedT1_BTU.slices(vertexRegion)));
        graph.connect(vtx["labels"], concat(labels_BTU.slices(vertexRegion)));
        graph.connect(vtx["labelLengths"], labelLengths);
        graph.connect(vtx["intervalStart"], intervalStart);
        graph.connect(vtx["dimsT"], dimsT);

        graph.setPerfEstimate(vtx, subT_Slices.dim(0));
      }
    }
  }
  program.add(program::Execute(cs));
}

// Shift log probabilities from [B, T, U, 2] to [N, U, B, 2]
Tensor shiftLogProbs(Graph &graph, const Tensor &logProbs, program::Sequence &prog, bool shiftAlsoInU) {
  // logProbs [B, T, U, 2]
  assert(logProbs.rank() == 4);
  unsigned B = logProbs.dim(0);
  unsigned T = logProbs.dim(1);
  unsigned U = logProbs.dim(2);
  const unsigned A2 = 2;
  assert(logProbs.dim(3) == A2);
  unsigned N = T + U - 1;

  Type dataType = logProbs.elementType();

  // T, U, B, 2
  Tensor logProbs_T_U_B_2 = graph.addVariable(dataType, {T, U, B, 2}, "logProbs_T_U_B_2");
  poputil::mapTensorLinearly(graph, logProbs_T_U_B_2, 1, 1);
  prog.add(program::Copy(logProbs.dimShuffle({1, 2, 0, 3}), logProbs_T_U_B_2));

  Tensor padSpace = graph.addConstant(dataType, {1}, ArrayRef<float>(std::vector<float>(1, -std::numeric_limits<float>::infinity())));
  graph.setTileMapping(padSpace, 0);
  padSpace = padSpace.expand({1, 1, 1});
  assert(padSpace.rank() == 4);
  Tensor padSpaceU = padSpace.broadcast(U - 1, 0).broadcast(1, 1).broadcast(B, 2).broadcast(A2, 3);
  assert(padSpaceU.shape() == std::vector<std::size_t>({U - 1, 1, B, A2}));

  Tensor shiftedSliceA1Prev;
  if (shiftAlsoInU) {
    Tensor padSpaceA = padSpace.broadcast(N, 0).broadcast(1, 1).broadcast(B, 2).broadcast(1, 2);
    assert(padSpaceA.shape() == std::vector<std::size_t>({N, 1, B, 1}));
    shiftedSliceA1Prev = padSpaceA;
  }
  std::vector<Tensor> shiftedSlices; 
  for (unsigned u = 0; u < U; ++u) {
    Tensor leftPad = padSpaceU.slice(0, u);
    Tensor rowValues = logProbs_T_U_B_2.slice(u, u + 1, 1);
    Tensor rightPad = padSpaceU.slice(u, U - 1);
    Tensor shiftedSlice = concat({leftPad, rowValues, rightPad});
    assert(shiftedSlice.shape() == std::vector<std::size_t>({N, 1, B, A2}));
    if (shiftAlsoInU) {
      Tensor shiftedSliceA0 = shiftedSlice.slice(0, 1, 3);
      Tensor shiftedSliceA1 = shiftedSlice.slice(1, A2, 3);
      assert(shiftedSliceA0.shape() == shiftedSliceA1Prev.shape());
      shiftedSlice = concat(shiftedSliceA0, shiftedSliceA1Prev, 3);
      assert(shiftedSlice.shape() == std::vector<std::size_t>({N, 1, B, A2}));
      shiftedSliceA1Prev = shiftedSliceA1;
    }
    shiftedSlices.push_back(shiftedSlice);
  }
  Tensor logProbsShifted = concat(shiftedSlices, 1);
  assert(logProbsShifted.shape() == std::vector<std::size_t>({N, U, B, A2}));
  Tensor logProbsShiftedSlice = graph.addVariable(dataType, {U, B, A2}, "logProbsShiftedSlice");
  poputil::mapTensorLinearly(graph, logProbsShiftedSlice, 1, 1);
  Tensor logProbsShiftedCopy = popops::createSliceableTensorFromSlice(graph, logProbsShiftedSlice.expand({0}), {0}, {N}, "logProbsShiftedCopy");
  prog.add(program::Copy(logProbsShifted, logProbsShiftedCopy));
#if 0
  dumpTileMapping(graph, logProbsShiftedCopy);
#endif
  return logProbsShiftedCopy;
}

class ProbsNSlicer {
  Graph &graph_;
  Tensor logProbs_;
  Tensor counterN_;

public:
  ProbsNSlicer(Graph &graph, const Tensor &logProbs, const Tensor& counterN)
    : graph_(graph)
    , logProbs_(logProbs)
    , counterN_(counterN)
  {}

  Tensor createSlice(program::Sequence &loop) {
    Tensor logProbsDiagSlice = popops::dynamicSlice(graph_, logProbs_, counterN_, {0}, {1}, loop);
#if 0
    loop.add(program::PrintTensor("logProbsDiagSlice", logProbsDiagSlice.flatten()));
#endif
    return logProbsDiagSlice;
  }
};

void incrementN(Graph &graph, Tensor &counterN, ComputeSet &cs) {
  VertexRef vtx = graph.addVertex(cs, "IncrementNVertex");
  graph.setPerfEstimate(vtx, 1);
  graph.connect(vtx["counterN"], counterN);
  graph.setTileMapping(vtx, 0);
}

void decrementN(Graph &graph, Tensor &counterN, ComputeSet &cs) {
  VertexRef vtx = graph.addVertex(cs, "DecrementNVertex");
  graph.setPerfEstimate(vtx, 1);
  graph.connect(vtx["counterN"], counterN);
  graph.setTileMapping(vtx, 0);
}

Tensor deshiftAB(const Tensor &t) {
  assert(t.rank() == 3);
  unsigned N = t.dim(0);
  unsigned U = t.dim(1);
  unsigned B = t.dim(2);
  assert(N > U);
  unsigned T = N - U + 1;
  std::vector<Tensor> tSlices;
  for (unsigned u = 0; u < U; ++u) {
    tSlices.push_back(t.slice({u, u, 0}, {u + T, u + 1, B}));
  }
  Tensor t_T_U_B = concat(tSlices, 1);
  assert(t_T_U_B.shape() == std::vector<std::size_t>({T, U, B}));
  return t_T_U_B;
}

Tensor computeAlpha(Graph &graph, const Tensor &logProbs, const Tensor &inputLengths, program::Sequence &prog) {
  // logProbs [N, U, B, 2]
  assert(logProbs.rank() == 4);

  unsigned N = logProbs.dim(0);
  unsigned U = logProbs.dim(1);
  unsigned B = logProbs.dim(2);
  assert(N > U);
  const unsigned A2 = 2;
  assert(logProbs.dim(3) == A2);

  assert(inputLengths.shape() == std::vector<std::size_t>({B}));
  assert(inputLengths.elementType() == INT);

  Type dataType = logProbs.elementType();

  Tensor logAlpha;
  Tensor logAlphaSliceIn = graph.addVariable(FLOAT, {U + 1, B}, "logAlphaSliceIn");
  std::vector<float> logAlphaSliceInHost((U + 1) * B, -std::numeric_limits<float>::infinity());
  // Initializing boundary condition for alpha
  // U=1 elements of logAlphaSlice iz 0
  for (unsigned b = 0; b < B; ++b) {
    logAlphaSliceInHost[B + b] = 0.0f;
  }
  ArrayRef<float> logAlphaSliceInRef(logAlphaSliceInHost);
  graph.setInitialValue(logAlphaSliceIn, logAlphaSliceInRef);
  poputil::mapTensorLinearly(graph, logAlphaSliceIn, 1, 1);
  Tensor logAlphaSliceIn_Um1 = logAlphaSliceIn.slice(0, U, 0);
  Tensor logAlphaSliceIn_U = logAlphaSliceIn.slice(1, U + 1);
  Tensor logAlphaSliceIn_Um1B = logAlphaSliceIn_Um1.flatten();
  Tensor logAlphaSliceIn_UB = logAlphaSliceIn_U.flatten();

  Tensor logAlphaSliceOut = graph.addVariable(FLOAT, {U, B}, "logAlphaSliceOut");
  poputil::mapTensorLinearly(graph, logAlphaSliceOut, 1, 1);
  Tensor logAlphaSliceOut_UB = logAlphaSliceOut.flatten();

  logAlpha = popops::createSliceableTensorFromSlice(graph, logAlphaSliceOut.expand({0}), {0}, {N}, "logAlpha");
  assert(logAlpha.shape() == std::vector<std::size_t>({N, U, B}));
  std::vector<float> logAlphaHost(N * U * B, -std::numeric_limits<float>::infinity());
  // Initializing boundary condition for alpha
  // U=1 elements of logAlphaSlice iz 0
  for (unsigned b = 0; b < B; ++b) {
    logAlphaHost[b] = 0.0f;
  }

  ArrayRef<float> logAlphaRef(logAlphaHost);
  graph.setInitialValue(logAlpha, logAlphaRef);
  Tensor logAlpha_N_UB = logAlpha.flatten(1, 3);

  Tensor inputLengths_U_B = inputLengths.expand({0}).broadcast(U, 0);
  assert(inputLengths_U_B.shape() == std::vector<std::size_t>({U, B}));
  Tensor inputLengths_UB = inputLengths_U_B.flatten();

  std::vector<int> uMapHost(U);
  for (int i = 0; i < static_cast<int>(U); ++i) {
    uMapHost[i] = i;
  }
  Tensor uMap = graph.addConstant(INT, {U}, uMapHost.data(), "uMap");
  poputil::mapTensorLinearly(graph, uMap, 1, 1);
  Tensor uMap_U_B = uMap.expand({1}).broadcast(B, 1);
  assert(uMap_U_B.shape() == std::vector<std::size_t>({U, B}));
  Tensor uMap_UB = uMap_U_B.flatten();
  
  Tensor counterN = graph.addVariable(UNSIGNED_INT, {1}, "counterN");
  graph.setTileMapping(counterN, 0);
  graph.setInitialValue(counterN, ArrayRef<unsigned>({0}));

  ProbsNSlicer probsSlicer(graph, logProbs, counterN);
  ComputeSet csAlpha = graph.addComputeSet("computeSetAlpha");
  ComputeSet csIncrementN = graph.addComputeSet("IncrementN");
  program::Sequence loopByN;
  {
    Tensor logProbsNSlice = probsSlicer.createSlice(loopByN);
    assert(logProbsNSlice.shape() == std::vector<std::size_t>({1, U, B, A2}));
    Tensor logProbsNSlice_UB_A2 = logProbsNSlice.flatten(0, 3);

    const std::vector<std::vector<Interval>> mapping_UB = graph.getTileMapping(logAlphaSliceOut_UB);
    const auto &target = graph.getTarget();
    const auto numTiles = target.getTilesPerIPU();
    for (size_t idxTile = 0; idxTile < numTiles; ++idxTile) {
      const auto &regions = mapping_UB[idxTile];
      if (!regions.empty()) {
        auto vertexRegions = poputil::splitRegionsBetweenWorkers(target, regions, 1, 1);
        for (auto &vertexRegion : vertexRegions) {
          VertexRef vtx = graph.addVertex(csAlpha, poputil::templateVertex("AlphaDynamicVertex", dataType));
          graph.setTileMapping(vtx, idxTile);

          graph.connect(vtx["logAlphaSliceIn_Um1B"], concat(logAlphaSliceIn_Um1B.slices(vertexRegion, 0)));
          graph.connect(vtx["logAlphaSliceIn_UB"], concat(logAlphaSliceIn_UB.slices(vertexRegion, 0)));
          Tensor logAlphaSliceOut_UB_Slices = concat(logAlphaSliceOut_UB.slices(vertexRegion, 0));
          graph.connect(vtx["logAlphaSliceOut_UB"], logAlphaSliceOut_UB_Slices);
          graph.connect(vtx["logProbsNSlice_UB_A2"], concat(logProbsNSlice_UB_A2.slices(vertexRegion, 0)));
          graph.connect(vtx["inputLengths_UB"], concat(inputLengths_UB.slices(vertexRegion, 0)));
          graph.connect(vtx["uMap_UB"], concat(uMap_UB.slices(vertexRegion, 0)));
          graph.connect(vtx["counterN"], counterN[0]);

          graph.setPerfEstimate(vtx, logAlphaSliceOut_UB_Slices.dim(0));
        }
      }
    }
 
    loopByN.add(program::Execute(csAlpha));
#if 0
    loopByN.add(program::PrintTensor("counterN", counterN));
    loopByN.add(program::PrintTensor("logProbsNSlice", logProbsNSlice));
    loopByN.add(program::PrintTensor("logAlphaSliceIn_Um1B", logAlphaSliceIn_Um1B));
    loopByN.add(program::PrintTensor("logAlphaSliceIn_UB", logAlphaSliceIn_UB));
    loopByN.add(program::PrintTensor("logAlphaSliceOut_UB", logAlphaSliceOut_UB));
#endif
    loopByN.add(program::Copy(logAlphaSliceOut_UB, logAlphaSliceIn_UB));

    incrementN(graph, counterN, csIncrementN);
    loopByN.add(program::Execute(csIncrementN));

    // Update next N layer
    popops::dynamicUpdate(graph, logAlpha_N_UB, logAlphaSliceOut_UB.expand({0}), counterN, {0}, {1}, loopByN);
  }
  prog.add(program::Repeat(N - 1, loopByN));
#if 0
  prog.add(program::PrintTensor("logAlphaShifted", logAlpha));
#endif
  logAlpha = deshiftAB(logAlpha);
  return logAlpha;
}

Tensor computeBeta(Graph &graph, const Tensor &logProbs, const Tensor &inputLengths, const Tensor &labelLengths, program::Sequence &prog) {
  // logProbs [N, U, B, 2]
  assert(logProbs.rank() == 4);

  unsigned N = logProbs.dim(0);
  unsigned U = logProbs.dim(1);
  unsigned B = logProbs.dim(2);
  assert(N > U);
  const unsigned A2 = 2;
  assert(logProbs.dim(3) == A2);

  assert(inputLengths.shape() == std::vector<std::size_t>({B}));
  assert(inputLengths.elementType() == INT);
  assert(labelLengths.shape() == std::vector<std::size_t>({B}));
  assert(labelLengths.elementType() == INT);

  Type dataType = logProbs.elementType();

  Tensor logBeta;
  Tensor logBetaSliceIn = graph.addVariable(FLOAT, {U + 1, B}, "logBetaSliceIn");
  std::vector<float> logBetaSliceInHost((U + 1) * B, -std::numeric_limits<float>::infinity());
  ArrayRef<float> logBetaSliceInRef(logBetaSliceInHost);
  graph.setInitialValue(logBetaSliceIn, logBetaSliceInRef);
  poputil::mapTensorLinearly(graph, logBetaSliceIn, 1, 1);
  Tensor logBetaSliceIn_Up1 = logBetaSliceIn.slice(1, U + 1);
  Tensor logBetaSliceIn_U = logBetaSliceIn.slice(0, U);
  Tensor logBetaSliceIn_Up1B = logBetaSliceIn_Up1.flatten();
  Tensor logBetaSliceIn_UB = logBetaSliceIn_U.flatten();

  Tensor logBetaSliceOut = graph.addVariable(FLOAT, {U, B}, "logBetaSliceOut");
  poputil::mapTensorLinearly(graph, logBetaSliceOut, 1, 1);
  Tensor logBetaSliceOut_UB = logBetaSliceOut.flatten();

  logBeta = popops::createSliceableTensorFromSlice(graph, logBetaSliceOut.expand({0}), {0}, {N}, "logBeta");
  assert(logBeta.shape() == std::vector<std::size_t>({N, U, B}));
  std::vector<float> logBetaHost(N * U * B, -std::numeric_limits<float>::infinity());
  ArrayRef<float> logBetaRef(logBetaHost);
  graph.setInitialValue(logBeta, logBetaRef);
  Tensor logBeta_N_UB = logBeta.flatten(1, 3);

  Tensor inputLengths_U_B = inputLengths.expand({0}).broadcast(U, 0);
  assert(inputLengths_U_B.shape() == std::vector<std::size_t>({U, B}));
  Tensor inputLengths_UB = inputLengths_U_B.flatten();

  Tensor labelLengths_U_B = labelLengths.expand({0}).broadcast(U, 0);
  assert(labelLengths_U_B.shape() == std::vector<std::size_t>({U, B}));
  Tensor labelLengths_UB = labelLengths_U_B.flatten();

  std::vector<int> uMapHost(U);
  for (int i = 0; i < static_cast<int>(U); ++i) {
    uMapHost[i] = i;
  }
  Tensor uMap = graph.addConstant(INT, {U}, uMapHost.data(), "uMap");
  poputil::mapTensorLinearly(graph, uMap, 1, 1);
  Tensor uMap_U_B = uMap.expand({1}).broadcast(B, 1);
  assert(uMap_U_B.shape() == std::vector<std::size_t>({U, B}));
  Tensor uMap_UB = uMap_U_B.flatten();
  
  Tensor counterN = graph.addVariable(UNSIGNED_INT, {1}, "counterN");
  graph.setTileMapping(counterN, 0);
  graph.setInitialValue(counterN, ArrayRef<unsigned>({N - 1}));

  ProbsNSlicer probsSlicer(graph, logProbs, counterN);
  ComputeSet csBeta = graph.addComputeSet("computeSetBeta");
  ComputeSet csDecrementN = graph.addComputeSet("DecrementN");
  program::Sequence loopByN;
  {
    Tensor logProbsNSlice = probsSlicer.createSlice(loopByN);
    assert(logProbsNSlice.shape() == std::vector<std::size_t>({1, U, B, A2}));
    Tensor logProbsNSlice_UB_A2 = logProbsNSlice.flatten(0, 3);

    const std::vector<std::vector<Interval>> mapping_UB = graph.getTileMapping(logBetaSliceOut_UB);
    const auto &target = graph.getTarget();
    const auto numTiles = target.getTilesPerIPU();
    for (size_t idxTile = 0; idxTile < numTiles; ++idxTile) {
      const auto &regions = mapping_UB[idxTile];
      if (!regions.empty()) {
        auto vertexRegions = poputil::splitRegionsBetweenWorkers(target, regions, 1, 1);
        for (auto &vertexRegion : vertexRegions) {
          VertexRef vtx = graph.addVertex(csBeta, poputil::templateVertex("BetaDynamicVertex", dataType));
          graph.setTileMapping(vtx, idxTile);

          graph.connect(vtx["logBetaSliceIn_Up1B"], concat(logBetaSliceIn_Up1B.slices(vertexRegion, 0)));
          graph.connect(vtx["logBetaSliceIn_UB"], concat(logBetaSliceIn_UB.slices(vertexRegion, 0)));
          Tensor logBetaSliceOut_UB_Slices = concat(logBetaSliceOut_UB.slices(vertexRegion, 0));
          graph.connect(vtx["logBetaSliceOut_UB"], logBetaSliceOut_UB_Slices);
          graph.connect(vtx["logProbsNSlice_UB_A2"], concat(logProbsNSlice_UB_A2.slices(vertexRegion, 0)));
          graph.connect(vtx["inputLengths_UB"], concat(inputLengths_UB.slices(vertexRegion, 0)));
          graph.connect(vtx["labelLengths_UB"], concat(labelLengths_UB.slices(vertexRegion, 0)));
          graph.connect(vtx["uMap_UB"], concat(uMap_UB.slices(vertexRegion, 0)));
          graph.connect(vtx["counterN"], counterN[0]);

          graph.setPerfEstimate(vtx, logBetaSliceOut_UB_Slices.dim(0));
        }
      }
    }
 
    loopByN.add(program::Execute(csBeta));
#if 0
    loopByN.add(program::PrintTensor("counterN", counterN));
    loopByN.add(program::PrintTensor("logProbsNSlice", logProbsNSlice));
    loopByN.add(program::PrintTensor("logBetaSliceIn_Up1B", logBetaSliceIn_Up1B));
    loopByN.add(program::PrintTensor("logBetaSliceIn_UB", logBetaSliceIn_UB));
    loopByN.add(program::PrintTensor("logBetaSliceOut_UB", logBetaSliceOut_UB));
#endif
    loopByN.add(program::Copy(logBetaSliceOut_UB, logBetaSliceIn_UB));

    // Update next N layer
    popops::dynamicUpdate(graph, logBeta_N_UB, logBetaSliceOut_UB.expand({0}), counterN, {0}, {1}, loopByN);

    decrementN(graph, counterN, csDecrementN);
    loopByN.add(program::Execute(csDecrementN));
  }
  prog.add(program::Repeat(N, loopByN));
  logBeta = deshiftAB(logBeta);
  return logBeta;
}

Tensor computeLoss(Graph &graph, const Tensor &logBeta, program::Sequence &prog) {
  // logBeta [T, U, B]
  assert(logBeta.rank() == 3);
  Tensor loss = popops::map(graph, popops::expr::UnaryOpType::NEGATE, logBeta[0][0], prog);
  return loss;
}

Tensor extractTUSlice(Graph &graph, const Tensor &t, const Tensor &inputLengths, const Tensor &labelLengths, program::Sequence &prog) {
  // logProbs [T, U, B]
  assert(t.rank() == 3);
  unsigned B = t.dim(2);

  assert(inputLengths.shape() == std::vector<std::size_t>({B}));
  assert(inputLengths.elementType() == INT);
  assert(labelLengths.shape() == std::vector<std::size_t>({B}));
  assert(labelLengths.elementType() == INT);

  Tensor inputLengthsUint = popops::cast(graph, inputLengths, UNSIGNED_INT, prog);
  Tensor one = graph.addConstant(UNSIGNED_INT, {B}, ArrayRef<unsigned>(std::vector<unsigned>(B, 1)));
  graph.setTileMapping(one, 0);
  popops::subInPlace(graph, inputLengthsUint, one, prog);
  
  Tensor labelLengthsUint = popops::cast(graph, labelLengths, UNSIGNED_INT, prog);

  std::vector<unsigned> bMapHost(B);
  for (unsigned i = 0; i < B; ++i) {
    bMapHost[i] = i;
  }
  Tensor bMap = graph.addConstant(UNSIGNED_INT, {B}, bMapHost.data(), "bMap");
  poputil::mapTensorLinearly(graph, bMap, 1, 1);

  Tensor indicies_TUB = concat({inputLengthsUint.expand({0}), labelLengthsUint.expand({0}), bMap.expand({0})}, 0);
  assert(indicies_TUB.shape() == std::vector<std::size_t>({3, B}));

  std::size_t indexVectorDim = 0;

  std::vector<std::size_t> offsetDims = {};

  std::vector<std::size_t> sliceSizes = {1, 1, 1};

  std::vector<std::size_t> collapsedSliceDims = {0, 1, 2};

  std::vector<unsigned> startIndexMap = {0, 1, 2};

  Tensor tSlice_TU = 
    popops::gather(graph, t,
                   indicies_TUB,
                   indexVectorDim,
                   offsetDims,
                   sliceSizes,
                   collapsedSliceDims,
                   startIndexMap,
                   prog);                   
  return tSlice_TU;
}

Tensor computeGrads(Graph &graph, const Tensor &logProbs,
                    const Tensor &logAlpha, const Tensor &logBeta, const Tensor &logLoss,
                    const Tensor &inputLengths, const Tensor &labelLengths,
                    program::Sequence &prog) {
  // logProbs [T, U, B]
  assert(logProbs.rank() == 4);
  unsigned B = logProbs.dim(0);
  unsigned T = logProbs.dim(1);
  unsigned U = logProbs.dim(2);
  const unsigned A2 = 2;
  assert(logProbs.dim(3) == A2);

  assert(logAlpha.shape() == std::vector<std::size_t>({T, U, B}));
  assert(logBeta.shape() == std::vector<std::size_t>({T, U, B}));
  assert(logLoss.shape() == std::vector<std::size_t>({B}));

  assert(labelLengths.shape() == std::vector<std::size_t>({B}));
  assert(labelLengths.elementType() == INT);

  assert(inputLengths.shape() == std::vector<std::size_t>({B}));
  assert(inputLengths.elementType() == INT);

  Type dataType = logProbs.elementType();

  Tensor logProbs_0_B_T_U = logProbs.slice(0, 1, 3);
  Tensor logProbs_1_B_T_U = logProbs.slice(1, A2, 3);

  const std::vector<std::vector<Interval>> mapping_BTU0 = graph.getTileMapping(logProbs_0_B_T_U);
  const std::vector<std::vector<Interval>> mapping_BTU1 = graph.getTileMapping(logProbs_1_B_T_U);

  Tensor grads_B_T_U_0 = graph.addVariable(FLOAT, {B, T, U, 1}, "grads_B_T_U_0");
  graph.setTileMapping(grads_B_T_U_0, mapping_BTU0);
  Tensor grads_0 = grads_B_T_U_0.flatten();
  
  Tensor grads_B_T_U_1 = graph.addVariable(FLOAT, {B, T, U, 1}, "grads_B_T_U_1");
  graph.setTileMapping(grads_B_T_U_1, mapping_BTU1);
  Tensor grads_1 = grads_B_T_U_1.flatten();

  Tensor logProbs_0_BTU = logProbs_0_B_T_U.flatten();
  Tensor logProbs_1_BTU = logProbs_1_B_T_U.flatten();

  Tensor logAlpha_B_T_U = logAlpha.dimShuffle({2, 0, 1});
  assert(logAlpha_B_T_U.shape() == std::vector<std::size_t>({B, T, U}));
  Tensor logAlpha_BTU = logAlpha_B_T_U.flatten();

  Tensor logBeta_Tp1_Up1_B = graph.addVariable(FLOAT, {T + 1, U + 1, B}, "logBeta_Tp1_Up1_B");
  ArrayRef<float> logBeta_Tp1_Up1_BRef(std::vector<float>((T + 1) * (U + 1) * B, 0.0f));
  graph.setInitialValue(logBeta_Tp1_Up1_B, logBeta_Tp1_Up1_BRef);
  poputil::mapTensorLinearly(graph, logBeta_Tp1_Up1_B, 1, 1);
  prog.add(program::Copy(logBeta, logBeta_Tp1_Up1_B.slice({0, 0, 0}, {T, U, B})));

  Tensor logBeta_B_Ts1_U = logBeta_Tp1_Up1_B.slice({1, 0, 0}, {T + 1, U, B}).dimShuffle({2, 0, 1});
  assert(logBeta_B_Ts1_U.shape() == std::vector<std::size_t>({B, T, U}));
  Tensor logBeta_BTs1U = logBeta_B_Ts1_U.flatten();

  Tensor logBeta_B_T_Us1 = logBeta_Tp1_Up1_B.slice({0, 1, 0}, {T, U + 1, B}).dimShuffle({2, 0, 1});
  assert(logBeta_B_T_Us1.shape() == std::vector<std::size_t>({B, T, U}));
  Tensor logBeta_BTUs1 = logBeta_B_T_Us1.flatten();

  Tensor logLoss_BTU = logLoss.expand({1, 1}).broadcast(T, 1).broadcast(U, 2).flatten();
  assert(logLoss_BTU.shape() == std::vector<std::size_t>({B * T * U}));

  Tensor labelLengths_BTU = labelLengths.expand({1, 1}).broadcast(T, 1).broadcast(U, 2).flatten();
  assert(labelLengths_BTU.shape() == std::vector<std::size_t>({B * T * U}));

  Tensor inputLengths_BTU = inputLengths.expand({1, 1, 1}).broadcast(T, 1).broadcast(U, 2).flatten();
  assert(inputLengths_BTU.shape() == std::vector<std::size_t>({B * T * U}));

  std::vector<int> uMapHost(U);
  for (int i = 0; i < static_cast<int>(U); ++i) {
    uMapHost[i] = i;
  }
  Tensor uMap = graph.addConstant(INT, {U}, uMapHost.data(), "uMap");
  poputil::mapTensorLinearly(graph, uMap, 1, 1);
  Tensor uMap_BTU = uMap.expand({0, 0}).broadcast(B, 0).broadcast(T, 1).flatten();
  assert(uMap_BTU.shape() == std::vector<std::size_t>({B * T * U}));

  std::vector<int> tMapHost(T);
  for (int i = 0; i < static_cast<int>(T); ++i) {
    tMapHost[i] = i;
  }
  Tensor tMap = graph.addConstant(INT, {T}, tMapHost.data(), "tMap");
  poputil::mapTensorLinearly(graph, tMap, 1, 1);
  Tensor tMap_B_T = tMap.expand({0}).broadcast(B, 0);
  Tensor tMap_B_T_U = tMap_B_T.expand({2}).broadcast(U, 2);
  assert(tMap_B_T_U.shape() == std::vector<std::size_t>({B, T, U}));
  Tensor tMap_BTU = tMap_B_T_U.flatten();

  ComputeSet csGrads = graph.addComputeSet("computeSetGrads");
  const auto &target = graph.getTarget();
  const auto numTiles = target.getTilesPerIPU();
  const auto workersPerTile = target.getNumWorkerContexts();
  const auto workersPerTile0 = workersPerTile / 2;
  const auto workersPerTile1 = workersPerTile - workersPerTile0;
  for (size_t idxTile = 0; idxTile < numTiles; ++idxTile) {
    const auto &regions0 = mapping_BTU0[idxTile];
    if (!regions0.empty()) {
      auto vertexRegions = poputil::splitRegions(regions0, 1, workersPerTile0, 1);
      for (auto &vertexRegion : vertexRegions) {
        VertexRef vtx = graph.addVertex(csGrads, poputil::templateVertex("Grads0DynamicVertex", dataType));
        graph.setTileMapping(vtx, idxTile);
        Tensor logProbs_BTUSlices = concat(logProbs_0_BTU.slices(vertexRegion, 0));
        graph.connect(vtx["logProbs_0_BTU"], logProbs_BTUSlices);
        graph.connect(vtx["logAlpha_BTU"], concat(logAlpha_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["logBeta_BTs1U"], concat(logBeta_BTs1U.slices(vertexRegion, 0)));
        graph.connect(vtx["logLoss_BTU"], concat(logLoss_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["labelLengths_BTU"], concat(labelLengths_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["inputLengths_BTU"], concat(inputLengths_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["uMap_BTU"], concat(uMap_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["tMap_BTU"], concat(tMap_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["grads_0"], concat(grads_0.slices(vertexRegion, 0)));
        graph.setPerfEstimate(vtx, logProbs_BTUSlices.dim(0));
      }
    }
    const auto &regions1 = mapping_BTU1[idxTile];
    if (!regions1.empty()) {
      auto vertexRegions = poputil::splitRegions(regions1, 1, workersPerTile1, 1);
      for (auto &vertexRegion : vertexRegions) {
        VertexRef vtx = graph.addVertex(csGrads, poputil::templateVertex("Grads1DynamicVertex", dataType));
        graph.setTileMapping(vtx, idxTile);
        Tensor logProbs_BTUSlices = concat(logProbs_1_BTU.slices(vertexRegion, 0));
        graph.connect(vtx["logProbs_1_BTU"], logProbs_BTUSlices);
        graph.connect(vtx["logAlpha_BTU"], concat(logAlpha_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["logBeta_BTUs1"], concat(logBeta_BTUs1.slices(vertexRegion, 0)));
        graph.connect(vtx["logLoss_BTU"], concat(logLoss_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["labelLengths_BTU"], concat(labelLengths_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["inputLengths_BTU"], concat(inputLengths_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["uMap_BTU"], concat(uMap_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["tMap_BTU"], concat(tMap_BTU.slices(vertexRegion, 0)));
        graph.connect(vtx["grads_1"], concat(grads_1.slices(vertexRegion, 0)));
        graph.setPerfEstimate(vtx, logProbs_BTUSlices.dim(0));
      }
    }
  }
  prog.add(program::Execute(csGrads));

  Tensor grads = concat(grads_B_T_U_0, grads_B_T_U_1, 3);
  assert(grads.shape() == std::vector<std::size_t>({B, T, U, A2}));

  return grads;
}
