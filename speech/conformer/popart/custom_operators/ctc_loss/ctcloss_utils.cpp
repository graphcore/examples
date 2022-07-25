// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "ctcloss_utils.hpp"

#include <popnn/Loss.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>

Tensor initBatchIndices(Graph &graph, const Tensor &targets,
                        uint32_t sequenceLength) {
  // prepare the gather of t=0
  auto batchSize = targets.dim(0);
  std::vector<uint16_t> bsRange;
  for (size_t i = 0; i < batchSize; ++i)
    bsRange.push_back(i);
  poplar::ArrayRef<uint16_t> bsArrayRef(bsRange);
  poplar::Tensor batchIndices = graph.addConstant(
      poplar::UNSIGNED_SHORT, {batchSize, 1}, bsArrayRef, "BatchSizeIndices");
  poplar::Tensor result = batchIndices.broadcast(sequenceLength, 1).transpose();
  return result;
}

Tensor initSlicesIndices(Graph &graph, const Tensor &targets,
                         uint32_t sequenceLength) {
  // prepare the gather of t=0
  auto batchSize = targets.dim(0);
  std::vector<uint16_t> sliceRange;
  for (size_t i = 0; i < sequenceLength; ++i)
    sliceRange.push_back(i);
  poplar::ArrayRef<uint16_t> sliceArrayRef(sliceRange);
  poplar::Tensor sliceIndices =
      graph.addConstant(poplar::UNSIGNED_SHORT, {sequenceLength, 1},
                        sliceArrayRef, "SliceIndices");
  poplar::Tensor result = sliceIndices.broadcast(batchSize, 1);
  return result;
}

void iterVertex(Graph &graph,
                const std::unordered_map<std::string, Tensor> &indexed,
                size_t range, ComputeSet &computeSet,
                const std::string &vertexName, size_t estimate,
                size_t startIndex) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getTilesPerIPU();
  for (size_t i = 0; i < range; ++i) {
    VertexRef vtx = graph.addVertex(computeSet, vertexName);
    for (auto &p : indexed) {
      graph.connect(vtx[p.first],
                    p.second.rank() == 0 ? p.second : p.second[i]);
    }
    graph.setTileMapping(vtx, numTiles - 1 - ((startIndex + i) % numTiles));

    graph.setPerfEstimate(vtx, estimate);
  }
}

void mapTileVertex(Graph &graph,
                   const std::unordered_map<std::string, Tensor> &flat,
                   const std::unordered_map<std::string, Tensor> &full,
                   const std::unordered_map<std::string, Tensor> &indices,
                   const std::unordered_map<std::string, Tensor> &tiles,
                   const std::unordered_map<std::string, Tensor> &workers,
                   ComputeSet &computeSet, const std::string &vertexName,
                   const std::vector<Interval> &regions, uint16_t index,
                   uint32_t tileNumber, uint32_t splitSize) {
  auto vertexRegions =
      poputil::splitRegionsBetweenWorkers(graph.getTarget(), regions, 1, 1);
  size_t j = 0;
  for (auto &r : vertexRegions) {
    VertexRef vtx = graph.addVertex(computeSet, vertexName);
    for (auto &p : flat) {
      graph.connect(vtx[p.first], poplar::concat(p.second.slices(r)));
    }
    for (auto &p : full) {
      graph.connect(vtx[p.first], p.second);
    }
    for (auto &p : indices) {
      graph.connect(vtx[p.first], p.second[index]);
    }
    for (auto &p : tiles) {
      if (r.size() > 1) {
        std::cerr << "Debug " << tileNumber << ":" << regions.size() << " "
                  << r.size() << " ";
        for (const auto &t : r) {
          std::cerr << "(" << t.begin() << "," << t.end() << ") ";
        }
        std::cerr << std::endl;
      }
      const size_t dim0 = p.second.dim(0);
      if (r.size() > 1) {
        std::vector<Tensor> tensors;
        for (const auto &t : r) {
          tensors.emplace_back(p.second.slice({0, t.begin()}, {dim0, t.end()}));
        }
        graph.connect(vtx[p.first], poplar::concat(tensors));
      } else {
        const auto &interval = *r.begin();
        Tensor dest =
            p.second.slice({0, interval.begin()}, {dim0, interval.end()});
        graph.connect(vtx[p.first], dest);
      }
    }
    for (auto &p : workers) {
      graph.connect(vtx[p.first], p.second[index][j]);
    }
    graph.setPerfEstimate(vtx, r.size()); // wrong ...
    graph.setTileMapping(vtx, tileNumber);
    ++j;
  }
}

void mapVertex(Graph &graph,
               const std::unordered_map<std::string, Tensor> &flat,
               const std::unordered_map<std::string, Tensor> &full,
               const std::unordered_map<std::string, Tensor> &indices,
               const std::unordered_map<std::string, Tensor> &tiles,
               const std::unordered_map<std::string, Tensor> &workers,
               poplar::Type elementType, ComputeSet &computeSet,
               const std::string &vertexName,
               const std::vector<std::vector<Interval>> &mapping) {
  std::unordered_map<std::string, Tensor> flatten;
  for (auto &p : flat) {
    flatten.insert({p.first, p.second.flatten()});
  }
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(elementType);
  const auto splitSize =
      std::max<uint32_t>(vectorWidth, target.getAtomicStoreGranularity());
  const auto numTiles = target.getTilesPerIPU();
  size_t index = 0;
  for (size_t i = 0; i < numTiles; ++i) {
    auto &regions = mapping[i];
    if (regions.size() > 0)
      mapTileVertex(graph, flatten, full, indices, tiles, workers, computeSet,
                    vertexName, regions, index++, i, splitSize);
  }
}

void initInfinity(Graph &graph, Tensor &t, ComputeSet &cs) {
  mapVertex(graph, {{"out", t}}, {}, {}, {}, {}, t.elementType(), cs,
            poputil::templateVertex("InitInfinityVertex", t.elementType()),
            graph.getTileMapping(t));
}

void initInfinity(Graph &graph, program::Sequence &prog, Tensor &t) {
  ComputeSet cs = graph.addComputeSet("InitInfinity");
  initInfinity(graph, t, cs);
  prog.add(program::Execute(cs));
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
ctcLossPrepare(Graph &graph, const Tensor &targets, const Tensor &targetLengths,
               const poplar::Type probsType, program::Sequence &prog,
               bool partial32, size_t inputLength) {

  ComputeSet computeSet = graph.addComputeSet("computeSetInit");
  Tensor sequenceBs, diff, reverse, logAlpha, loss, batchIndices, sliceIndices;
  std::tie(sequenceBs, diff, reverse, logAlpha, loss, batchIndices,
           sliceIndices) =
      ctcLossPrepare(graph, targets, targetLengths, probsType, computeSet,
                     partial32, inputLength);
  prog.add(program::Execute(computeSet));
  const auto mapping = graph.getTileMapping(logAlpha[0]);
  Tensor sequenceCopy = graph.addVariable(sequenceBs.elementType(),
                                          sequenceBs.shape(), "sequence");
  Tensor sequence_ = sequenceCopy.transpose();
  graph.setTileMapping(sequence_, mapping);
  prog.add(program::Copy(sequenceBs, sequenceCopy));
  Tensor diffCopy = graph.addVariable(diff.elementType(), diff.shape(), "diff");
  Tensor diff_ = diffCopy.transpose();
  graph.setTileMapping(diff_, mapping);
  prog.add(program::Copy(diff, diffCopy));
  Tensor reverseCopy =
      graph.addVariable(reverse.elementType(), reverse.shape(), "reverse");
  Tensor reverse_ = reverseCopy.transpose();
  graph.setTileMapping(reverse_, mapping);
  prog.add(program::Copy(reverse, reverseCopy));
  return std::make_tuple(sequence_, sequenceBs, diff_, reverse_, logAlpha, loss,
                         batchIndices, sliceIndices);
}

std::vector<uint32_t>
getTiles(const std::vector<std::vector<Interval>> &mapping) {
  std::vector<uint32_t> res;
  for (uint32_t i = 0; i < mapping.size(); ++i) {
    if (mapping[i].size() > 0)
      res.push_back(i);
  }
  return res;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
ctcLossPrepare(Graph &graph, const Tensor &targets, const Tensor &targetLengths,
               const poplar::Type probsType, ComputeSet &cs, bool partial32,
               size_t inputLength) {
  auto batchSize = targets.dim(0);
  auto targetLength = targets.dim(1);
  unsigned sequenceLength = 2 * targetLength + 1;
  inputLength = (inputLength % 2) == 1 ? inputLength + 1 : inputLength;
  const auto &target = graph.getTarget();
  const auto numTiles = target.getTilesPerIPU();
  const auto numWorkers = target.getNumWorkerContexts();
  // Add variables to the graph
  Tensor sequence_ = graph.addVariable(
      targets.elementType(), {batchSize, sequenceLength}, "sequence_");
  poputil::mapTensorLinearly(graph, sequence_, 1, sequenceLength);
  Tensor diff_ =
      graph.addVariable(poplar::BOOL, {batchSize, sequenceLength}, "diff_");
  poputil::mapTensorLinearly(graph, diff_, 1, sequenceLength);
  Tensor reverse_ =
      graph.addVariable(poplar::BOOL, {batchSize, sequenceLength}, "reverse_");
  poputil::mapTensorLinearly(graph, reverse_, 1, sequenceLength);
  // initialize duplicate and sequence
  for (unsigned i = 0; i < batchSize; ++i) {
    VertexRef vtx =
        graph.addVertex(cs, poputil::templateVertex("InitVertex", probsType,
                                                    targets.elementType()));
    graph.setPerfEstimate(vtx, targetLength);
    graph.connect(vtx["in"], targets[i]);
    graph.connect(vtx["len"], targetLengths[i]);
    graph.connect(vtx["seq"], sequence_[i]);
    graph.connect(vtx["diff"], diff_[i]);
    graph.connect(vtx["revDiff"], reverse_[i]);
    graph.setTileMapping(vtx, numTiles - 1 -
                                  (i % (numTiles - 1))); // map vertex to tile i
  }

  auto elementType = partial32 ? poplar::FLOAT : probsType;
  Tensor dummySlice =
      graph.addVariable(elementType, {sequenceLength + 2, batchSize}, "dummy_");
  poputil::mapTensorLinearly(graph, dummySlice, 1, 1);
  dummySlice = dummySlice.slice({2, 0}, {sequenceLength + 2, batchSize});
  const auto mapping = graph.getTileMapping(dummySlice, true);
  Tensor logAlpha = popops::createSliceableTensorFromSlice(
      graph, dummySlice.expand({0}), {0, 1, 2}, {inputLength, 1, 1});

  const auto tiles = getTiles(mapping);
  Tensor loss = graph.addVariable(
      elementType, {tiles.size(), numWorkers, 2, batchSize}, "lossInput");
  for (size_t i = 0; i < tiles.size(); ++i)
    graph.setTileMapping(loss[i], tiles[i]);

  initInfinity(graph, loss, cs);
  Tensor batchIndices = initBatchIndices(graph, targets, sequenceLength);
  Tensor sliceIndices = initSlicesIndices(graph, targets, sequenceLength);
  graph.setTileMapping(batchIndices, mapping);
  graph.setTileMapping(sliceIndices, mapping);

  return std::make_tuple(sequence_, diff_, reverse_, logAlpha, loss,
                         batchIndices, sliceIndices);
}

std::pair<size_t, size_t> computeIndices(size_t input, size_t step) {
  return {input / step, input % step};
}

void mapOverTiles(Graph &graph, const std::vector<uint32_t> &tiles,
                  Tensor &input) {
  for (size_t i = 0; i < tiles.size(); ++i) {
    graph.setTileMapping(input[i], tiles[i]);
  }
}
Tensor replicate(Graph &graph, const std::vector<uint32_t> &tiles,
                 const Tensor &input) {
  const auto numTiles = tiles.size();
  Tensor inputRep = input.expand({0}).broadcast(numTiles, 0);
  mapOverTiles(graph, tiles, inputRep);
  return inputRep;
}

struct ContextBeta {
  const std::string vertexName;
  Tensor bufferIn, bufferOut, beta_t1, beta_t1_s1, beta_t1_s2;
  Tensor inputLengths_, targetLengths_;
  Tensor tensorT;
  size_t maxInputLength_;

  ContextBeta(Graph &graph, const std::vector<uint32_t> &tiles,
              poplar::Type type, poplar::Type probsType,
              poplar::Type sequenceType, size_t sequenceLength,
              size_t batchSize, const Tensor &inputLengths,
              const Tensor &targetLengths,
              const std::vector<std::vector<Interval>> &mapping,
              size_t maxInputLength)
      : vertexName{poputil::templateVertex("CTCBackwardVertex", type, probsType,
                                           sequenceType)},
        inputLengths_{inputLengths}, targetLengths_{targetLengths},
        maxInputLength_{maxInputLength} {
    const auto &target = graph.getTarget();
    const auto numWorkers = target.getNumWorkerContexts();
    bufferIn = graph.addVariable(type, {sequenceLength + 4, batchSize});
    poputil::mapTensorLinearly(graph, bufferIn, 1, 1);
    bufferIn = bufferIn.slice({2, 0}, {sequenceLength + 4, batchSize});
    bufferOut = graph.addVariable(type, {sequenceLength, batchSize});
    graph.setTileMapping(bufferOut, mapping);

    beta_t1 = bufferIn.slice({0, 0}, {sequenceLength, batchSize});
    beta_t1_s1 = bufferIn.slice({1, 0}, {sequenceLength + 1, batchSize});
    beta_t1_s2 = bufferIn.slice({2, 0}, {sequenceLength + 2, batchSize});
    tensorT = graph.addVariable(poplar::UNSIGNED_INT,
                                {tiles.size(), numWorkers}, "tensorT");
    mapOverTiles(graph, tiles, tensorT);
  }
  void map(Graph &graph, const Tensor &sequence, const Tensor &batchIndices,
           const Tensor &sliceIndices, const Tensor &diff,
           const Tensor &probs_t, Tensor &alpha, ComputeSet &computeSet) {
    mapVertex(graph,
              {{"beta_t1", beta_t1},
               {"out", bufferOut},
               {"beta_t1_s1", beta_t1_s1},
               {"beta_t1_s2", beta_t1_s2},
               {"seq", sequence},
               {"diff", diff},
               {"batchIndices", batchIndices},
               {"sliceIndices", sliceIndices}},
              {{"inputLengths", inputLengths_},
               {"targetLengths", targetLengths_},
               {"lpp_t", probs_t}},
              {}, {{"alpha", alpha}}, {{"t", tensorT}}, beta_t1.elementType(),
              computeSet, vertexName, graph.getTileMapping(beta_t1));
  }
  void init(Graph &graph, ComputeSet &computeSet) {
    initInfinity(graph, bufferIn, computeSet);
    popops::fill(graph, tensorT, graph.getTileMapping(tensorT), computeSet,
                 (unsigned int)maxInputLength_ - 1);
  }
  void copyBuffer(program::Sequence &loop) {
    loop.add(program::Copy(bufferOut, beta_t1));
  }
};

struct ContextAlpha {
  const std::string vertexName;
  Tensor bufferIn, bufferOut, alpha_t1, alpha_t1_s1, alpha_t1_s2, alpha_t;
  Tensor inputLengthsRep, targetLengthsRep;
  Tensor tensorT;

  ContextAlpha(Graph &graph, const std::vector<uint32_t> &tiles,
               poplar::Type type, poplar::Type probsType,
               poplar::Type sequenceType, size_t sequenceLength,
               size_t batchSize, const Tensor &inputLengths,
               const Tensor &targetLengths)
      : vertexName{poputil::templateVertex("CTCForwardVertex", type, probsType,
                                           sequenceType)} {
    const auto &target = graph.getTarget();
    const auto numWorkers = target.getNumWorkerContexts();
    bufferIn = graph.addVariable(type, {sequenceLength + 2, batchSize});
    poputil::mapTensorLinearly(graph, bufferIn, 1, 1);
    bufferOut = graph.addVariable(type, {sequenceLength + 2, batchSize});
    poputil::mapTensorLinearly(graph, bufferOut, 1, 1);
    alpha_t1 = bufferIn.slice({2, 0}, {sequenceLength + 2, batchSize});
    alpha_t1_s1 = bufferIn.slice({1, 0}, {sequenceLength + 1, batchSize});
    alpha_t1_s2 = bufferIn.slice({0, 0}, {sequenceLength, batchSize});
    alpha_t = bufferOut.slice({2, 0}, {sequenceLength + 2, batchSize});
    tensorT = graph.addVariable(poplar::UNSIGNED_INT,
                                {tiles.size(), numWorkers}, "tensorT");
    mapOverTiles(graph, tiles, tensorT);
    inputLengthsRep = replicate(graph, tiles, inputLengths);
    targetLengthsRep = replicate(graph, tiles, targetLengths);
  }
  void map(Graph &graph, const Tensor &sequence, const Tensor &batchIndices,
           const Tensor &sliceIndices, const Tensor &diff,
           const Tensor &probs_t, Tensor &loss, Tensor &alpha,
           ComputeSet &computeSet) {
    mapVertex(graph,
              {{"out", alpha_t},
               {"alpha_t1", alpha_t1},
               {"alpha_t1_s1", alpha_t1_s1},
               {"alpha_t1_s2", alpha_t1_s2},
               {"seq", sequence},
               {"batchIndices", batchIndices},
               {"sliceIndices", sliceIndices},
               {"diff", diff}},
              {{"lpp_t", probs_t}},
              {{"inputLengths", inputLengthsRep},
               {"targetLengths", targetLengthsRep}},
              {{"alpha", alpha}}, {{"loss", loss}, {"t", tensorT}},
              alpha_t1.elementType(), computeSet, vertexName,
              graph.getTileMapping(alpha_t.flatten()));
  }
  void init(Graph &graph, ComputeSet &computeSet) {
    initInfinity(graph, bufferIn, computeSet);
    popops::zero(graph, tensorT, graph.getTileMapping(tensorT), computeSet);
  }
  void copyBuffer(program::Sequence &loop) {
    loop.add(program::Copy(alpha_t, alpha_t1));
  }
};

void reverseMapping(Graph &graph, Tensor &t) {
  auto mapping = graph.getTileMapping(t);
  auto &target = graph.getTarget();
  auto numTiles = target.getTilesPerIPU();
  std::vector<std::vector<Interval>> res(numTiles);
  for (size_t i = 0; i < mapping.size(); ++i) {
    res[numTiles - 1 - i] = std::move(mapping[i]);
  }
  graph.setTileMapping(t, res);
}

void transposeMapping(Graph &graph, Tensor &t) {
  auto mapping = poputil::calcLinearTileMapping(graph, t.shape(), 1, t.dim(2));
  auto &target = graph.getTarget();
  auto numTiles = target.getTilesPerIPU();
  std::vector<std::vector<Interval>> res(numTiles);
  for (size_t i = 0; i < mapping.size(); ++i) {
    res[numTiles - 1 - i] = std::move(mapping[i]);
  }
  graph.setTileMapping(t, res);
}

Tensor computeAlpha(Graph &graph, const Tensor &probs, const Tensor &sequence,
                    const Tensor &diff, const Tensor &inputLengths,
                    const Tensor &targetLengths, const Tensor &batchIndices,
                    const Tensor &sliceIndices, Tensor &logAlpha, Tensor &loss,
                    program::Sequence &prog) {
  auto batchSize = probs.dim(2);
  auto inputLength = logAlpha.dim(0);
  auto sequenceLength = sequence.dim(0);

  const auto tiles = getTiles(graph.getTileMapping(logAlpha));
  const auto numTiles = tiles.size();
  ContextAlpha ctxt{graph,
                    tiles,
                    logAlpha.elementType(),
                    probs.elementType(),
                    sequence.elementType(),
                    sequenceLength,
                    batchSize,
                    inputLengths,
                    targetLengths};
  Tensor probs_ = probs.dimShuffle({1, 2, 0});
  poputil::mapTensorLinearly(graph, probs_, 1, probs.dim(0));
  probs_ = probs_.dimShuffle({2, 0, 1});
  reverseMapping(graph, probs_);
  Tensor alpha = logAlpha.flatten(1, 3);
  const auto &target = graph.getTarget();
  const auto numWorkers = target.getNumWorkerContexts();
  ComputeSet cs = graph.addComputeSet("InitAlpha");
  ctxt.init(graph, cs);
  popops::zero(graph, alpha, graph.getTileMapping(alpha), cs);
  prog.add(program::Execute(cs));
  program::Sequence loop;
  {
    Tensor probs_t =
        popops::dynamicSlice(graph, probs_, ctxt.tensorT[0][0].expand({0}), {0},
                             {1}, loop)
            .squeeze({0});
    ComputeSet computeSet = graph.addComputeSet("computeSetAlpha");
    ctxt.map(graph, sequence, batchIndices, sliceIndices, diff, probs_t, loss,
             alpha, computeSet);
    loop.add(program::Execute(computeSet));
    ctxt.copyBuffer(loop);
  }
  prog.add(program::Repeat(inputLength, loop));
  Tensor lossResult = loss.dimShuffle({3, 2, 0, 1})
                          .reshape({batchSize * 2, numTiles * numWorkers});
  return lossResult;
}

void updateT(Graph &graph, Tensor &t, program::Sequence &prog) {
  ComputeSet cs = graph.addComputeSet("UpdateT");
  VertexRef vtx = graph.addVertex(cs, "UpdateTVertex");
  graph.setPerfEstimate(vtx, 2);
  graph.connect(vtx["t"], t);
  graph.setTileMapping(vtx, 0);
  prog.add(program::Execute(cs));
}

Tensor computeAlphaBeta(Graph &graph, const Tensor &probs,
                        const Tensor &sequence, const Tensor &diff,
                        const Tensor &revDiff, const Tensor &inputLengths,
                        const Tensor &targetLengths, const Tensor &batchIndices,
                        const Tensor &sliceIndices, Tensor &logAlpha,
                        Tensor &loss, program::Sequence &prog) {
  auto batchSize = probs.dim(2);
  auto inputLength = logAlpha.dim(0);
  auto sequenceLength = sequence.dim(0);
  const auto tiles = getTiles(graph.getTileMapping(logAlpha));
  const auto numTiles = tiles.size();

  ContextAlpha ctxtA{graph,
                     tiles,
                     logAlpha.elementType(),
                     probs.elementType(),
                     sequence.elementType(),
                     sequenceLength,
                     batchSize,
                     inputLengths,
                     targetLengths};
  const auto mapping = graph.getTileMapping(logAlpha[0]);
  ContextBeta ctxtB{graph,
                    tiles,
                    logAlpha.elementType(),
                    probs.elementType(),
                    sequence.elementType(),
                    sequenceLength,
                    batchSize,
                    inputLengths,
                    targetLengths,
                    mapping,
                    inputLength};

  Tensor probs_ = probs.dimShuffle({1, 2, 0});
  poputil::mapTensorLinearly(graph, probs_, 1, inputLength);
  probs_ = probs_.dimShuffle({2, 0, 1});
  reverseMapping(graph, probs_);
  Tensor initialT = graph.addConstant(
      poplar::UNSIGNED_INT, {2, 1},
      poplar::ArrayRef<unsigned>({0, unsigned(inputLength - 1)}));
  graph.setTileMapping(initialT, 0);
  Tensor tensorT = graph.addVariable(poplar::UNSIGNED_INT, {2, 1}, "tensorT");
  graph.setTileMapping(tensorT, 0);
  prog.add(program::Copy(initialT, tensorT));

  Tensor alpha = logAlpha.flatten(1, 3);
  Tensor alpha1 = alpha.slice({0, 0}, {inputLength / 2, alpha.dim(1)});
  Tensor alpha2 =
      alpha.slice({inputLength / 2, 0}, {inputLength, alpha.dim(1)});

  const auto &target = graph.getTarget();
  const auto numWorkers = target.getNumWorkerContexts();

  ComputeSet cs = graph.addComputeSet("InitInfinityBuffer");
  ctxtA.init(graph, cs);
  popops::zero(graph, alpha, graph.getTileMapping(alpha), cs);
  ctxtB.init(graph, cs);
  prog.add(program::Execute(cs));
  Tensor flatT = tensorT.flatten();
  program::Sequence loop1;
  {
    Tensor probs_t = popops::multiSlice(graph, probs_, tensorT, {0}, {1}, loop1,
                                        popops::SlicePlan{}, OptionFlags{})
                         .squeeze({1});
    ComputeSet computeSet = graph.addComputeSet("computeSetAlphaBeta");
    ctxtA.map(graph, sequence, batchIndices, sliceIndices, diff, probs_t[0],
              loss, alpha1, computeSet);
    ctxtB.map(graph, sequence, batchIndices, sliceIndices, revDiff, probs_t[1],
              alpha2, computeSet);
    loop1.add(program::Execute(computeSet));
    ctxtA.copyBuffer(loop1);
    ctxtB.copyBuffer(loop1);
    updateT(graph, flatT, loop1);
  }
  prog.add(program::Repeat(inputLength / 2, loop1));

  program::Sequence loop2;
  {
    Tensor probs_t = popops::multiSlice(graph, probs_, tensorT, {0}, {1}, loop2,
                                        popops::SlicePlan{}, OptionFlags{})
                         .squeeze({1});
    ComputeSet computeSet = graph.addComputeSet("computeSetAlphaBeta");
    ctxtA.map(graph, sequence, batchIndices, sliceIndices, diff, probs_t[0],
              loss, alpha2, computeSet);
    ctxtB.map(graph, sequence, batchIndices, sliceIndices, revDiff, probs_t[1],
              alpha1, computeSet);
    loop2.add(program::Execute(computeSet));
    ctxtA.copyBuffer(loop2);
    ctxtB.copyBuffer(loop2);
    updateT(graph, flatT, loop2);
  }
  prog.add(program::Repeat(inputLength / 2, loop2));
  Tensor lossResult = loss.dimShuffle({3, 2, 0, 1})
                          .reshape({batchSize * 2, numTiles * numWorkers});
  return lossResult;
}

Tensor nllLoss(Graph &graph, const Tensor &loss, program::Sequence &prog,
               size_t batchSize) {
  Tensor lossArgMax = popnn::argMax(graph, loss, prog).reshape({batchSize, 2});
  Tensor loss_ = loss.reshape({batchSize, 2, loss.dim(1)});
  Tensor gatheredLoss =
      graph.addVariable(loss.elementType(), {batchSize, 2}, "nllLossGather");
  poputil::mapTensorLinearly(graph, gatheredLoss, 1, 1);
  ComputeSet computeSetGather = graph.addComputeSet("computeSetGatherNllLoss");
  size_t offset = 0;
  for (size_t i = 0; i < batchSize; ++i) {
    iterVertex(graph,
               {{"loss", loss_[i]},
                {"indice", lossArgMax[i]},
                {"out", gatheredLoss[i]}},
               2, computeSetGather,
               poputil::templateVertex("GatherLossVertex", loss.elementType()),
               1, offset);
    offset += 2;
  }
  prog.add(program::Execute(computeSetGather));
  Tensor result = graph.addVariable(loss.elementType(), {batchSize}, "nllLoss");
  poputil::mapTensorLinearly(graph, result, 1, 1);
  ComputeSet computeSet = graph.addComputeSet("computeSetNllLoss");
  iterVertex(graph, {{"loss", gatheredLoss}, {"out", result}}, batchSize,
             computeSet,
             poputil::templateVertex("NLLLossVertex", loss.elementType()), 2);
  prog.add(program::Execute(computeSet));
  return result;
}

void computeBeta(Graph &graph, Tensor &logAlpha, const Tensor &probs,
                 const Tensor &sequence, const Tensor &diff,
                 const Tensor &inputLengths, const Tensor &targetLengths,
                 const Tensor &batchIndices, const Tensor &sliceIndices,
                 program::Sequence &prog) {
  auto batchSize = probs.dim(2);
  auto maxInputLength = logAlpha.dim(0);
  auto sequenceLength = sequence.dim(0);

  const auto mapping = graph.getTileMapping(logAlpha[0]);
  const auto tiles = getTiles(graph.getTileMapping(logAlpha));
  ContextBeta ctxt{graph,
                   tiles,
                   logAlpha.elementType(),
                   probs.elementType(),
                   sequence.elementType(),
                   sequenceLength,
                   batchSize,
                   inputLengths,
                   targetLengths,
                   mapping,
                   maxInputLength};
  ComputeSet cs = graph.addComputeSet("InitBeta");
  ctxt.init(graph, cs);
  prog.add(program::Execute(cs));

  Tensor alpha = logAlpha.flatten(1, 3);

  program::Sequence loop;
  {
    Tensor probs_t =
        popops::dynamicSlice(graph, probs, ctxt.tensorT[0][0].expand({0}), {0},
                             {1}, loop)
            .squeeze({0});
    ComputeSet computeSet = graph.addComputeSet("computeSetBeta");
    ctxt.map(graph, sequence, batchIndices, sliceIndices, diff, probs_t, alpha,
             computeSet);
    loop.add(program::Execute(computeSet));
    ctxt.copyBuffer(loop);
  }
  prog.add(program::Repeat(maxInputLength, loop));
}

Tensor computeGrad(Graph &graph, Tensor &logAlpha, const Tensor &sequence,
                   const Tensor &inputLengths, const Tensor &targetLengths,
                   const Tensor &nll, const Tensor &probs,
                   const Tensor &gradOut, program::Sequence &prog) {
  const auto sequenceLength = sequence.dim(1);
  // grad in fp16 is not accurate enough
  auto grad = graph.addVariable(poplar::FLOAT, probs.shape(), "grad");
  Tensor grad_ = grad.dimShuffle({0, 2, 1});
  transposeMapping(graph, grad_);
  initInfinity(graph, prog, grad);
  const auto batchSize = probs.dim(2);
  const auto maxInputLength = grad.dim(0);
  Tensor alphaBeta =
      graph.addVariable(logAlpha.elementType(), logAlpha.shape(), "alphaCopy");
  Tensor alphaBeta_ = alphaBeta.dimShuffle({0, 2, 1});
  prog.add(program::Copy(logAlpha, alphaBeta));
  transposeMapping(graph, alphaBeta_);
  prog.add(program::WriteUndef(logAlpha));
  Tensor probsCopy =
      graph.addVariable(probs.elementType(), probs.shape(), "probsCopy");
  Tensor probs_ = probsCopy.dimShuffle({0, 2, 1});
  transposeMapping(graph, probs_);
  prog.add(program::Copy(probs, probsCopy));
  size_t start = 0;
  ComputeSet computeSet = graph.addComputeSet("computeSetGrad");
  const auto &target = graph.getTarget();
  const auto numTiles = target.getTilesPerIPU();
  const std::string vertexName =
      poputil::templateVertex("GradVertex", logAlpha.elementType(),
                              probs.elementType(), sequence.elementType());

  for (size_t t = 0; t < maxInputLength; ++t) {
    poplar::Tensor tensorT = graph.addConstant(
        poplar::UNSIGNED_INT, {}, poplar::ArrayRef<unsigned>({unsigned(t)}));
    graph.setTileMapping(tensorT, 0);
    iterVertex(graph,
               {{"grad", grad_[t]},
                {"lpp", probs_[t]},
                {"alphaBeta", alphaBeta_[t]},
                {"seq", sequence},
                {"targetLen", targetLengths},
                {"inputLen", inputLengths},
                {"gr", gradOut},
                {"nll", nll},
                {"t", tensorT}},
               batchSize, computeSet, vertexName, sequenceLength, start);
    start = (start + batchSize) % numTiles;
  }
  prog.add(program::Execute(computeSet));
  return grad;
}
