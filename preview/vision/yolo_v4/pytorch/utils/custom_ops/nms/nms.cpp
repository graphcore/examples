// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "nms.hpp"
#include <poplar/VariableMappingMethod.hpp>
#include <popnn/Loss.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

namespace {
uint32_t
countRegions(const std::vector<std::vector<poplar::Interval>> &regions) {
  uint32_t res = 0;
  for (const auto &r : regions) {
    for (const auto &i : r) {
      res += i.size();
    }
  }
  return res;
}

std::vector<poplar::Interval>
scaleRegion(const std::vector<poplar::Interval> &region, uint32_t factor) {
  std::vector<poplar::Interval> res;
  for (const auto &i : region) {
    poplar::Interval newInterval{i.begin() * factor, i.end() * factor};
    res.push_back(newInterval);
  }
  return res;
}

// todo:
//  - full -> global
void mapTileVertex(
    Graph &graph,
    const std::unordered_map<std::string, std::pair<poplar::Tensor, uint32_t>>
        &aligned,
    const std::unordered_map<std::string, poplar::Tensor> &full,
    poplar::ComputeSet &computeSet, const std::string &vertexName,
    const std::vector<poplar::Interval> &regions, size_t index,
    uint32_t tileNumber, uint32_t splitSize,
    const std::function<void(VertexRef &, uint32_t, uint32_t)> &callback) {
  auto vertexRegions =
      poputil::splitRegionsBetweenWorkers(graph.getTarget(), regions, 1, 1);
  size_t j = 0;
  for (auto &r : vertexRegions) {
    VertexRef vtx = graph.addVertex(computeSet, vertexName);
    for (auto &p : full) {
      graph.connect(vtx[p.first], p.second);
    }
    for (auto &p : aligned) {
      uint32_t factor = p.second.second;
      const auto mapping = factor > 1 ? scaleRegion(r, factor) : r;
      graph.connect(vtx[p.first],
                    poplar::concat(p.second.first.slices(mapping)));
    }
    graph.setPerfEstimate(vtx, r.size()); // wrong ...
    graph.setTileMapping(vtx, tileNumber);
    callback(vtx, tileNumber, index);
    ++j;
  }
}

void mapTileMultiVertex(
    Graph &graph,
    const std::unordered_map<std::string, std::pair<poplar::Tensor, uint32_t>>
        &aligned,
    const std::unordered_map<std::string, poplar::Tensor> &full,
    poplar::ComputeSet &computeSet, const std::string &vertexName,
    const std::vector<poplar::Interval> &regions, size_t index,
    uint32_t tileNumber, uint32_t splitSize,
    const std::function<void(VertexRef &, uint32_t, uint32_t)> &callback) {
  VertexRef vtx = graph.addVertex(computeSet, vertexName);
  for (auto &p : full) {
    graph.connect(vtx[p.first], p.second);
  }
  for (auto &p : aligned) {
    uint32_t factor = p.second.second;
    const auto mapping = factor > 1 ? scaleRegion(regions, factor) : regions;
    graph.connect(vtx[p.first], poplar::concat(p.second.first.slices(mapping)));
  }
  graph.setPerfEstimate(vtx, regions.size()); // wrong ...
  graph.setTileMapping(vtx, tileNumber);
  callback(vtx, tileNumber, index);
}

void mapVertex(
    Graph &graph,
    const std::unordered_map<std::string, poplar::Tensor> &aligned,
    const std::unordered_map<std::string, poplar::Tensor> &full,
    poplar::Type elementType, poplar::ComputeSet &computeSet,
    const std::string &vertexName,
    const std::vector<std::vector<poplar::Interval>> &mapping,
    bool multiVertex = false,
    const std::function<void(VertexRef &, uint32_t, uint32_t)> &callback =
        [](VertexRef &, uint32_t, uint32_t) {}) {
  std::unordered_map<std::string, poplar::Tensor> flatten;
  uint32_t mappingSize = countRegions(mapping);
  std::unordered_map<std::string, std::pair<poplar::Tensor, uint32_t>> aligned_;
  for (auto &p : aligned) {
    size_t tensorSize = p.second.numElements();
    assert(tensorSize >= mappingSize);
    uint32_t factor = tensorSize / mappingSize;
    assert(mappingSize * factor == tensorSize);
    aligned_.insert({p.first, {p.second, factor}});
  }

  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(elementType);
  const auto splitSize =
      std::max<uint32_t>(vectorWidth, target.getAtomicStoreGranularity());
  const auto numTiles = target.getTilesPerIPU();
  size_t index = 0;
  for (size_t i = 0; i < numTiles; ++i) {
    auto &regions = mapping[i];
    if (regions.size() > 0) {
      if (multiVertex) {
        mapTileMultiVertex(graph, aligned_, full, computeSet, vertexName,
                           regions, index++, i, splitSize, callback);
      } else {
        mapTileVertex(graph, aligned_, full, computeSet, vertexName, regions,
                      index++, i, splitSize, callback);
      }
    }
  }
}

using Mapping = std::vector<std::vector<poplar::Interval>>;
std::vector<size_t> unflattenRegion(const Mapping &mapping, uint32_t tile,
                                    const std::vector<size_t> &shape) {
  const std::vector<Interval> &r = mapping[tile];
  assert(r.size() == 1);
  return poputil::unflattenIndex(shape, r.begin()->begin());
}

std::vector<Mapping> split_mapping(const Mapping &m, uint32_t partitions,
                                   uint32_t block_size) {
  if (partitions == 1) {
    return {m};
  }
  std::vector<Mapping> res(partitions);
  for (size_t i = 0; i < m.size(); ++i) {
    const std::vector<poplar::Interval> &m_i = m[i];
    const auto regions = poputil::splitRegions(m_i, block_size, partitions);
    for (size_t j = 0; j < regions.size(); ++j) {
      res[j].push_back(regions[j]);
    }
  }
  return res;
}

template <typename T>
void initializeTensor(poplar::Graph &graph, poplar::program::Sequence &program,
                      poplar::Tensor &t, T value) {
  poplar::Tensor v =
      graph.addConstant(t.elementType(), {1}, poplar::ArrayRef<T>({value}));
  graph.setTileMapping(v, 1);
  program.add(poplar::program::Copy(
      v.broadcast(t.numElements(), 0).reshape(t.shape()), t));
}

} // namespace

using namespace popops::expr;

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor, poplar::Tensor>
initGraphCommon(poplar::Graph &graph, program::Sequence &prog,
                const poplar::Tensor &scores, uint32_t numDetections,
                poplar::Type classType, const poplar::DebugContext &dc) {
  const size_t min_per_tile = 1;
  const size_t num_batches = scores.dim(0);
  std::vector<uint32_t> indicesUpdate;
  indicesUpdate.reserve(numDetections);
  for (size_t i = 0; i < numDetections; ++i) {
    indicesUpdate.push_back(i);
  }
  poplar::Tensor indicesUpdateT = graph.addConstant(
      poplar::UNSIGNED_INT, {numDetections}, ArrayRef<uint32_t>{indicesUpdate},
      {dc, "indicesUpdate"});
  poputil::mapTensorLinearly(graph, indicesUpdateT, min_per_tile, 1);
  poplar::Tensor lengths = graph.addVariable(
      poplar::INT, {num_batches, (numDetections)}, {dc, "lengths"});
  poputil::mapTensorLinearly(graph, lengths, 1, 1);
  initializeTensor<int32_t>(graph, prog, lengths, int32_t(numDetections));
  poplar::Tensor topBoxes = graph.addVariable(
      scores.elementType(), {num_batches, size_t(numDetections), 4},
      {dc, "topBoxes"});
  poputil::mapTensorLinearly(graph, topBoxes, 1, 4);
  initializeTensor<float>(graph, prog, topBoxes, 0.0);
  poplar::Tensor topScores = graph.addVariable(
      scores.elementType(), {num_batches, size_t(numDetections)},
      {dc, "topScores"});
  poputil::mapTensorLinearly(graph, topScores, 1, 1);
  initializeTensor<float>(graph, prog, topScores, 0.0);
  poplar::Tensor topIndices = graph.addVariable(
      poplar::INT, {num_batches, size_t(numDetections)}, {dc, "topIndices"});
  poputil::mapTensorLinearly(graph, topIndices, 1, 1);
  initializeTensor<int32_t>(graph, prog, topIndices, -1);
  poplar::Tensor topClasses = graph.addVariable(
      classType, {num_batches, size_t(numDetections)}, {dc, "topClasses"});
  poputil::mapTensorLinearly(graph, topClasses, 1, 1);
  initializeTensor<int32_t>(graph, prog, topClasses,
                            std::numeric_limits<int32_t>::max());

  return {topIndices, topScores, topBoxes, topClasses, indicesUpdateT, lengths};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor, poplar::Tensor, poplar::Tensor>
initGraphMulti(poplar::Graph &graph, program::Sequence &prog,
               const poplar::Tensor &scores, uint32_t numDetections,
               const poplar::DebugContext &dc) {
  const size_t min_per_tile = 1;
  const size_t C = scores.dim(2);
  poplar::Tensor scores_copy =
      graph.addVariable(scores.elementType(), scores.shape());
  poputil::mapTensorLinearly(graph, scores_copy, min_per_tile, C);
  prog.add(program::Copy(scores, scores_copy));

  poplar::Tensor topIndices, topScores, topBoxes, topClasses, indicesUpdateT,
      lengths;
  std::tie(topIndices, topScores, topBoxes, topClasses, indicesUpdateT,
           lengths) =
      initGraphCommon(graph, prog, scores, numDetections, poplar::INT, dc);

  return {scores_copy, topIndices,     topScores, topBoxes,
          topClasses,  indicesUpdateT, lengths};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor, poplar::Tensor, poplar::Tensor>
initGraph(poplar::Graph &graph, program::Sequence &prog,
          const poplar::Tensor &scores, const poplar::Tensor &classes,
          uint32_t numDetections, const poplar::DebugContext &dc) {
  const size_t min_per_tile = 1;
  poplar::Tensor scores_copy =
      graph.addVariable(scores.elementType(), scores.shape());
  poputil::mapTensorLinearly(graph, scores_copy, min_per_tile, 1);
  prog.add(program::Copy(scores, scores_copy));
  poplar::Tensor topIndices, topScores, topBoxes, topClasses, indicesUpdateT,
      lengths;
  std::tie(topIndices, topScores, topBoxes, topClasses, indicesUpdateT,
           lengths) = initGraphCommon(graph, prog, scores, numDetections,
                                      classes.elementType(), dc);

  return {scores_copy, topIndices,     topScores, topBoxes,
          topClasses,  indicesUpdateT, lengths};
}

poplar::Tensor gather(poplar::Graph &graph, program::Sequence &prog,
                      const poplar::Tensor &t, const poplar::Tensor &indices,
                      const poplar::DebugContext &dc) {
  std::vector<poplar::Tensor> slices;
  size_t batchSize = t.dim(0);
  for (size_t b = 0; b < batchSize; ++b) {
    poplar::Tensor subTensor = t[b];
    poplar::Tensor slice_b =
        popops::dynamicSlice(graph, subTensor, indices[b].expand({0}), {0}, {1},
                             prog, {dc, "gather"});
    slices.push_back(slice_b);
  }
  return poplar::concat(slices);
}

std::pair<poplar::Tensor, size_t>
createGatherTensor(poplar::Graph &graph, const Mapping &mapping,
                   const poplar::Tensor &output) {
  size_t nbTiles = 0;
  for (const auto &r : mapping) {
    if (r.size() > 0)
      ++nbTiles;
  }
  poplar::Tensor res =
      graph.addVariable(output.elementType(), {nbTiles, output.numElements()});
  size_t i = 0;
  for (size_t tile = 0; tile < mapping.size(); ++tile) {
    if (mapping[tile].size() > 0) {
      graph.setTileMapping(res[i++], tile);
    }
  }
  return {res, nbTiles};
}

void mapReduceGather(
    poplar::Graph &graph, program::Sequence &prog, const Mapping &mapping,
    const std::vector<size_t> &shape, poplar::Tensor &output,
    poplar::Tensor &outputClass, const std::string &vertexName,
    const poplar::Tensor &index,
    const std::unordered_map<std::string, poplar::Tensor> &aligned,
    bool withClasses, const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(output));
  poplar::Tensor gatherTensor;
  size_t firstDim;
  size_t batchSize = shape[0];
  std::tie(gatherTensor, firstDim) = createGatherTensor(graph, mapping, output);
  gatherTensor = gatherTensor.reshape({firstDim, batchSize, 4});
  initializeTensor(graph, prog, gatherTensor, 0.0);
  poplar::Tensor gatherTensorClass;
  size_t firstDimClass;
  std::tie(gatherTensorClass, firstDimClass) =
      createGatherTensor(graph, mapping, outputClass);
  initializeTensor<int>(graph, prog, gatherTensorClass, 0);
  gatherTensorClass = gatherTensorClass.reshape({firstDimClass, batchSize});
  poplar::ComputeSet cs = graph.addComputeSet({di, "mapReduceGather"});
  mapVertex(graph, aligned, {}, output.elementType(), cs, vertexName, mapping,
            true, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
              std::vector<size_t> indices =
                  unflattenRegion(mapping, tile, shape);
              size_t batch = indices[0], indice = indices[1];
              graph.connect(vtx["index"], index[batch]);
              graph.connect(vtx["output"], gatherTensor[tile][batch]);
              graph.connect(vtx["outputClass"], gatherTensorClass[tile][batch]);
              graph.setInitialValue(vtx["offset"], indice);
            });
  prog.add(program::Execute(cs));
  popops::ReduceParams params(popops::Operation::ADD);
  poplar::Tensor min = popops::reduce(
      graph, gatherTensor.reshape({firstDim, output.numElements()}),
      output.elementType(), {0}, params, prog, {di, "mapReduceGatherReduce"},
      {{"accumType.interTile", "float"}, {"accumType.inVertex", "float"}});
  prog.add(program::Copy(min, output.flatten()));
  if (!withClasses)
    return;
  if (outputClass.elementType() == poplar::UNSIGNED_INT) {
    gatherTensorClass = gatherTensorClass.reinterpret(poplar::INT);
  }
  poplar::Tensor minClass = popops::reduce(
      graph,
      gatherTensorClass.reshape({firstDimClass, outputClass.numElements()}),
      poplar::INT, {0}, params, prog, {di, "mapReduceGatherReduceClass"});
  if (outputClass.elementType() == poplar::UNSIGNED_INT) {
    minClass = popops::cast(graph, minClass, poplar::UNSIGNED_INT, prog,
                            {di, "castBestClassUInt"});
  }
  prog.add(program::Copy(minClass, outputClass.flatten()));
}

void mapReduceGatherMulti(
    poplar::Graph &graph, program::Sequence &prog, const Mapping &mapping,
    const std::vector<size_t> &shape, poplar::Tensor &output,
    const uint32_t numClasses, const std::string &vertexName,
    const poplar::Tensor &index,
    const std::unordered_map<std::string, poplar::Tensor> &aligned,
    const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(output));
  poplar::Tensor gatherTensor;
  size_t firstDim;
  size_t batchSize = shape[0];
  std::tie(gatherTensor, firstDim) = createGatherTensor(graph, mapping, output);
  initializeTensor(graph, prog, gatherTensor, 0.0);
  gatherTensor = gatherTensor.reshape({firstDim, batchSize, 4});
  poplar::ComputeSet cs = graph.addComputeSet({di, "mapReduceGather"});
  mapVertex(graph, aligned, {}, output.elementType(), cs, vertexName, mapping,
            true, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
              std::vector<size_t> indices =
                  unflattenRegion(mapping, tile, shape);
              size_t batch = indices[0], indice = indices[1];
              graph.connect(vtx["index"], index[batch]);
              graph.connect(vtx["output"], gatherTensor[tile][batch]);
              graph.setInitialValue(vtx["C"], numClasses);
              graph.setInitialValue(vtx["offset"], indice);
            });
  prog.add(program::Execute(cs));
  popops::ReduceParams params(popops::Operation::ADD);
  poplar::Tensor min = popops::reduce(
      graph, gatherTensor.reshape({firstDim, output.numElements()}),
      output.elementType(), {0}, params, prog, {di, "mapReduceGatherReduce"});
  prog.add(program::Copy(min, output.flatten()));
}

poplar::Tensor gather(poplar::Graph &graph, program::Sequence &prog,
                      const poplar::Tensor &t, const poplar::Tensor &indices,
                      const poplar::Tensor &indices0,
                      const poplar::DebugContext &dc) {
  std::vector<poplar::Tensor> slices;
  size_t batchSize = t.dim(0);
  for (size_t b = 0; b < batchSize; ++b) {
    poplar::Tensor subTensor = t[b];
    poplar::Tensor slice_b =
        popops::dynamicSlice(graph, subTensor, indices[b].expand({0}), {0}, {1},
                             prog, {dc, "gather"});
    slices.push_back(slice_b);
  }
  return poplar::concat(slices);
}

std::tuple<program::Execute, poplar::Tensor, poplar::Tensor, poplar::Tensor>
condition(poplar::Graph &graph, program::Sequence &prog,
          const poplar::Tensor &scores, uint32_t numIterations,
          float scoreThreshold, const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(numIterations));
  poplar::Tensor bestScores = graph.addVariable(
      scores.elementType(), {scores.dim(0)}, {di, "bestScores"});
  graph.setTileMapping(bestScores, 1);
  poplar::Tensor zeroF =
      graph.addConstant(scores.elementType(), {1}, 0.0, {di, "zeroF"});
  graph.setTileMapping(zeroF, 1);
  prog.add(
      program::Copy(zeroF.broadcast(bestScores.numElements(), 0), bestScores));
  poplar::Tensor iterT = graph.addConstant(
      poplar::UNSIGNED_INT, {}, numIterations, {di, "numIterations"});
  poplar::Tensor predicate = graph.addVariable(poplar::BOOL, {});
  graph.setTileMapping(predicate, 1);
  graph.setTileMapping(iterT, 1);
  poplar::Tensor zero =
      graph.addConstant(poplar::UNSIGNED_INT, {}, 0, {di, "zero"});
  graph.setTileMapping(zero, 1);
  poplar::Tensor i = graph.addVariable(poplar::UNSIGNED_INT, {});
  graph.setTileMapping(i, 1);
  prog.add(program::Copy(zero, i, false, {di, "initializeI"}));
  poplar::ComputeSet cs = graph.addComputeSet({di, "condition"});
  // call vertex
  VertexRef vtx = graph.addVertex(
      cs, poputil::templateVertex("ConditionVertex", bestScores.elementType()));
  graph.connect(vtx["bestScores"], bestScores);
  graph.connect(vtx["condition"], predicate);
  graph.connect(vtx["i"], i);
  graph.connect(vtx["numIter"], iterT);
  graph.setPerfEstimate(vtx, bestScores.numElements() + 1);
  graph.setTileMapping(vtx, 1);
  graph.setInitialValue(vtx["score_threshold"], scoreThreshold);
  return {program::Execute(cs), bestScores, predicate, i};
}

void connectSliceVertex(poplar::Graph &graph, ComputeSet &cs,
                        const poplar::Tensor &input,
                        const poplar::Tensor &index,
                        const poplar::Tensor &output) {
  const auto vertexName =
      poputil::templateVertex("SliceVertex", input.elementType());
  VertexRef vtx = graph.addVertex(cs, vertexName);
  size_t tile = getTile(graph, input);
  graph.connect(vtx["input"], input);
  graph.connect(vtx["index"], index);
  graph.connect(vtx["output"], output);
  graph.setTileMapping(vtx, tile);
  graph.setPerfEstimate(vtx, 1);
}

void connectSliceMultiVertex(poplar::Graph &graph, ComputeSet &cs,
                             const poplar::Tensor &input,
                             const poplar::Tensor &index,
                             const poplar::Tensor &output,
                             uint32_t numClasses) {
  const auto vertexName =
      poputil::templateVertex("SliceMultiVertex", input.elementType());
  VertexRef vtx = graph.addVertex(cs, vertexName);
  size_t tile = getTile(graph, input);
  graph.connect(vtx["input"], input);
  graph.connect(vtx["index"], index);
  graph.connect(vtx["output"], output);
  graph.setInitialValue(vtx["C"], numClasses);
  graph.setTileMapping(vtx, tile);
  graph.setPerfEstimate(vtx, 1);
}

struct NMSContext {
  // references
  poplar::Graph &graph_;
  // returned tensors
  poplar::Tensor topIndices, topScores, topBoxes, topClasses, lengths;
  // temporary tensors
  poplar::Tensor scoresCopy, indices, indicesUpdate;
  poplar::Tensor bestBox, bestClass;
  poplar::Tensor bestScoresCond, predicate, i;
  program::Program cond;
  // constants
  poplar::Tensor one;
  // options
  float iouThreshold_, scoreThreshold_, sigma_;
  int numDetections_;
  bool withClasses_;
  bool useGather_;
  poplar::Tensor boxesGather, classesGather;
  size_t batchSize;
  size_t N;

  NMSContext(poplar::Graph &graph, float iouThreshold, float scoreThreshold,
             float sigma, int numDetections, const poplar::DebugContext &dc,
             bool withClasses = true, bool useGather = false)
      : graph_{graph}, iouThreshold_{iouThreshold},
        scoreThreshold_{scoreThreshold}, sigma_{sigma},
        numDetections_{numDetections}, withClasses_{withClasses},
        useGather_{useGather} {
    one = graph.addConstant(poplar::UNSIGNED_INT, {}, 1, {dc, "one"});
    graph.setTileMapping(one, 1);
  }
  void init(program::Sequence &prog, const poplar::Tensor &scores,
            const poplar::Tensor &boxes, const poplar::Tensor &classes,
            const poplar::DebugContext &dc) {
    batchSize = scores.dim(0);
    N = scores.dim(1);
    std::tie(scoresCopy, topIndices, topScores, topBoxes, topClasses,
             indicesUpdate, lengths) =
        initGraph(graph_, prog, scores, classes, numDetections_, dc);
    std::tie(cond, bestScoresCond, predicate, i) =
        condition(graph_, prog, scores, numDetections_, scoreThreshold_, dc);
    bestBox = graph_.addVariable(boxes.elementType(), {batchSize, 4},
                                 poplar::VariableMappingMethod::LINEAR);
    if (withClasses_) {
      bestClass = graph_.addVariable(classes.elementType(), {batchSize},
                                     poplar::VariableMappingMethod::LINEAR);
    } else {
      bestClass = graph_.addConstant(classes.elementType(), {batchSize}, 0,
                                     "dummyBestClass");
      poputil::mapTensorLinearly(graph_, bestClass);
    }
    if (useGather_) {
      boxesGather = graph_.addVariable(topBoxes.elementType(),
                                       {batchSize, 4, N}, {dc, "boxesGather"});
      poputil::mapTensorLinearlyWithOffset(graph_, boxesGather, 1, N, 0, false);
      prog.add(
          program::Copy(boxes, boxesGather.dimShufflePartial({1, 2}, {2, 1})));
      size_t offset = batchSize * 4;
      if (withClasses_) {
        offset += batchSize;
        classesGather = graph_.addVariable(
            classes.elementType(), {batchSize, N}, {dc, "classesGather"});
        poputil::mapTensorLinearlyWithOffset(graph_, classesGather, 1, N,
                                             offset, false);
        prog.add(program::Copy(classes, classesGather));
      }
    }
  }

  void gatherBoxes(program::Sequence &loop, const poplar::Tensor &best_idx,
                   const poplar::Tensor &boxes, const poplar::Tensor &classes,
                   const poplar::DebugContext &dc) {
    if (useGather_) {
      poplar::ComputeSet cs = graph_.addComputeSet({dc, "gather"});
      for (size_t b = 0; b < batchSize; ++b) {
        // gather boxes
        connectSliceVertex(graph_, cs, boxesGather[b][0], best_idx[b],
                           bestBox[b][0]);
        connectSliceVertex(graph_, cs, boxesGather[b][1], best_idx[b],
                           bestBox[b][1]);
        connectSliceVertex(graph_, cs, boxesGather[b][2], best_idx[b],
                           bestBox[b][2]);
        connectSliceVertex(graph_, cs, boxesGather[b][3], best_idx[b],
                           bestBox[b][3]);
        if (withClasses_) {
          connectSliceVertex(graph_, cs, classesGather[b], best_idx[b],
                             bestClass[b]);
        }
      }
      loop.add(program::Execute(cs));
    } else {
      mapReduceGather(
          graph_, loop, graph_.getTileMapping(scoresCopy), scoresCopy.shape(),
          bestBox, bestClass,
          poputil::templateVertex("GatherVertex", scoresCopy.elementType(),
                                  bestClass.elementType()),
          best_idx,
          {{"classes", classes.flatten()}, {"boxes", boxes.flatten()}},
          withClasses_, {dc, "mapReduce"});
    }
  }

  void updateState(program::Sequence &loop, const poplar::Tensor &best_idx,
                   const poplar::Tensor &best_score,
                   const poplar::DebugContext &dc) {
    const auto mapping = graph_.getTileMapping(scoresCopy);
    poplar::ComputeSet cs = graph_.addComputeSet({dc, "updateBest"});
    mapVertex(
        graph_, {{"scores", scoresCopy.flatten()}}, {},
        scoresCopy.elementType(), cs,
        poputil::templateVertex("UpdateBestVertex", scoresCopy.elementType()),
        mapping, true, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
          std::vector<size_t> indices =
              unflattenRegion(mapping, tile, scoresCopy.shape());
          size_t batch = indices[0], indice = indices[1];
          graph_.connect(vtx["best"], best_idx[batch]);
          graph_.setInitialValue(vtx["offset"], indice);
        });
    loop.add(program::Execute(cs));

    poplar::ComputeSet csTop = graph_.addComputeSet({dc, "updateAnswer"});
    mapVertex(graph_,
              {{"lengths", lengths.flatten()},
               {"top_indices", topIndices.flatten()},
               {"top_scores", topScores.flatten()},
               {"top_boxes", topBoxes.flatten()},
               {"top_classes", topClasses.flatten()}},
              {{"i", i}}, scoresCopy.elementType(), csTop,
              poputil::templateVertex("UpdateAnswerVertex",
                                      scoresCopy.elementType(),
                                      bestClass.elementType()),
              graph_.getTileMapping(topIndices.flatten()), true,
              [&](VertexRef &vtx, uint32_t tile, uint32_t index) {
                const std::vector<size_t> indices =
                    unflattenRegion(graph_.getTileMapping(topIndices.flatten()),
                                    tile, topIndices.shape());
                const size_t batch = indices[0];
                graph_.setInitialValue(vtx["score_threshold"], scoreThreshold_);
                graph_.setInitialValue(vtx["K"], numDetections_);
                graph_.setInitialValue(vtx["offset"], indices[1]);
                graph_.connect(vtx["best_indices"], best_idx[batch]);
                graph_.connect(vtx["best_scores"], best_score[batch]);
                graph_.connect(vtx["best_boxes"], bestBox[batch]);
                graph_.connect(vtx["best_classes"], bestClass[batch]);
              });
    loop.add(program::Execute(csTop));
  }
  void nms(program::Sequence &loop, const poplar::Tensor &boxes,
           const poplar::Tensor &classes, const poplar::DebugContext &dc) {
    poplar::ComputeSet csNms = graph_.addComputeSet({dc, "Nms"});
    const auto mapping = graph_.getTileMapping(scoresCopy);
    mapVertex(graph_,
              {{"scores", scoresCopy.flatten()},
               {"classes", classes.flatten()},
               {"boxes", boxes.flatten()}},
              {}, scoresCopy.elementType(), csNms,
              poputil::templateVertex("NmsVertex", scoresCopy.elementType(),
                                      bestClass.elementType()),
              mapping, true, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
                std::vector<size_t> indices =
                    unflattenRegion(mapping, tile, scoresCopy.shape());
                size_t batch = indices[0];
                graph_.connect(vtx["bestClass"], bestClass[batch]);
                graph_.connect(vtx["bestBox"], bestBox[batch]);
                graph_.setInitialValue(vtx["sigma"], sigma_);
                graph_.setInitialValue(vtx["threshold"], iouThreshold_);
                graph_.setInitialValue(vtx["score_threshold"], scoreThreshold_);
              });
    loop.add(program::Execute(csNms));
  }
};

// class-less version
// for now with a dummy classes tensor.
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
nms(poplar::Graph &graph, program::Sequence &prog, const poplar::Tensor &scores,
    const poplar::Tensor &boxes, float threshold, int num_detections,
    float score_threshold, float sigma, bool useGather,
    const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(
      dc, DI_ARGS(scores, boxes, threshold, num_detections));
  assert(boxes.rank() == 3);
  assert(scores.rank() == 2);
  assert(threshold > 0.0);
  assert(num_detections > 0);
  assert(num_detections <= int(scores.dim(1)));
  const size_t min_per_tile = 1;
  poputil::mapTensorLinearly(graph, scores, min_per_tile, 1);
  poplar::Tensor classes = graph.addConstant(poplar::UNSIGNED_INT,
                                             scores.shape(), 0, "dummyClasses");
  poputil::mapTensorLinearly(graph, classes, min_per_tile, 1);
  poputil::mapTensorLinearly(graph, boxes, min_per_tile, 4);
  NMSContext context{graph,          threshold, score_threshold, sigma,
                     num_detections, di,        false,           useGather};
  context.init(prog, scores, boxes, classes, di);

  program::Sequence loop;
  poplar::Tensor best_score, best_idx;
  std::tie(best_score, best_idx) =
      popnn::maxAndArgMax(graph, context.scoresCopy, loop, {di, "maxArgMax"});
  loop.add(program::Copy(best_score, context.bestScoresCond));

  context.gatherBoxes(loop, best_idx, boxes, classes, di);
  context.updateState(loop, best_idx, best_score, di);
  context.nms(loop, boxes, classes, di);
  popops::addInPlace(graph, context.i, context.one, loop, {di, "incrementI"});
  prog.add(program::RepeatWhileTrue(context.cond, context.predicate, loop));
  popops::ReduceParams params(popops::Operation::MIN);
  poplar::Tensor lengths_ =
      popops::reduce(graph, context.lengths, poplar::INT, {1}, params, prog,
                     {di, "reduceLengths"});
  return {context.topIndices, context.topScores, context.topBoxes, lengths_};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nms(poplar::Graph &graph, program::Sequence &prog, const poplar::Tensor &scores,
    const poplar::Tensor &boxes, const poplar::Tensor &classes, float threshold,
    int num_detections, float score_threshold, float sigma, bool useGather,
    const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(
      dc, DI_ARGS(scores, boxes, classes, threshold, num_detections));
  assert(boxes.rank() == 3);
  assert(scores.rank() == 2);
  assert(classes.rank() == 2);
  assert(threshold > 0.0);
  assert(num_detections > 0);
  assert(num_detections <= int(scores.dim(1)));
  const size_t min_per_tile = 1;
  poputil::mapTensorLinearly(graph, scores, min_per_tile, 1);
  poputil::mapTensorLinearly(graph, classes, min_per_tile, 1);
  poputil::mapTensorLinearly(graph, boxes, min_per_tile, 4);
  NMSContext context{graph, threshold, score_threshold, sigma, num_detections,
                     di,    true,      useGather};
  context.init(prog, scores, boxes, classes, di);

  program::Sequence loop;
  poplar::Tensor best_score, best_idx;
  std::tie(best_score, best_idx) =
      popnn::maxAndArgMax(graph, context.scoresCopy, loop, {di, "maxArgMax"});
  loop.add(program::Copy(best_score, context.bestScoresCond));

  context.gatherBoxes(loop, best_idx, boxes, classes, di);
  context.updateState(loop, best_idx, best_score, di);
  context.nms(loop, boxes, classes, di);

  popops::addInPlace(graph, context.i, context.one, loop, {di, "incrementI"});
  prog.add(program::RepeatWhileTrue(context.cond, context.predicate, loop));
  popops::ReduceParams params(popops::Operation::MIN);
  poplar::Tensor lengths_ =
      popops::reduce(graph, context.lengths, poplar::INT, {1}, params, prog,
                     {di, "reduceLengths"});

  return {context.topIndices, context.topScores, context.topBoxes,
          context.topClasses, lengths_};
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nmsMulti(poplar::Graph &graph, program::Sequence &prog,
         const poplar::Tensor &scores, const poplar::Tensor &boxes,
         float threshold, int num_detections, float score_threshold,
         float sigma, bool useGather, const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(
      dc, DI_ARGS(scores, boxes, threshold, num_detections));
  assert(boxes.rank() == 3);
  assert(scores.rank() == 3);
  assert(threshold > 0.0);
  assert(num_detections > 0);
  assert(num_detections <= int(scores.dim(1)));
  const size_t min_per_tile = 1;
  const size_t bs = scores.dim(0);
  const size_t N = scores.dim(1);
  const size_t C = scores.dim(2);
  poputil::mapTensorLinearly(graph, scores, min_per_tile, C);
  poputil::mapTensorLinearly(graph, boxes, min_per_tile, 4);
  poplar::Tensor dummy = graph.addVariable(boxes.elementType(), {bs, N});
  poputil::mapTensorLinearly(graph, dummy, min_per_tile, 1);
  const auto mapping = graph.getTileMapping(dummy);
  const auto shape = dummy.shape();
  poplar::Tensor boxesGather;
  poplar::Tensor scores_copy, topIndices, topScores, topBoxes, topClasses,
      indicesUpdate, lengths;
  std::tie(scores_copy, topIndices, topScores, topBoxes, topClasses,
           indicesUpdate, lengths) =
      initGraphMulti(graph, prog, scores, num_detections, di);
  if (useGather) {
    boxesGather = graph.addVariable(topBoxes.elementType(), {bs, 4, N},
                                    {di, "boxesGather"});
    poputil::mapTensorLinearlyWithOffset(graph, boxesGather, 1, N, 0, false);
    prog.add(
        program::Copy(boxes, boxesGather.dimShufflePartial({1, 2}, {2, 1})));
  }

  poplar::Tensor one =
      graph.addConstant(poplar::UNSIGNED_INT, {}, 1, {di, "one"});
  graph.setTileMapping(one, 1);
  program::Program cond;
  poplar::Tensor bestScores, predicate, i;
  std::tie(cond, bestScores, predicate, i) =
      condition(graph, prog, scores, num_detections, score_threshold, di);
  program::Sequence loop;
  poplar::Tensor best_score, best_idx;
  std::tie(best_score, best_idx) = popnn::maxAndArgMax(
      graph, scores_copy.reshape({bs, N * C}), loop, {di, "maxArgMax"});
  loop.add(program::Copy(best_score, bestScores));
  poplar::Tensor best_box =
      graph.addVariable(boxes.elementType(), {scores.dim(0), 4},
                        poplar::VariableMappingMethod::LINEAR);
  if (useGather) {
    poplar::ComputeSet csGather = graph.addComputeSet({dc, "gather"});
    for (size_t b = 0; b < bs; ++b) {
      // gather boxes
      connectSliceMultiVertex(graph, csGather, boxesGather[b][0], best_idx[b],
                              best_box[b][0], C);
      connectSliceMultiVertex(graph, csGather, boxesGather[b][1], best_idx[b],
                              best_box[b][1], C);
      connectSliceMultiVertex(graph, csGather, boxesGather[b][2], best_idx[b],
                              best_box[b][2], C);
      connectSliceMultiVertex(graph, csGather, boxesGather[b][3], best_idx[b],
                              best_box[b][3], C);
    }
    loop.add(program::Execute(csGather));
  } else {
    mapReduceGatherMulti(
        graph, loop, graph.getTileMapping(dummy), dummy.shape(), best_box, C,
        poputil::templateVertex("GatherMultiVertex", scores.elementType()),
        best_idx, {{"boxes", boxes.flatten()}}, {di, "mapReduceBoxes"});
  }
  poplar::ComputeSet cs = graph.addComputeSet({di, "updateBest"});
  mapVertex(
      graph, {{"scores", scores_copy.flatten()}}, {}, scores.elementType(), cs,
      poputil::templateVertex("UpdateBestMultiVertex", scores.elementType()),
      mapping, true, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
        std::vector<size_t> indices = unflattenRegion(mapping, tile, shape);
        size_t batch = indices[0], indice = indices[1];
        graph.connect(vtx["best"], best_idx[batch]);
        graph.setInitialValue(vtx["offset"], indice);
        graph.setInitialValue(vtx["C"], C);
      });
  loop.add(program::Execute(cs));

  poplar::ComputeSet csTop = graph.addComputeSet({di, "updateAnswer"});
  mapVertex(
      graph,
      {{"top_indices", topIndices.flatten()},
       {"lengths", lengths.flatten()},
       {"top_scores", topScores.flatten()},
       {"top_boxes", topBoxes.flatten()},
       {"top_classes", topClasses.flatten()}},
      {{"i", i}}, scores.elementType(), csTop,
      poputil::templateVertex("UpdateAnswerMultiVertex", scores.elementType()),
      graph.getTileMapping(topIndices.flatten()), true,
      [&](VertexRef &vtx, uint32_t tile, uint32_t index) {
        const std::vector<size_t> indices =
            unflattenRegion(graph.getTileMapping(topIndices.flatten()), tile,
                            topIndices.shape());
        const size_t batch = indices[0];
        graph.setInitialValue(vtx["score_threshold"], score_threshold);
        graph.setInitialValue(vtx["K"], num_detections);
        graph.setInitialValue(vtx["C"], C);
        graph.setInitialValue(vtx["offset"], indices[1]);
        graph.connect(vtx["best_indices"], best_idx[batch]);
        graph.connect(vtx["best_scores"], best_score[batch]);
        graph.connect(vtx["best_boxes"], best_box[batch]);
      });
  loop.add(program::Execute(csTop));

  poplar::ComputeSet csNms = graph.addComputeSet({di, "Nms"});
  mapVertex(
      graph, {{"boxes", boxes.flatten()}, {"scores", scores_copy.flatten()}},
      {}, scores.elementType(), csNms,
      poputil::templateVertex("NmsMultiVertex", scores.elementType()), mapping,
      true, [&](VertexRef &vtx, uint32_t tile, uint32_t) {
        std::vector<size_t> indices = unflattenRegion(mapping, tile, shape);
        size_t batch = indices[0];
        graph.connect(vtx["bestBox"], best_box[batch]);
        graph.connect(vtx["bestIndices"], best_idx[batch]);
        graph.setInitialValue(vtx["sigma"], sigma);
        graph.setInitialValue(vtx["threshold"], threshold);
        graph.setInitialValue(vtx["C"], C);
        graph.setInitialValue(vtx["score_threshold"], score_threshold);
      });
  loop.add(program::Execute(csNms));

  popops::addInPlace(graph, i, one, loop, {di, "incrementI"});
  prog.add(program::RepeatWhileTrue(cond, predicate, loop));
  popops::ReduceParams params(popops::Operation::MIN);
  poplar::Tensor lengths_ = popops::reduce(graph, lengths, poplar::INT, {1},
                                           params, prog, {di, "reduceLengths"});

  return {topIndices, topScores, topBoxes, topClasses, lengths_};
}
