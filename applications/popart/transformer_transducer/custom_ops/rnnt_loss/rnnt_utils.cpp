// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "rnnt_utils.hpp"
#include "ipu_utils.hpp"
#include "poplar/Program.hpp"
#include <popnn/Loss.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <tuple>
#include <unordered_map>

poplar::Tensor initBatchIndices(poplar::Graph &graph, uint32_t batchSize,
                                uint32_t T, uint32_t U) {
  std::vector<uint16_t> bsRange;
  for (size_t i = 0; i < batchSize; ++i)
    bsRange.push_back(i);
  poplar::ArrayRef<uint16_t> bsArrayRef(bsRange);
  poplar::Tensor batchIndices = graph.addConstant(
      poplar::UNSIGNED_SHORT, {batchSize, 1}, bsArrayRef, "BatchSizeIndices");
  poplar::Tensor result =
      batchIndices.broadcast(T * U, 1).reshape({batchSize, T, U});
  return result;
}

poplar::Tensor initTIndices(poplar::Graph &graph, uint32_t batchSize,
                            uint32_t T, uint32_t U) {
  std::vector<uint16_t> sliceRange;
  for (size_t i = 0; i < T; ++i)
    sliceRange.push_back(i);
  poplar::ArrayRef<uint16_t> sliceArrayRef(sliceRange);
  poplar::Tensor sliceIndices = graph.addConstant(
      poplar::UNSIGNED_SHORT, {1, T, 1}, sliceArrayRef, "TIndices");
  poplar::Tensor result = sliceIndices.broadcast(batchSize, 0).broadcast(U, 2);
  return result;
}

poplar::Tensor initUIndices(poplar::Graph &graph, uint32_t batchSize,
                            uint32_t T, uint32_t U) {
  std::vector<uint16_t> sliceRange;
  for (size_t i = 0; i < U; ++i)
    sliceRange.push_back(i);
  poplar::ArrayRef<uint16_t> sliceArrayRef(sliceRange);
  poplar::Tensor sliceIndices = graph.addConstant(
      poplar::UNSIGNED_SHORT, {1, 1, U}, sliceArrayRef, "TIndices");
  poplar::Tensor result = sliceIndices.broadcast(batchSize, 0).broadcast(T, 1);
  return result;
}

void mapTileVertex(
    poplar::Graph &graph,
    const std::unordered_map<std::string, poplar::Tensor> &flat,
    const std::unordered_map<std::string, poplar::Tensor> &full,
    const std::unordered_map<std::string, poplar::Tensor> &indices,
    const std::unordered_map<std::string, poplar::Tensor> &tiles,
    const std::unordered_map<std::string, poplar::Tensor> &workers,
    poplar::ComputeSet &computeSet, const std::string &vertexName,
    const std::vector<poplar::Interval> &regions, uint16_t index,
    uint32_t tileNumber, uint32_t splitSize) {
  auto vertexRegions =
      poputil::splitRegionsBetweenWorkers(graph.getTarget(), regions, 1, 1);
  size_t j = 0;
  for (auto &r : vertexRegions) {
    poplar::VertexRef vtx = graph.addVertex(computeSet, vertexName);
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
        std::vector<poplar::Tensor> tensors;
        for (const auto &t : r) {
          tensors.emplace_back(p.second.slice({0, t.begin()}, {dim0, t.end()}));
        }
        graph.connect(vtx[p.first], poplar::concat(tensors));
      } else {
        const auto &interval = *r.begin();
        poplar::Tensor dest =
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

void mapVertex(poplar::Graph &graph,
               const std::unordered_map<std::string, poplar::Tensor> &flat,
               const std::unordered_map<std::string, poplar::Tensor> &full,
               const std::unordered_map<std::string, poplar::Tensor> &indices,
               const std::unordered_map<std::string, poplar::Tensor> &tiles,
               const std::unordered_map<std::string, poplar::Tensor> &workers,
               poplar::Type elementType, poplar::ComputeSet &computeSet,
               const std::string &vertexName,
               const std::vector<std::vector<poplar::Interval>> &mapping) {
  std::unordered_map<std::string, poplar::Tensor> flatten;
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

void maplosses(poplar::Graph &graph, poplar::program::Sequence &program,
               const poplar::Tensor &log_probs, const poplar::Tensor &log_alpha,
               const poplar::Tensor &input_lengths,
               const poplar::Tensor &label_lengths, poplar::Tensor &losses) {
  size_t batch_size = log_probs.dim(0);
  size_t U = log_probs.dim(2);

  auto probs = log_probs.dimShuffle({3, 0, 2, 1})[0];
  auto alpha = log_alpha.dimShuffle({0, 2, 1});
  auto losses_ =
      graph.addVariable(log_alpha.elementType(), {batch_size, U}, "map_losses");
  poputil::mapTensorLinearly(graph, losses_, 1, 1);
  poplar::ComputeSet computeSet = graph.addComputeSet("alphaLoss");
  for (size_t i = 0; i < batch_size; ++i) {
    for (int u = 0; u < static_cast<int>(U); ++u) {
      poplar::Tensor uT =
          graph.addConstant(poplar::INT, {}, poplar::ArrayRef<int>({u}));
      graph.setTileMapping(uT, 0);
      poplar::VertexRef vtx = graph.addVertex(
          computeSet,
          poputil::templateVertex("MapLossVertex", probs.elementType()));
      graph.connect(vtx["input_len"], input_lengths[i]);
      graph.connect(vtx["label_len"], label_lengths[i]);
      graph.connect(vtx["alphas"], alpha[i][u]);
      graph.connect(vtx["probs"], probs[i][u]);
      graph.connect(vtx["out"], losses_[i][u]);
      graph.connect(vtx["u"], uT);
      graph.setPerfEstimate(vtx, 1);
      graph.setTileMapping(vtx, getTile(graph, losses_[i][u]));
    }
  }
  program.add(poplar::program::Execute(computeSet));
  popops::ReduceParams params{popops::Operation::MAX};
  popops::reduceWithOutput(graph, losses_, losses, {1}, params, program);
}

poplar::Tensor losses(poplar::Graph &graph, poplar::program::Sequence &program,
                      const poplar::Tensor &log_beta) {
  return popops::neg(graph, log_beta.dimShuffle({1, 2, 0})[0][0], program,
                     "NegLoss");
}

void initializeTensor(poplar::Graph &graph, poplar::program::Sequence &program,
                      poplar::Tensor &t, float value) {
  poplar::Tensor v =
      graph.addConstant(t.elementType(), {1}, poplar::ArrayRef<float>({value}));
  graph.setTileMapping(v, 1);
  program.add(poplar::program::Copy(
      v.broadcast(t.numElements(), 0).reshape(t.shape()), t));
}

void grads(poplar::Graph &graph, poplar::program::Sequence &program,
           const poplar::Tensor &log_probs, const poplar::Tensor &input_lengths,
           const poplar::Tensor &label_lengths, const poplar::Tensor &alphas,
           const poplar::Tensor &betas, const poplar::Tensor &losses,
           poplar::Tensor &compactedGrads) {
  size_t batch_size = log_probs.dim(0);
  size_t maxT = log_probs.dim(1);
  size_t maxU = log_probs.dim(2);
  initializeTensor(graph, program, compactedGrads, 0.0f);
  poplar::ComputeSet computeSet = graph.addComputeSet("gradients");
  for (size_t t = 0; t < maxT; t++) {
    for (size_t u = 0; u < maxU; u++) {
      for (size_t i = 0; i < batch_size; ++i) {
        if (t < maxT - 1) {
          auto vtx =
              connectVertex(graph, computeSet,
                            poputil::templateVertex("GradientsTVertex",
                                                    log_probs.elementType()),
                            {{"out", compactedGrads[i][t][u][0]},
                             {"prob", log_probs[i][t][u][0]},
                             {"loss", losses[i]},
                             {"alpha_T_U", alphas[i][t][u]},
                             {"beta_T1_U", betas[i][t + 1][u]},
                             {"label_len", label_lengths[i]},
                             {"input_len", input_lengths[i]}},
                            getTile(graph, compactedGrads[i][t][u][0]));
          graph.setInitialValue(vtx["u"], u);
          graph.setInitialValue(vtx["t"], t);
        } else {
          auto vtx =
              connectVertex(graph, computeSet,
                            poputil::templateVertex("Gradients0Vertex",
                                                    log_probs.elementType()),
                            {{"out", compactedGrads[i][t][u][0]},
                             {"prob", log_probs[i][t][u][0]},
                             {"loss", losses[i]},
                             {"alpha_T_U", alphas[i][t][u]},
                             {"label_len", label_lengths[i]},
                             {"input_len", input_lengths[i]}},
                            getTile(graph, compactedGrads[i][t][u][0]));
          graph.setInitialValue(vtx["u"], u);
          graph.setInitialValue(vtx["t"], t);
        }
        if (u < maxU - 1) {
          auto vtx =
              connectVertex(graph, computeSet,
                            poputil::templateVertex("GradientsUVertexCompact",
                                                    log_probs.elementType()),
                            {{"out", compactedGrads[i][t][u][1]},
                             {"prob", log_probs[i][t][u][1]},
                             {"loss", losses[i]},
                             {"alpha_T_U", alphas[i][t][u]},
                             {"beta_T_U1", betas[i][t][u + 1]},
                             {"label_len", label_lengths[i]},
                             {"input_len", input_lengths[i]}},
                            getTile(graph, compactedGrads[i][t][u][1]));
          graph.setInitialValue(vtx["u"], u);
          graph.setInitialValue(vtx["t"], t);
        }
      }
    }
  }
  program.add(poplar::program::Execute(computeSet));
}

void computeBetaSlice(poplar::Graph &graph, size_t start, size_t end, size_t tp,
                      size_t batch_size, size_t U, size_t T,
                      const poplar::Tensor &log_probs,
                      const poplar::Tensor &input_lengths,
                      const poplar::Tensor &label_lengths,
                      poplar::Tensor &log_beta,
                      poplar::ComputeSet &computeSet) {
  for (size_t z = start; z <= end; z++) { // this loop is parallel
    size_t u = z;
    size_t t = tp - z;
    if (t == T - 1) {
      if (u == U - 1) {
        for (size_t i = 0; i < batch_size; ++i) {
          auto out = log_beta[i][t][u];
          size_t tile = getTile(graph, out);
          auto vtx = connectVertex(
              graph, computeSet,
              poputil::templateVertex("BetaTUVertex", log_probs.elementType()),
              {{"out", out},
               {"label_len", label_lengths[i]},
               {"input_len", input_lengths[i]},
               {"prob_0", log_probs[i][t][u][0]}},
              tile);
          graph.setInitialValue(vtx["u"], u);
          graph.setInitialValue(vtx["t"], t);
        }
      } else {
        for (size_t i = 0; i < batch_size; ++i) {
          auto out = log_beta[i][t][u];
          size_t tile = getTile(graph, out);
          auto vtx =
              connectVertex(graph, computeSet,
                            poputil::templateVertex("BetaTVertexCompact",
                                                    log_probs.elementType()),
                            {{"out", out},
                             {"label_len", label_lengths[i]},
                             {"input_len", input_lengths[i]},
                             {"beta_T_U1", log_beta[i][t][u + 1]},
                             {"probs_T_U", log_probs[i][t][u]}},
                            tile);
          graph.setInitialValue(vtx["u"], u);
          graph.setInitialValue(vtx["t"], t);
        }
      }

    } else {
      if (u == U - 1) {
        for (size_t i = 0; i < batch_size; ++i) {
          auto out = log_beta[i][t][u];
          size_t tile = getTile(graph, out);
          auto vtx = connectVertex(
              graph, computeSet,
              poputil::templateVertex("BetaUVertex", log_probs.elementType()),
              {{"out", out},
               {"label_len", label_lengths[i]},
               {"input_len", input_lengths[i]},
               {"beta_T1_U", log_beta[i][t + 1][u]},
               {"prob_0", log_probs[i][t][u][0]}},
              tile);
          graph.setInitialValue(vtx["u"], u);
          graph.setInitialValue(vtx["t"], t);
        }
      } else {
        for (size_t i = 0; i < batch_size; ++i) {
          auto out = log_beta[i][t][u];
          size_t tile = getTile(graph, out);
          auto vtx =
              connectVertex(graph, computeSet,
                            poputil::templateVertex("BetaVertexCompact",
                                                    log_probs.elementType()),
                            {{"out", out},
                             {"label_len", label_lengths[i]},
                             {"input_len", input_lengths[i]},
                             {"beta_T1_U", log_beta[i][t + 1][u]},
                             {"beta_T_U1", log_beta[i][t][u + 1]},
                             {"probs_T_U", log_probs[i][t][u]}},
                            tile);
          graph.setInitialValue(vtx["u"], u);
          graph.setInitialValue(vtx["t"], t);
        }
      }
    }
  }
}

float infinity(poplar::Type t) {
  if (t == poplar::FLOAT)
    return -std::numeric_limits<float>::infinity();
  return -65504.0f;
}

void beta(poplar::Graph &graph, poplar::program::Sequence &program,
          const poplar::Tensor &log_probs, const poplar::Tensor &input_lengths,
          const poplar::Tensor &label_lengths, poplar::Tensor &log_beta) {
  size_t batch_size = log_probs.dim(0);
  size_t maxT = log_probs.dim(1);
  size_t maxU = log_probs.dim(2);
  size_t U = maxU;
  size_t T = maxT;
  auto probs_0 = log_probs.dimShuffle({3, 0, 1, 2})[0];
  initializeTensor(graph, program, log_beta, infinity(log_beta.elementType()));

  for (int tp = T + U - 2; tp >= 0; --tp) {
    poplar::ComputeSet computeSet =
        graph.addComputeSet("beta" + std::to_string(tp));
    size_t start = size_t(std::max(tp - int(T) + 1, 0));
    size_t end = std::min(U - 1, size_t(tp)); // inclusive end
    computeBetaSlice(graph, start, end, tp, batch_size, U, T, log_probs,
                     input_lengths, label_lengths, log_beta, computeSet);
    program.add(poplar::program::Execute(computeSet));
  }
}
void computeAlphaSlice(poplar::Graph &graph, size_t start, size_t end,
                       size_t tp, size_t batch_size,
                       const poplar::Tensor &log_probs,
                       const poplar::Tensor &input_lengths,
                       const poplar::Tensor &label_lengths,
                       poplar::Tensor &log_alpha,
                       poplar::ComputeSet &computeSet) {
  for (size_t z = start; z <= end; z++) { // this loop is parallel
    size_t u = z;
    size_t t = tp - z;
    for (size_t i = 0; i < batch_size; ++i) {
      auto out = log_alpha[i][t][u];
      size_t tile = getTile(graph, out);
      if (t == 0) {
        if (u == 0) {
          connectVertex(
              graph, computeSet,
              poputil::templateVertex("AlphaZeroVertex", out.elementType()),
              {{"out", out}}, tile);
        } else {
          auto alpha = log_alpha[i][0][u - 1];
          auto p = log_probs[i][0][u - 1][1];
          auto vtx = connectVertex(
              graph, computeSet,
              poputil::templateVertex("AlphaUVertex", p.elementType()),
              {{"out", out},
               {"alpha", alpha},
               {"prob", p},
               {"label_len", label_lengths[i]}},
              tile);
          graph.setInitialValue(vtx["u"], u);
        }
      } else {
        if (u == 0) {
          auto alpha = log_alpha[i][t - 1][0];
          auto p = log_probs[i][t - 1][0][0];
          auto vtx = connectVertex(
              graph, computeSet,
              poputil::templateVertex("AlphaT0Vertex", p.elementType()),
              {{"out", out},
               {"alpha", alpha},
               {"prob", p},
               {"input_len", input_lengths[i]}},
              tile);
          graph.setInitialValue(vtx["t"], t);
        } else {
          auto alpha1 = log_alpha[i][t - 1][u];
          auto p1 = log_probs[i][t - 1][u][0];
          auto alpha2 = log_alpha[i][t][u - 1];
          auto p2 = log_probs[i][t][u - 1][1]; // vector[label]
          connectVertex(
              graph, computeSet,
              // should check that u <=
              // label_len and t < input_len
              poputil::templateVertex("AlphaVertexCompact", p1.elementType()),
              {{"out", out},
               {"alpha_1", alpha1},
               {"alpha_2", alpha2},
               {"prob_1", p1},
               {"prob_2", p2}},
              tile);
        }
      }
    }
  }
}

void alpha(poplar::Graph &graph, poplar::program::Sequence &program,
           const poplar::Tensor &log_probs, const poplar::Tensor &input_lengths,
           const poplar::Tensor &label_lengths, poplar::Tensor &log_alpha) {
  size_t batch_size = log_probs.dim(0);
  size_t maxT = log_probs.dim(1);
  size_t maxU = log_probs.dim(2);
  size_t U = maxU;
  size_t T = maxT;
  initializeTensor(graph, program, log_alpha,
                   infinity(log_alpha.elementType()));
  for (size_t tp = 0; tp < T + U - 1; tp++) {
    size_t start = size_t(std::max(int(tp) - int(T) + 1, 0));
    size_t end = std::min(U - 1, tp); // inclusive end
    poplar::ComputeSet computeSet =
        graph.addComputeSet("alpha" + std::to_string(tp));
    computeAlphaSlice(graph, start, end, tp, batch_size, log_probs,
                      input_lengths, label_lengths, log_alpha, computeSet);
    program.add(poplar::program::Execute(computeSet));
  }
}

void alpha_beta(poplar::Graph &graph, poplar::program::Sequence &program,
                const poplar::Tensor &log_probs,
                const poplar::Tensor &input_lengths,
                const poplar::Tensor &label_lengths, poplar::Tensor &log_alpha,
                poplar::Tensor &log_beta) {
  size_t batch_size = log_probs.dim(0);
  size_t maxT = log_probs.dim(1);
  size_t maxU = log_probs.dim(2);
  size_t U = maxU;
  size_t T = maxT;
  initializeTensor(graph, program, log_alpha,
                   infinity(log_alpha.elementType()));
  initializeTensor(graph, program, log_beta, infinity(log_beta.elementType()));
  auto probs_0 = log_probs.dimShuffle({3, 0, 1, 2})[0];
  for (size_t tpA = 0; tpA < T + U - 1; tpA++) {
    size_t tpB = T + U - 2 - tpA;
    size_t startA = size_t(std::max(int(tpA) - int(T) + 1, 0));
    size_t endA = std::min(U - 1, tpA); // inclusive end
    poplar::ComputeSet computeSet =
        graph.addComputeSet("alpha_beta" + std::to_string(tpA));
    size_t startB = size_t(std::max(int(tpB) - int(T) + 1, 0));
    size_t endB = std::min(U - 1, tpB); // inclusive end
    computeBetaSlice(graph, startB, endB, tpB, batch_size, U, T, log_probs,
                     input_lengths, label_lengths, log_beta, computeSet);
    computeAlphaSlice(graph, startA, endA, tpA, batch_size, log_probs,
                      input_lengths, label_lengths, log_alpha, computeSet);
    program.add(poplar::program::Execute(computeSet));
  }
}

void compactProbs(poplar::Graph &graph, poplar::program::Sequence &program,
                  const poplar::Tensor &log_probs,
                  poplar::Tensor &compacted_probs, const poplar::Tensor &labels,
                  const poplar::Tensor &label_lengths) {
  size_t batch_size = log_probs.dim(0);
  size_t maxT = log_probs.dim(1);
  size_t maxU = log_probs.dim(2);
  initializeTensor(graph, program, compacted_probs,
                   infinity(compacted_probs.elementType()));
  poplar::ComputeSet computeSet = graph.addComputeSet("compaction");
  for (size_t t = 0; t < maxT; ++t) {
    for (size_t u = 0; u < maxU; ++u) {
      for (size_t i = 0; i < batch_size; ++i) {
        const size_t tile = getTile(graph, log_probs[i][t][u]);
        connectVertex(graph, computeSet,
                      poputil::templateVertex("CopyVertex",
                                              log_probs.elementType(),
                                              compacted_probs.elementType()),
                      {{"out", compacted_probs[i][t][u][0]},
                       {"in", log_probs[i][t][u][0]}},
                      tile);
        if (u < maxU - 1) {
          auto vtx =
              connectVertex(graph, computeSet,
                            poputil::templateVertex("CopyIndexVertex",
                                                    log_probs.elementType()),
                            {{"out", compacted_probs[i][t][u][1]},
                             {"in", log_probs[i][t][u]},
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

poplar::Tensor expandGradients(poplar::Graph &graph,
                               poplar::program::Sequence &program,
                               const poplar::Tensor &compacted_gradients,
                               const poplar::Tensor &log_probs,
                               const poplar::Tensor &labels,
                               const poplar::Tensor &label_lengths) {
  size_t batch_size = log_probs.dim(0);
  size_t maxT = log_probs.dim(1);
  size_t maxU = log_probs.dim(2);
  size_t alphabet = log_probs.dim(3);
  poplar::Tensor gradients = graph.addVariable(
      log_probs.elementType(), {batch_size, maxT, maxU, alphabet}, "Gradients");
  poputil::mapTensorLinearly(graph, gradients, 1, alphabet);
  poplar::Tensor compactedGradientsFinal =
      graph.addVariable(compacted_gradients.elementType(),
                        {batch_size, maxT, maxU, 2}, "compactedGradients_");
  poputil::mapTensorLinearly(graph, compactedGradientsFinal, 1, 2);
  program.add(
      poplar::program::Copy(compacted_gradients, compactedGradientsFinal));
  program.add(poplar::program::WriteUndef(compacted_gradients));
  initializeTensor(graph, program, gradients, 0.0f);

  poplar::ComputeSet computeSet = graph.addComputeSet("expansion");
  for (size_t t = 0; t < maxT; ++t) {
    for (size_t u = 0; u < maxU; ++u) {
      poplar::Tensor uT =
          graph.addConstant(poplar::UNSIGNED_INT, {},
                            poplar::ArrayRef<unsigned int>({(unsigned int)u}));
      graph.setTileMapping(uT, 1);
      for (size_t i = 0; i < batch_size; ++i) {
        const size_t tile = getTile(graph, log_probs[i][t][u]);
        connectVertex(graph, computeSet,
                      poputil::templateVertex(
                          "CopyVertex", compactedGradientsFinal.elementType(),
                          gradients.elementType()),
                      {{"out", gradients[i][t][u][0]},
                       {"in", compactedGradientsFinal[i][t][u][0]}},
                      tile);
        if (u < maxU - 1) {
          auto vtx = connectVertex(
              graph, computeSet,
              poputil::templateVertex("CopyExpandVertex",
                                      compactedGradientsFinal.elementType(),
                                      gradients.elementType()),
              {{"out", gradients[i][t][u].slice({1, alphabet})},
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
  return gradients;
}
