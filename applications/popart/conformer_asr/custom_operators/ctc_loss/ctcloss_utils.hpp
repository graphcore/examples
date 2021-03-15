// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once
#include <poplar/Graph.hpp>

#include <unordered_map>

using namespace poplar;

Tensor initBatchIndices(Graph &graph, const Tensor &targets,
                        uint32_t sequenceLength);
Tensor initSlicesIndices(Graph &graph, const Tensor &targets,
                         uint32_t sequenceLength);
void initInfinity(Graph &graph, Tensor &t, ComputeSet &cs);
void initInfinity(Graph &graph, program::Sequence &prog, Tensor &t);
void iterVertex(Graph &graph,
                const std::unordered_map<std::string, Tensor> &indexed,
                size_t range, ComputeSet &computeSet,
                const std::string &vertexName, size_t estimate,
                size_t start = 0);
void mapTileVertex(Graph &graph,
                   const std::unordered_map<std::string, Tensor> &flat,
                   const std::unordered_map<std::string, Tensor> &full,
                   const std::unordered_map<std::string, Tensor> &tiles,
                   const std::unordered_map<std::string, Tensor> &workers,
                   ComputeSet &computeSet, const std::string &vertexName,
                   std::vector<Interval> &regions, uint32_t tileNumber,
                   uint32_t splitSize);
void mapVertex(Graph &graph,
               const std::unordered_map<std::string, Tensor> &flat,
               const std::unordered_map<std::string, Tensor> &full,
               const std::unordered_map<std::string, Tensor> &tiles,
               const std::unordered_map<std::string, Tensor> &workers,
               poplar::Type elementType, ComputeSet &computeSet,
               const std::string &vertexName,
               const std::vector<std::vector<Interval>> &mapping);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
ctcLossPrepare(Graph &graph, const Tensor &targets, const Tensor &targetLengths,
               const poplar::Type probsType, ComputeSet &cs, bool partial32,
               size_t inputLength);
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
ctcLossPrepare(Graph &graph, const Tensor &targets, const Tensor &targetLengths,
               const poplar::Type probsType, program::Sequence &prog,
               bool partial32, size_t inputLength);
Tensor computeAlpha(Graph &graph, const Tensor &probs, const Tensor &sequence,
                    const Tensor &diff, const Tensor &inputLengths,
                    const Tensor &targetLengths, const Tensor &batchIndices,
                    const Tensor &sliceIndices, Tensor &logAlpha, Tensor &loss,
                    program::Sequence &prog);
Tensor computeAlphaBeta(Graph &graph, const Tensor &probs,
                        const Tensor &sequence, const Tensor &diff,
                        const Tensor &revDiff, const Tensor &inputLengths,
                        const Tensor &targetLengths, const Tensor &batchIndices,
                        const Tensor &sliceIndices, Tensor &logAlpha,
                        Tensor &loss, program::Sequence &prog);
Tensor nllLoss(Graph &graph, const Tensor &loss, program::Sequence &prog,
               size_t batchSize);
void computeBeta(Graph &graph, Tensor &logAlpha, const Tensor &probs,
                 const Tensor &sequence, const Tensor &diff,
                 const Tensor &inputLengths, const Tensor &targetLengths,
                 const Tensor &batchIndices, const Tensor &sliceIndices,
                 program::Sequence &prog);
Tensor computeGrad(Graph &graph, Tensor &logAlpha, const Tensor &sequence,
                   const Tensor &inputLengths, const Tensor &targetLengths,
                   const Tensor &nll, const Tensor &probs,
                   const Tensor &gradOut, program::Sequence &prog);
