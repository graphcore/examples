// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include "ipu_utils.hpp"
#include <poputil/DebugInfo.hpp>

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nms(poplar::Graph &graph, program::Sequence &prog, const poplar::Tensor &scores,
    const poplar::Tensor &boxes, const poplar::Tensor &classes, float threshold,
    int num_detections, float score_threshold = 0.0, float sigma = 0.0f,
    bool useGather = false, const poplar::DebugContext &dc = {});

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
nms(poplar::Graph &graph, program::Sequence &prog, const poplar::Tensor &scores,
    const poplar::Tensor &boxes, float threshold, int num_detections,
    float score_threshold = 0.0, float sigma = 0.0f, bool useGather = false,
    const poplar::DebugContext &dc = {});

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor,
           poplar::Tensor>
nmsMulti(poplar::Graph &graph, program::Sequence &prog,
         const poplar::Tensor &scores, const poplar::Tensor &boxes,
         float threshold, int num_detections,
         float score_threshold = -std::numeric_limits<float>::max(),
         float sigma = 0.0f, bool useGather = false,
         const poplar::DebugContext &dc = {});
