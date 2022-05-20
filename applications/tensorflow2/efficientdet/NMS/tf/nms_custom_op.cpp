// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "nms.hpp"
#include "poplar/DebugContext.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <poplar/Graph.hpp>
#include <popops/Cast.hpp>
#include <poputil/exceptions.hpp>

extern "C" {
int32_t custom_op_api_level = 5;
}

namespace {} // namespace

extern "C" void Build_metadata(
    std::vector<std::int64_t> &allocating_indices,
    std::vector<std::int64_t> &replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t> &input_to_output_tensor_aliasing,
    bool &is_elementwise, bool &is_stateless, bool &is_hashable,
    std::uint32_t num_inputs) {
  is_elementwise = false;
  is_hashable = true;
  is_stateless = true;
}

extern "C" poplar::program::Program Build(poplar::Graph &graph,
                                          std::vector<poplar::Tensor> &inputs,
                                          std::vector<poplar::Tensor> &outputs,
                                          const std::string &attributes,
                                          const std::string &debug_prefix) {
  if (inputs.size() != 3 && inputs.size() != 2) {
    throw poputil::poplibs_error("NMS requires at least 2 inputs.");
  }
  poplar::program::Sequence seq;
  poplar::Tensor output;

  std::stringstream json_args(attributes);
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(json_args, pt);

  const bool useGather = pt.get<bool>("useGather", false);
  const bool inPlace = pt.get<bool>("inPlace", false);
  const int num_detections = pt.get<int>("numDetections", 100);
  assert(num_detections > 0);
  const float threshold = pt.get<float>("threshold", 0.5f);
  const float score_threshold = pt.get<float>("scoreThreshold", 0.0f);
  const float sigma = pt.get<float>("sigma", 0.0f);
  const auto &scores = inputs[0];
  const auto &boxes = inputs[1];
  if (scores.rank() != 2)
    throw poputil::poplibs_error("Scores need 3 dims.");
  if (boxes.rank() != 3)
    throw poputil::poplibs_error("Boxes need 4 dims.");
  if (inputs.size() == 3) {
    const auto &classes = inputs[2];
    if (classes.rank() != 2)
      throw poputil::poplibs_error("Classes need 3 dims.");

    poputil::PoplibsOpDebugInfo di(
        debug_prefix, DI_ARGS(scores, boxes, classes, num_detections, threshold,
                              score_threshold));
    poplar::Tensor indicesAns, scoresAns, boxesAns, classesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengthsAns) =
        nms(graph, seq, scores, boxes, classes, threshold, num_detections,
            score_threshold, sigma, useGather, inPlace, debug_prefix);
    outputs.push_back(indicesAns);
    outputs.push_back(scoresAns);
    outputs.push_back(boxesAns);
    outputs.push_back(classesAns);
    outputs.push_back(lengthsAns);
  } else {
    poputil::PoplibsOpDebugInfo di(
        debug_prefix,
        DI_ARGS(scores, boxes, num_detections, threshold, score_threshold));
    poplar::Tensor indicesAns, scoresAns, boxesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, lengthsAns) =
        nms(graph, seq, scores, boxes, threshold, num_detections,
            score_threshold, sigma, useGather, inPlace);
    outputs.push_back(indicesAns);
    outputs.push_back(scoresAns);
    outputs.push_back(boxesAns);
    outputs.push_back(lengthsAns);
  }
  return seq;
}
