// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "crop.hpp"
#include <poplar/DebugContext.hpp>
#include <popops/Zero.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

poplar::Tensor crop(poplar::Graph &graph, poplar::program::Sequence &prog,
                    const poplar::Tensor &input, float central_fraction,
                    const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(input, central_fraction));

  const auto input_dims = input.shape();
  assert(input_dims.size() == 4);
  auto output_dims = input_dims;
  // get the right dimension for cropped output
  size_t bbox_h_start = static_cast<size_t>(
      (input_dims[1] - input_dims[1] * central_fraction) / 2);
  size_t bbox_w_start = static_cast<size_t>(
      (input_dims[2] - input_dims[2] * central_fraction) / 2);
  size_t bbox_h_size = input_dims[1] - bbox_h_start * 2;
  size_t bbox_w_size = input_dims[2] - bbox_w_start * 2;

  output_dims[1] = static_cast<size_t>(bbox_h_size);
  output_dims[2] = static_cast<size_t>(bbox_w_size);

  // slice the input to get output
  poplar::Tensor output_to_copy =
      input.slice({0, bbox_h_start, bbox_w_start, 0},
                  {input_dims[0], input_dims[1] - bbox_h_start,
                   input_dims[2] - bbox_w_start, input_dims[3]});
  poplar::Tensor output =
      graph.addVariable(input.elementType(), output_dims,
                        {di, "dims_crop_" + std::to_string(output_dims[2])});
  ;
  poputil::mapTensorLinearly(graph, output);
  prog.add(poplar::program::Copy(output_to_copy, output));
  return output;
}

poplar::Tensor crop_grads(poplar::Graph &graph, poplar::program::Sequence &prog,
                          const poplar::Tensor &grad_output,
                          float central_fraction,
                          const poplar::DebugContext &dc) {
  poputil::PoplibsOpDebugInfo di(dc, DI_ARGS(grad_output, central_fraction));
  const auto grad_output_dims = grad_output.shape();
  assert(grad_output_dims.size() == 4);
  auto grad_input_dims = grad_output_dims;
  grad_input_dims[1] = grad_output_dims[1] / central_fraction;
  grad_input_dims[2] = grad_output_dims[2] / central_fraction;
  size_t bbox_h_start =
      static_cast<size_t>((grad_input_dims[1] - grad_output_dims[1]) / 2);
  size_t bbox_w_start =
      static_cast<size_t>((grad_input_dims[2] - grad_output_dims[1]) / 2);
  size_t bbox_h_size = grad_input_dims[1] - bbox_h_start * 2;
  size_t bbox_w_size = grad_input_dims[2] - bbox_w_start * 2;

  auto grad_input = graph.addVariable(
      grad_output.elementType(), grad_input_dims,
      {di, "gradients_crop_" + std::to_string(grad_input_dims[2])});
  poputil::mapTensorLinearly(graph, grad_input);

  popops::zero(graph, grad_input, prog);
  prog.add(poplar::program::Copy(
      grad_output,
      grad_input.slice({0, bbox_h_start, bbox_w_start, 0},
                       {grad_input_dims[0], grad_input_dims[1] - bbox_h_start,
                        grad_input_dims[2] - bbox_w_start,
                        grad_input_dims[3]})));

  return grad_input;
}
