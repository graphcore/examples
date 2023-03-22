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
#include <poplar/Graph.hpp>
#include <poputil/exceptions.hpp>

extern "C" {

/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
int32_t custom_op_api_level = 5;

/// Set the properties of the forward op.
void Build_metadata(
    std::vector<std::int64_t> &allocating_indices,
    std::vector<std::int64_t> &replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t> &input_to_output_tensor_aliasing,
    bool &is_elementwise, bool &is_stateless, bool &is_hashable,
    std::uint32_t num_inputs) {

  // The forward op is just a function of its inputs (no internal state)
  // so it can be marked as stateless.
  is_stateless = true;
  is_hashable = true;
}

/// Define the forward op
poplar::program::Program Build(poplar::Graph &graph,
                               const std::vector<poplar::Tensor> &inputs,
                               std::vector<poplar::Tensor> &outputs,
                               const std::string &attributes,
                               const std::string &debug_prefix) {

  if (inputs.size() != 1) {
    throw poputil::poplibs_error("product requires 1 input.");
  }

  auto input = inputs[0];
  if (input.rank() != 4) {
    throw poputil::poplibs_error("Input matrix rank must be 4.");
  }
  poplar::program::Sequence prog;
  const float central_fraction = std::stof(attributes);
  poplar::Tensor result = crop(graph, prog, input, central_fraction);

  outputs.push_back(result);

  return prog;
}

/// The gradient op requires its own metadata. Since it does not have any
/// internal state we can mark the op as stateless.
/// For stateless ops only one instance of the op is compiled even when
/// we ask for the gradient multiple times (e.g. we use tf.gradients() in
/// the python code).
void Build_grad_metadata(
    std::vector<std::int64_t> &allocating_indices,
    std::vector<std::int64_t> &replica_identical_output_indices,
    std::map<std::int64_t, std::int64_t> &input_to_output_tensor_aliasing,
    bool &is_elementwise, bool &is_stateless, bool &is_hashable,
    std::uint32_t num_inputs) {

  is_stateless = true;
  is_hashable = true;
}

/// Define the gradient op.
poplar::program::Program
Build_grad(poplar::Graph &graph, int input_grad_index,
           const std::vector<poplar::Tensor> &gradients,
           const std::vector<poplar::Tensor> &fwd_inputs,
           const std::vector<poplar::Tensor> &fwd_outputs,
           std::vector<poplar::Tensor> &outputs, const std::string &attributes,
           const std::string &debug_prefix) {

  poplar::program::Sequence prog;
  const float central_fraction = std::stof(attributes);
  auto gradOfLossWrtInput =
      crop_grads(graph, prog, gradients[0], central_fraction);

  outputs.push_back(gradOfLossWrtInput);

  return prog;
}

} // end extern "C"
