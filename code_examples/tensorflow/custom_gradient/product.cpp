// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/exceptions.hpp>

/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
extern "C" {
  int32_t custom_op_api_level = 4;
}

/// Set the properties of the forward op.
extern "C"
void Build_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {

  // The forward op is just a function of its inputs (no internal state)
  // so it can be marked as stateless.
  is_stateless = true;
}

/// Define the forward op
extern "C" poplar::program::Program Build(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs, const std::string& debug_prefix) {

  if (inputs.size() != 2) {
    throw poputil::poplibs_error("product requires 2 inputs.");
  }

  auto input = inputs[0];
  auto weights = inputs[1];
  if (input.rank() != 2 && weights.rank() != 2) {
    throw poputil::poplibs_error("Both inputs must be matrices.");
  }

  if (input.dim(1) != weights.dim(0)) {
    throw poputil::poplibs_error("Product shapes incompatible.");
  }

  poplar::program::Sequence prog;
  auto result = poplin::matMul(graph, input, weights, prog,
                               debug_prefix + "/product");
  outputs.push_back(result);

  return prog;
}

/// The gradient op requires its own metadata. Since it does not have any
/// internal state we can mark the op as stateless.
/// For stateless ops only one instance of the op is compiled even when
/// we ask for the gradient multiple times (e.g. we use tf.gradients() in
/// the python code).
extern "C"
void Build_grad_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {

  is_stateless = true;
}

/// Define the gradient op.
extern "C"
poplar::program::Program Build_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& debug_prefix) {

  poplar::program::Sequence prog;
  auto inputsTransposed = fwd_inputs[0].dimShuffle({1, 0});
  auto weightsTransposed = fwd_inputs[1].dimShuffle({1, 0});
  auto gradOfLossWrtWeights =
    poplin::matMul(graph, inputsTransposed, gradients[0],
    prog, debug_prefix + "/dLdW");
  auto gradOfLossWrtInput =
    popops::mul(graph,
              gradients[0].broadcast(fwd_inputs[1].dim(0), 1),
              weightsTransposed.broadcast(gradients[0].dim(0), 0),
              prog,
              debug_prefix + "/dLdX");

  outputs.push_back(gradOfLossWrtInput);
  outputs.push_back(gradOfLossWrtWeights);

  return prog;
}
