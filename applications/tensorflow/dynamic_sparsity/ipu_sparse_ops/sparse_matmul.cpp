// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <popsparse/FullyConnected.hpp>
#include <poputil/exceptions.hpp>
#include <popsparse/codelets.hpp>
#include <popnn/Loss.hpp>
#include <popops/ElementWise.hpp>
#include "utils.hpp"

extern "C" {

/// We are using a stateless op which requires
/// API level 1 or higher.
int32_t custom_op_api_level = 2;

/// Meta data function sets properties of the forward op.
void Build_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::uint32_t& num_inplace,
                    bool& is_elementwise,
                    bool& is_stateless,
                    std::uint32_t num_inputs) {
  // lhs, metainfo, and nzvalues need special allocators but the current
  // API doesn't allow extra compile time arguments to be passed so we can't
  // use them. The workaround is a separate allocator op.
  allocating_indices = {0, 1, 2};
  num_inplace = 0;
  is_elementwise = false;
  is_stateless = true;
}

// Return a planning cache that is global in this shared
// object (but have one per thread):
popsparse::dynamic::PlanningCache* getCache() {
  thread_local popsparse::dynamic::PlanningCache global_cache;
  return &global_cache;
}

poplar::program::Program Build(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix) {

  auto json_args = read_json_args(attributes);
  poplar::OptionFlags options = getSparseMulDefaultOptions(json_args.matmul_options);

  popsparse::addCodelets(graph);
  using namespace popsparse::dynamic;

  if (inputs.size() != 5) {
    throw poputil::poplibs_error("Sparse matmul requires 5 inputs.");
  }

  auto metaInfoType = poplar::UNSIGNED_SHORT;

  auto lhs = inputs[0];
  auto metainfo = inputs[1].reinterpret(metaInfoType);
  auto nzvalues = inputs[2];

  auto batchSize = json_args.batch_size;
  auto inputSize = json_args.input_size;
  auto outputSize = json_args.output_size;
  auto sparsityFactor = json_args.max_non_zeros / float(inputSize * outputSize);
  auto numGroups = json_args.group_size;

  SparsityParams sparsityParams(SparsityType::Element, SparsityStructure::Unstructured);
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);

  poplar::program::Sequence prog;

  const auto op_name = debug_prefix + "/sparse_matmul_fwd";

  auto weights = SparseTensor(metainfo, nzvalues);
  auto sparseResult = fullyConnectedFwd(
      graph, weights, lhs, params, prog, op_name, options, getCache());
  outputs.push_back(sparseResult);
  return prog;
}

poplar::Tensor Build_allocator(
  poplar::Graph& graph, std::uint32_t operand,
  const std::vector<size_t>& shape, poplar::Type type,
  const std::string& attributes,
  const std::string& debug_prefix) {

  auto json_args = read_json_args(attributes);
  poplar::OptionFlags options = getSparseMulDefaultOptions(json_args.matmul_options);
  auto dataType = json_args.data_type;
  auto batchSize = json_args.batch_size;
  auto inputSize = json_args.input_size;
  auto outputSize = json_args.output_size;
  auto sparsityFactor = json_args.max_non_zeros / float(inputSize * outputSize);
  auto numGroups = json_args.group_size;

  using namespace popsparse::dynamic;

  SparsityParams sparsityParams(SparsityType::Element, SparsityStructure::Unstructured);
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);

  if (operand == 0) {
    // Allocate the dense layer input:
    return createFullyConnectedInput(
      graph, dataType, params, "dense_lhs", options, getCache());
  }

  if (operand == 1) {
    // Allocate the sparse metainfo for the weights:
    const SparseTensor weights = createFullyConnectedWeights(
	        graph, dataType, params, "sparse_fc_weights", options, getCache());
    return weights.getMetaInfoTensor().reinterpret(poplar::HALF);
  }

  if (operand == 2) {
    // Allocate the sparse non-zero values for the weights:
    const SparseTensor weights = createFullyConnectedWeights(
	        graph, dataType, params, "sparse_fc_weights", options, getCache());
    return weights.getNzValuesTensor();
  }

  if (operand > 2) {
    throw std::logic_error("Unexpected operand index in sparse_matmul allocator: " + operand);
  }
}

/// Meta data function sets properties of the gradient op.
void Build_grad_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::uint32_t& num_inplace,
                    bool& is_elementwise,
                    bool& is_stateless,
                    std::uint32_t num_inputs) {
  allocating_indices.clear();
  num_inplace = 0;
  is_elementwise = false;
  is_stateless = true;
}

poplar::Tensor topKIndices(
    poplar::Graph& graph,
    unsigned k,
    const poplar::Tensor gradient,
    poplar::program::Sequence &prog,
    const std::string& debug_prefix) {
  // top-k only works on flat indices:
  auto absGradsFlat = popops::abs(
    graph, gradient.reshape({1, gradient.numElements()}), prog);
  auto indices = poplar::Tensor();
  popnn::topK(graph, absGradsFlat, indices, k, true, prog, debug_prefix + "/indices");
  return indices;
}

poplar::program::Program Build_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix) {

  using namespace popsparse::dynamic;

  auto json_args = read_json_args(attributes);
  poplar::OptionFlags options = getSparseMulDefaultOptions(json_args.matmul_options);

  auto metaInfoType = poplar::UNSIGNED_SHORT;
  auto lhs = fwd_inputs[0];
  auto metainfo = fwd_inputs[1].reinterpret(metaInfoType);
  auto nzvalues = fwd_inputs[2];
  auto doDenseGradComputation = fwd_inputs[3];

  auto batchSize = json_args.batch_size;
  auto inputSize = json_args.input_size;
  auto outputSize = json_args.output_size;
  auto sparsityFactor = json_args.max_non_zeros / float(inputSize * outputSize);
  auto numGroups = json_args.group_size;

  SparsityParams sparsityParams(SparsityType::Element, SparsityStructure::Unstructured);
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);

  auto inflowingGrad = gradients[0];
  auto weights = SparseTensor(metainfo, nzvalues);

  poplar::program::Sequence prog;

  auto lossWrtInput = fullyConnectedGradA(
    graph, weights, inflowingGrad, params, prog,
    debug_prefix + "/sparse_matmul_gradA", options, getCache());

  auto lossWrtNzValues = fullyConnectedSparseGradW(
    graph, weights.getMetaInfoTensor(), inflowingGrad, lhs, params, prog,
    debug_prefix + "/sparse_matmul_gradW", options, getCache());

  outputs.push_back(lossWrtInput);
  outputs.push_back(lossWrtNzValues);

  // As an additional step we conditionally return the full weight grad.
  // Eventually this will be seruaklused in a memory efficient way (and)
  // perhaps top-k can be taken on the IPU also:
  poplar::program::Sequence conditionalProg;
  auto inputsTransposed = lhs.dimShuffle({1, 0});

  auto splits = fullyConnectedDenseGradWSerialSplits(
    graph, inputsTransposed.elementType(), params, options, getCache());
  /// TODO: use the splits to serialize the following matmul:
  auto gradOfLossWrtWeights =
    poplin::matMul(graph, inputsTransposed, inflowingGrad,
                   conditionalProg, debug_prefix + "/dense_matmul_gradW");

  auto ifProg = poplar::program::If(doDenseGradComputation,
    conditionalProg, poplar::program::Sequence());
  prog.add(ifProg);

  // The standard weight grad seen by TF is the sparse one so the dense
  // one has to be requested separately by using tf.gradients() with a
  // dummy var:
  outputs.push_back(gradOfLossWrtWeights);

  return prog;
}

} // end extern "C"
