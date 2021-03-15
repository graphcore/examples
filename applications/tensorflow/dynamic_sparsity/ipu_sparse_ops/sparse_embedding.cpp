// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <popsparse/Embedding.hpp>
#include <popsparse/FullyConnected.hpp>
#include <popops/Fill.hpp>
#include <poputil/exceptions.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include "utils.hpp"

extern "C" {
/// We are using a stateless op which requires
/// API level 1 or higher.
int32_t custom_op_api_level = 4;

/// Meta data function sets properties of the forward op.
void Build_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
                    bool& is_elementwise,
                    bool& is_stateless,
                    bool& is_hashable,
                    std::uint32_t num_inputs) {
  allocating_indices = {0, 1, 2};
  is_elementwise = false;
  is_stateless = true;
}

// Return a planning cache that is global in this shared
// object (but have one per thread):
popsparse::dynamic::PlanningCache* getSparseCache() {
  thread_local popsparse::dynamic::PlanningCache global_sparse_cache;
  return &global_sparse_cache;
}

poplin::matmul::PlanningCache* getDenseCache() {
  thread_local poplin::matmul::PlanningCache global_dense_cache;
  return &global_dense_cache;
}

/**
 * Build the fwd function for a sparse embedding that can be used with a
 * tied projection.
 * 
 * Requires 3 inputs. Input at position [0] is the token ids to lookup in the
 * embedding matrix. The inputs at positions [1]/[2] are two sparse weight
 * representation tensors: where [1] is the metainfo (encodes the positions
 * of the non-zeros) and [2] are the non-zero embedding weights themselves.
 * Note: as tensors in their own right these two inputs are dense
 * (i.e. all entries are used to execute the describe the sparse op).
 *
 * Extra information describing the sparse op is carried in the attributes
 * parameter string encoded as JSON.
 **/
poplar::program::Program Build(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix) {

  auto json_args = read_json_args(attributes);
  poplar::OptionFlags matmulOptions = getSparseMulDefaultOptions(json_args.matmul_options);

  using namespace popsparse::dynamic;

  if (inputs.size() != 3) {
    throw poputil::poplibs_error("Sparse embedding requires 3 inputs.");
  }

  auto indicesType = poplar::UNSIGNED_INT;
  auto metaInfoType = poplar::UNSIGNED_SHORT;
  auto blockSize = json_args.block_size;
  auto batchSize = json_args.batch_size;
  auto inputSize = json_args.input_size;
  auto outputSize = json_args.output_size;
  auto sparsityFactor = computeDensity(json_args);
  auto numGroups = json_args.group_size;
  SparsityParams sparsityParams(
    SparsityType::Element, SparsityStructure::Unstructured, {blockSize, blockSize});
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);

  // Build the sparse weights object from the inputs:
  auto indices = inputs[0].reinterpret(indicesType);
  auto metainfo = inputs[1].reinterpret(metaInfoType);
  auto nzvalues = inputs[2];
  const auto embeddingMatrix = SparseTensor(metainfo, nzvalues);

  // Construct the embedding lookup program:
  poplar::program::Sequence prog;
  const auto op_name = debug_prefix + "/sparse_embedding_fwd";
  auto denseActivations = embeddingSlice(graph, embeddingMatrix, indices, prog, params,
                                         op_name, matmulOptions, getSparseCache());
  outputs.push_back(denseActivations);
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
  auto blockSize = json_args.block_size;
  auto batchSize = json_args.batch_size;
  auto inputSize = json_args.input_size;
  auto outputSize = json_args.output_size;
  auto sparsityFactor = computeDensity(json_args);
  auto numGroups = json_args.group_size;

  using namespace popsparse::dynamic;

  SparsityParams sparsityParams(
    SparsityType::Element, SparsityStructure::Unstructured, {blockSize, blockSize});
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);

  if (operand == 0) {
    auto op_name = debug_prefix + "/token_indices";
    auto indices =
      createIndicesTensor(graph, params, batchSize, options, op_name);
    return indices.reinterpret(poplar::FLOAT);
  }

  if (operand == 1) {
    // Allocate the sparse metainfo for the weights:
    const SparseTensor weights = createFullyConnectedWeights(
	        graph, dataType, params, "sparse_embedding_weights", options, getSparseCache());
    return weights.getMetaInfoTensor().reinterpret(poplar::HALF);
  }

  if (operand == 2) {
    // Allocate the sparse non-zero values for the weights:
    const SparseTensor weights = createFullyConnectedWeights(
	        graph, dataType, params, "sparse_embedding_metainfo", options, getSparseCache());
    return weights.getNzValuesTensor();
  }

  if (operand > 2) {
    throw std::logic_error("Unexpected operand index in sparse_matmul allocator: " + operand);
  }

  // Unreachable
  return poplar::Tensor();
}

/// Meta data function sets properties of the gradient op.
void Build_grad_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
                    bool& is_elementwise,
                    bool& is_stateless,
                    bool& is_hashable,
                    std::uint32_t num_inputs) {
  allocating_indices.clear();
  is_elementwise = false;
  is_stateless = true;
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

  if (input_grad_index != 2) {
    throw std::logic_error("Grad indices other than 2 are not supported. "
                           "Note: You must pass 'separate_gradients=True' to precompiled_user_op() "
                           "and can only use the sparse embedding at the first input to a model.");
  }

  auto json_args = read_json_args(attributes);
  poplar::OptionFlags options = getSparseMulDefaultOptions(json_args.matmul_options);
  auto blockSize = json_args.block_size;
  auto batchSize = json_args.batch_size;
  auto inputSize = json_args.input_size;
  auto outputSize = json_args.output_size;
  auto sparsityFactor = computeDensity(json_args);
  auto numGroups = json_args.group_size;
  auto poolingType = json_args.pooling_type;
  auto updateScale = json_args.embedding_grad_scale;
  auto indicesType = poplar::UNSIGNED_INT;
  auto metaInfoType = poplar::UNSIGNED_SHORT;

  SparsityParams sparsityParams(
    SparsityType::Element, SparsityStructure::Unstructured, {blockSize, blockSize});
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);

  auto indices = fwd_inputs[0].reinterpret(indicesType);
  auto metainfo = fwd_inputs[1].reinterpret(metaInfoType);
  auto nzvalues = fwd_inputs[2];

  auto weightsType = nzvalues.elementType();
  auto gradValues = graph.clone(nzvalues, debug_prefix + "/sparse_grad");
  const auto sparseGrad = SparseTensor(metainfo, gradValues);
  auto scale = graph.addConstant(poplar::FLOAT, {}, updateScale);
  graph.setTileMapping(scale, 0);

  poplar::program::Sequence prog;
  popops::fill(graph, gradValues, prog, 0.f, debug_prefix + "/init_grad_to_zero");
  embeddingUpdateAdd(graph, sparseGrad, gradients[0], indices, scale,
                     prog, params, debug_prefix + "/update_embedding",
                     options, getSparseCache());

  outputs.push_back(sparseGrad.getNzValuesTensor());
  return prog;
}

} // end extern "C"
