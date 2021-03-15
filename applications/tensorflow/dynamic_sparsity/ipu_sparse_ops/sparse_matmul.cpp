// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <popsparse/FullyConnected.hpp>
#include <poputil/exceptions.hpp>
#include <popsparse/codelets.hpp>
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
  // lhs, metainfo, and nzvalues need special allocators but the current
  // API doesn't allow extra compile time arguments to be passed so we can't
  // use them. The workaround is a separate allocator op.
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
 * Build the fwd pass of sparse matmul:
 *   dense-activation = matrix_multiply(sparse_weights, dense_input)
 *
 * The fwd function requires 5 inputs. Input at position [0] is the
 * dense input, positions [1]/[2] are two sparse weight representation
 * tensors: where [1] is the metainfo (encodes the positions
 * of the non-zeros) and [2] are the non-zero weight values themselves.
 * Note: as tensors in their own right these two inputs are dense
 * (i.e. all entries are used to execute the describe the sparse op).
 *
 * Input positions [3] and [4] are placeholders for use in
 * the backwards pass so are not used in this function.
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
  poplar::OptionFlags options = getSparseMulDefaultOptions(json_args.matmul_options);

  using namespace popsparse::dynamic;

  if (inputs.size() != 5) {
    throw poputil::poplibs_error("Sparse matmul requires 5 inputs.");
  }

  auto metaInfoType = poplar::UNSIGNED_SHORT;

  auto lhs = inputs[0];
  auto metainfo = inputs[1].reinterpret(metaInfoType);
  auto nzvalues = inputs[2];

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

  poplar::program::Sequence prog;

  const auto op_name = debug_prefix + "/sparse_matmul_fwd";

  auto weights = SparseTensor(metainfo, nzvalues);
  auto denseActivations = fullyConnectedFwd(
      graph, weights, lhs, params, prog, op_name, options, getSparseCache());
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
    // Allocate the dense layer input:
    auto op_name = debug_prefix + "/dense_lhs";
    return createFullyConnectedInput(
      graph, dataType, params, op_name, options, getSparseCache());
  }

  if (operand == 1) {
    // Allocate the sparse metainfo for the weights:
    const SparseTensor weights = createFullyConnectedWeights(
	        graph, dataType, params, "sparse_fc_weights", options, getSparseCache());
    return weights.getMetaInfoTensor().reinterpret(poplar::HALF);
  }

  if (operand == 2) {
    // Allocate the sparse non-zero values for the weights:
    const SparseTensor weights = createFullyConnectedWeights(
	        graph, dataType, params, "sparse_fc_metainfo", options, getSparseCache());
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

  auto json_args = read_json_args(attributes);
  poplar::OptionFlags sparse_options = getSparseMulDefaultOptions(json_args.matmul_options);
  poplar::OptionFlags dense_options = getDenseGradMulDefaultOptions(json_args.dense_grad_matmul_options);

  auto metaInfoType = poplar::UNSIGNED_SHORT;
  auto lhs = fwd_inputs[0];
  auto metainfo = fwd_inputs[1].reinterpret(metaInfoType);
  auto nzvalues = fwd_inputs[2];
  auto doDenseGradComputation = fwd_inputs[3];

  auto blockSize = json_args.block_size;
  auto batchSize = json_args.batch_size;
  auto inputSize = json_args.input_size;
  auto outputSize = json_args.output_size;
  auto sparsityFactor = computeDensity(json_args);
  auto numGroups = json_args.group_size;
  auto poolingType = json_args.pooling_type;

  SparsityParams sparsityParams(
    SparsityType::Element, SparsityStructure::Unstructured, {blockSize, blockSize});
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);

  auto inflowingGrad = gradients[0];
  auto weights = SparseTensor(metainfo, nzvalues);

  poplar::program::Sequence prog;

  auto gradOfLossWrtInput = fullyConnectedGradA(
    graph, weights, inflowingGrad, params, prog,
    debug_prefix + "/sparse_matmul_gradA", sparse_options, getSparseCache());

  auto gradOfLossWrtNzValues = fullyConnectedSparseGradW(
    graph, weights.getMetaInfoTensor(), inflowingGrad, lhs, params, prog,
    debug_prefix + "/sparse_matmul_gradW", sparse_options, getSparseCache());

  outputs.push_back(gradOfLossWrtInput);
  outputs.push_back(gradOfLossWrtNzValues);

  // As an additional step we conditionally return the full weight grad.
  // Eventually this will be seruaklused in a memory efficient way (and)
  // perhaps top-k can be taken on the IPU also:
  poplar::program::Sequence conditionalProg;
  auto inputsTransposed = lhs.dimShuffle({1, 0});

  // Retrieve recommended split number for dense grad matmul
  auto [ tmp, inSplits, outSplits ] = fullyConnectedDenseGradWSerialSplits(
    graph, inputsTransposed.elementType(), params, sparse_options, getSparseCache());

  // Compute the dense grad matmul, in a serialized fashion if needed
  auto gradOfLossWrtWeights = serializedMatmul(graph, conditionalProg,
                                               inputsTransposed, inflowingGrad,
                                               inSplits, outSplits,
                                               debug_prefix + "/dense_matmul_gradW",
                                               false, dense_options, getDenseCache());

  auto pooledGradsWrtWeights =
    pool(graph, poolingType, blockSize, gradOfLossWrtWeights,
         conditionalProg, debug_prefix + "/dense_matmul_gradW_pooling");

  // Make sure the dense grad doesn't remain live when not needed
  auto killDenseGrad = poplar::program::Sequence();
  killDenseGrad.add(poplar::program::WriteUndef(pooledGradsWrtWeights));

  if (json_args.debug_printing) {
    conditionalProg.add(poplar::program::PrintTensor("Computed dense grad for '" + debug_prefix + "'", doDenseGradComputation));
  }

  auto ifProg = poplar::program::If(doDenseGradComputation,
    conditionalProg, killDenseGrad);
  prog.add(ifProg);

  // The standard weight grad seen by TF is the sparse one so the dense
  // one has to be requested separately by using tf.gradients() with a
  // dummy var:
  outputs.push_back(pooledGradsWrtWeights);

  return prog;
}

} // end extern "C"
