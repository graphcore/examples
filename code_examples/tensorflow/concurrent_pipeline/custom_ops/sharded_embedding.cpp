// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popops/Fill.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Encoding.hpp>
#include <poputil/TileMapping.hpp>
#include <sstream>

#include "utils.hpp"
#include "sharded_utils.hpp"
#include "common.hpp"

extern "C" {

void debug_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {
  is_stateless = false;
  input_to_output_tensor_aliasing = {{0, 0}};
}

poplar::program::Program debug(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix)
{
  poplar::program::Sequence prog;
  std::cerr << "FWD Op: " << debug_prefix << "\n  Tensor '" << inputs[0].getDebugStr() << "' shape: " << inputs[0].shape() << " sharding: " << getIPUMapping(graph, inputs[0]) << "\n";
  prog.add(poplar::program::PrintTensor(
    debug_prefix + ": " + inputs[0].getDebugStr(), inputs[0]));
  outputs.push_back(inputs[0]);
  return prog;
}

void debug_grad_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs)
{
  is_stateless = false;
  input_to_output_tensor_aliasing = {{0, 0}};
}

poplar::program::Program debug_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix) {

  std::cerr << "BWD Op: " << debug_prefix << "\n  Tensor '" << gradients[0].getDebugStr() << "' shape: " << gradients[0].shape() << " sharding: " << getIPUMapping(graph, gradients[0]) << "\n";

  poplar::program::Sequence prog;
  prog.add(poplar::program::PrintTensor(
    debug_prefix + ": " + gradients[0].getDebugStr(), gradients[0]));
  outputs.push_back(gradients[0]);
  return prog;
}

void embedding_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs)
{
  allocating_indices = {0, 1};
  is_stateless = true;
}

poplar::program::Program embedding(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix)
{
  if (inputs.size() != 2) {
    throw poputil::poplibs_error("Embedding requires 2 inputs (features, indices).");
  }

  std::cerr << "Building custom gather op with attributes: " + attributes + "\n";
  auto attr = readJsonGatherAttributes(attributes);
  const auto& features = inputs[0];
  const auto& intIndices = inputs[1][0];
  checkGatherTensorDimensions(attr, features, intIndices);

  poplar::program::Sequence program;

  auto shards = createIpuShards(graph);
  auto featuresSharded = getShardedTensorChunks(shards, 0, features);
  auto partitionSizes = getPartitionSizes(featuresSharded, 0);

  poplar::Tensor adjustedIndices;
  std::vector<poplar::Tensor> masks;
  std::tie(adjustedIndices, masks) = adjustIndicesAndCreateMasks(
    graph, partitionSizes, inputs[1], program, debug_prefix + "/adjust_indices");

  auto opts = getSliceOptions(attr.slice_options);
  std::vector<poplar::Tensor> results;
  for (auto s = 0u; s < shards.size(); ++s) {
    auto shardSuffixStr = std::to_string(s);
    auto shardFeatures = featuresSharded[s];
    auto shardIndices = adjustedIndices[s];
    popops::SlicePlan slicePlan = popops::embedding::plan(shards[s], features.elementType(),
                                            shardFeatures.dim(0), shardFeatures.dim(1),
                                            {shardIndices.dim(1), 1}, opts);
    auto shardSlices = popops::multiSlice(shards[s], shardFeatures, shardIndices, {0}, {1}, program, slicePlan, {});

    // Masks are vectors so we need to broadcast the mask to match the feature dimension:
    auto mask = masks[s].broadcast(shardFeatures.dim(1), 1).expand({1});
    auto zeros = zerosLike(shards[s], shardSlices, program);
    popops::selectInPlace(shards[s], shardSlices, zeros, mask, program,
                                 debug_prefix + "/apply_shard_mask_" + shardSuffixStr);
    results.push_back(shardSlices.expand({0}));
  }

  // Reduce the partial masked gathered elements across shards
  // to get the final result onto a single IPU:
  std::vector<poplar::Tensor> resultsNotOnShard0(results.begin()+1, results.end());
  auto partials = poplar::concat(resultsNotOnShard0, 0);
  poplar::program::Sequence prog;
  popops::reduceWithOutput(graph, partials, results[0], {0},
                           popops::ReduceParams(popops::Operation::ADD, true), program,
                           debug_prefix + "/reduce_partial_slices");
  // Reduce with output produces an unnecessary extra dimension compared to regular reduce
  // and we also want to remove the extra dimension produced by slicing (NOTE: in the backwards
  // we need to undo the squeeze of the dimension introduced by slice but not the one due to reduce):
  outputs.push_back(results[0].squeeze({0, 2}));
  return program;
}

poplar::Tensor embedding_allocator(
  poplar::Graph& graph,
  std::uint32_t operand,
  const std::vector<size_t>& shape,
  poplar::Type type,
  const std::string& attributes,
  const std::string& prefix)
{
  auto attr = readJsonGatherAttributes(attributes);
  auto shards = createIpuShards(graph);

  if (attr.feature_count % shards.size()) {
    std::stringstream ss;
    ss << prefix << ": Features/vocab size does not divide equally into the number of shards." << "\n";
    throw std::logic_error(ss.str());
  }
  auto rowsPerShard = attr.feature_count / shards.size();

  auto opts = getSliceOptions(attr.slice_options);

  if (operand == 0) {
    if (shape[0] != attr.feature_count) {
      throw std::logic_error("Shape mismatch in allocator: " + prefix);
    }

    std::cerr << "Allocating sliceable weight tensor for operand " << std::to_string(operand) << " across " << shards.size() << " shards with " << rowsPerShard << " rows per shard\n";

    std::vector<poplar::Tensor> shardedWeights;

    for (auto s = 0u; s < shards.size(); ++s) {
      popops::SlicePlan slicePlan = popops::embedding::plan(shards[s], type,
                                              rowsPerShard, attr.feature_dim,
                                              {attr.output_count, 1}, opts);
      auto w = popops::createSliceableTensor(shards[s], type, {rowsPerShard, attr.feature_dim}, {0}, {1}, slicePlan, {});
      shardedWeights.push_back(w);
    }

    auto result = poplar::concat(shardedWeights, 0);
    return result;
  }

  if (operand == 1) {
    // Allocator for the indices.
    std::cerr << "Allocating index tensor for operand " << std::to_string(operand) << "\n";

    // We allocate indices entirely on shard 0: it is up to user code to broadcast it to other
    // shards.
    popops::SlicePlan slicePlan = popops::embedding::plan(shards[0], type,
                                        rowsPerShard, attr.feature_dim,
                                        {attr.output_count, 1}, opts);

    if (type != poplar::INT) {
      throw std::logic_error("Expected INT type for operand: " + std::to_string(operand));
    }

    auto indices = popops::createIndicesTensor(graph, {rowsPerShard}, attr.output_count, slicePlan, {});
    // We need to reinterpret the type as IPU XLA does not support unsigned ints:
    return indices.reinterpret(poplar::INT);
  }

  throw std::logic_error("Allocator called for invalid operand: " + std::to_string(operand));

  // Unreachable
  return poplar::Tensor();
}

/// Set various properties of the forward op:
void embedding_grad_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs)
{
  allocating_indices = {0, 1};
  is_stateless = true;
}

poplar::program::Program embedding_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix)
{
  std::cerr << "Building custom gather gradient op index " + std::to_string(input_grad_index) +
               " with attributes: " + attributes + "\n";

  if (input_grad_index != 0) {
    throw std::logic_error("Only the features have a gradient. "
                           "Note: You must pass 'separate_gradients=True' to precompiled_user_op() ");
  }

  auto attr = readJsonGatherAttributes(attributes);
  poplar::program::Sequence program;

  auto shards = createIpuShards(graph);
  auto featuresSharded = getShardedTensorChunks(shards, 0, fwd_inputs[0]);
  auto partitionSizes = getPartitionSizes(featuresSharded, 0);

  poplar::Tensor adjustedIndices;
  std::vector<poplar::Tensor> masks;
  std::tie(adjustedIndices, masks) = adjustIndicesAndCreateMasks(
    graph, partitionSizes, fwd_inputs[1], program, debug_prefix + "/adjust_indices");

  auto indices = adjustedIndices[0];
  auto features = fwd_inputs[0];
  //program.add(poplar::program::PrintTensor("Incoming grad", gradients[0]));

  auto grad = gradients[0].expand({1}); // Undo the squeeze we did in fwd pass
  poplar::Tensor gradBroadcast;
  program.add(copyToAll(graph, grad, gradBroadcast, debug_prefix + "/broadcast_grad_to_all_shards"));

  checkGatherTensorDimensions(attr, features, indices);

  auto opts = getSliceOptions(attr.slice_options);
  auto featuresType = features.elementType();
  auto gatherGrad = zerosLike(graph, features, program);
  auto gatherGradSharded = getShardedTensorChunks(shards, 0, gatherGrad);

  for (auto s = 0u; s < shards.size(); ++s) {
    auto shardSuffixStr = std::to_string(s);
    auto shardGatherGrad = gatherGradSharded[s];
    auto shardIndices = adjustedIndices[s];
    //program.add(poplar::program::PrintTensor("Adjusted Indices shard" + shardSuffixStr, adjustedIndices[s]));
    popops::SlicePlan slicePlan = popops::embedding::plan(shards[s], features.elementType(),
                                            shardGatherGrad.dim(0), shardGatherGrad.dim(1),
                                            {shardIndices.dim(1), 1}, opts);
    auto scale = shards[s].addConstant(shardGatherGrad.elementType(), {}, attr.gradient_scale);
    shards[s].setTileMapping(scale, getFirstTile(shards[s], shardGatherGrad));
    popops::multiUpdateAdd(shards[s], shardGatherGrad, gradBroadcast[s], shardIndices,
                           scale, {0}, {1},
                           program, slicePlan, {}, debug_prefix + "/gather_grad_update_add");
  }

  //program.add(poplar::program::PrintTensor("Outgoing grad", gatherGrad));
  outputs.push_back(gatherGrad);
  return program;
}

void allocate_tied_embedding_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {
  allocating_indices = {0, 1};
  input_to_output_tensor_aliasing = {{0, 0}, {1, 1}};
}

poplar::Tensor allocate_tied_embedding_allocator(
  poplar::Graph& graph,
  std::uint32_t operand,
  const std::vector<size_t>& shape,
  poplar::Type type,
  const std::string& attributes,
  const std::string& prefix)
{
  // Call straight through to the embedding allocator
  // to avoid duplicating code:
  return embedding_allocator(graph, operand, shape, type, attributes, prefix);
}

poplar::program::Program allocate_tied_embedding(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix)
{
  for (auto t : inputs) { outputs.push_back(t); }
  return poplar::program::Sequence();
}

void allocate_tied_embedding_grad_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs)
{
  input_to_output_tensor_aliasing = {{0, 0}, {1, 1}};
}

poplar::program::Program allocate_tied_embedding_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix) {
  for (auto t : gradients) { outputs.push_back(t); }
  return poplar::program::Sequence();
}

} // end extern "C"
