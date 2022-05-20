// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popops/Fill.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>

#include "utils.hpp"
#include "common.hpp"

extern "C" {

/// Set various properties of the forward op:
void sharded_matmul_metadata(
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

/// perform a matmul where the input has already been broadcast across all IPUs
/// and the weights are sharded across IPUs by columns.
poplar::program::Program sharded_matmul(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix)
{
  if (inputs.size() != 2) {
    throw poputil::poplibs_error("Matrix product requires 2 inputs.");
  }

  auto shard = createIpuShards(graph);
  auto input = inputs[0];
  auto weights = inputs[1];

  // Check all dimensions:
  if (input.dim(0) != shard.size() || input.rank() != 3 || weights.rank() != 2) {
    throw std::logic_error("LHS input must be a matrix broadcast across IPUs (rank 3) and RHS must be a matrix (rank 2).");
  }

  if (input[0].dim(1) != weights.dim(0)) {
    throw std::logic_error("Matrix product shapes incompatible.");
  }

  // Check input is on all shards:
  checkExpectedSharding(graph, input, getIpuSet(shard.size()));

  // Check the weights can be sharded:
  auto attr = readJsonAttributes(attributes);
  auto matmul_opts = getMatmulOptions(attr.matmul_options);
  checkOkForSharding(shard.size(), attr);

  // Get list of weights per IPU and check they are sharded correctly:
  auto shardedWeights = getShardedTensorChunks(shard, 1, weights);
  checkShardedTensorMapping(graph, shard.size(), shardedWeights);

  poplar::program::Sequence prog;

  // Compute partial matmul results on each shard:
  std::vector<poplar::Tensor> shardedResult;
  for (auto s = 0u; s < shard.size(); ++s) {
    auto result = poplin::matMul(shard[s], input[s], shardedWeights[s], prog,
                                 debug_prefix + "/fwd_matmul_shard" + std::to_string(s), matmul_opts, getPlanningCache());
    shardedResult.push_back(result);
  }

  // Result tensor is sharded (by columns) across all the IPUs:
  auto result = poplar::concat(shardedResult, 1);
  outputs.push_back(result);
  return prog;
}

poplar::Tensor sharded_matmul_allocator(
  poplar::Graph& graph, std::uint32_t operand,
  const std::vector<size_t>& shape,
  poplar::Type type,
  const std::string& attributes,
  const std::string& prefix) {

  auto shards = createIpuShards(graph);
  auto attr = readJsonAttributes(attributes);
  auto matmul_opts = getMatmulOptions(attr.matmul_options);
  checkOkForSharding(shards.size(), attr);

  if (operand == 0) {
    // Allocate the LHS (input): the input needs to be broadcast to
    // all shards but we can just let Poplar handle this.
    std::cerr << prefix << ": allocating input on shard 0\n";
    const auto weightColumnsPerShard = attr.cols_rhs / shards.size();
    return poplin::createMatMulInputLHS(shards[0], type,
                                        {attr.rows_lhs, attr.cols_lhs},
                                        {attr.cols_lhs, weightColumnsPerShard},
                                        prefix + "/lhs", matmul_opts, getPlanningCache());
  }

  if (operand == 1) {
    auto weights = allocateShardedWeightMatrix(shards, attr, type, prefix);
    std::cerr << prefix << ": allocating sharded weights with shape: " << weights.shape()
              << " and sharding: " << getIPUMapping(graph, weights) << "\n";
    return weights;
  }

  throw std::logic_error("Tensor allocation requested for unexpected operand.");

  // Unreachable
  return poplar::Tensor();
}

void sharded_matmul_grad_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::vector<std::int64_t>& replica_identical_output_indices,
                    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
                    bool& is_elementwise,
                    bool& is_stateless,
                    bool& is_hashable,
                    std::uint32_t num_inputs)
{
  allocating_indices.clear();
  is_elementwise = false;
  is_stateless = true;
}

poplar::program::Program sharded_matmul_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix)
{
  auto shard = createIpuShards(graph);
  auto attr = readJsonAttributes(attributes);
  auto matmul_opts = getMatmulOptions(attr.matmul_options);
  checkOkForSharding(shard.size(), attr);

  poplar::program::Sequence prog;
  auto input = fwd_inputs[0];
  auto weights = fwd_inputs[1];
  auto inputTransposed = input.dimShuffle({0, 2, 1});
  auto weightColsPerShard = weights.dim(1) / 2;

  // Check that the incoming gradient is sharded across all IPUs:
  auto gradient = gradients[0];
  checkExpectedSharding(graph, gradient, getIpuSet(shard.size()));
  // Check it is sharded by columns:
  auto shardedGradOfOutput = getShardedTensorChunks(shard, 1, gradient);
  checkShardedTensorMapping(graph, shard.size(), shardedGradOfOutput);

  // We checked the input is copied to every shard above and the incoming gradient
  // is sharded across IPUs by columns so all computation is local to each shard:
  std::vector<poplar::Tensor> sharded_dLdW;
  for (auto s = 0u; s < shard.size(); ++s) {
    auto chunkOfdLdW =
      poplin::matMul(shard[s], inputTransposed[s], shardedGradOfOutput[s], prog,
      debug_prefix + "/dLdW_shard_"  + std::to_string(s),
      matmul_opts, getPlanningCache());
      sharded_dLdW.push_back(chunkOfdLdW);
  }
  auto gradOfLossWrtWeights = poplar::concat(sharded_dLdW, 1);

  // Weights are sharded (by columns) so we need to
  // compute partial results of dL/dInput on each IPU:
  auto shardedWeights = getShardedTensorChunks(shard, 1, weights);

  // Compute the partial results per shard:
  std::vector<poplar::Tensor> partialResults;
  for (auto s = 0u; s < shard.size(); ++s) {
    auto result = poplin::matMul(shard[s], shardedGradOfOutput[s], shardedWeights[s].transpose(), prog,
                                 debug_prefix + "/dLdI_shard" + std::to_string(s),
                                 matmul_opts, getPlanningCache());
    partialResults.push_back(result.expand({0})); // expand tensors so they can be concatted and reduced
  }
  auto partialGradOfLossWrtInput = poplar::concat(partialResults, 0);

  // Partial input grads and weight grad should now be sharded across all IPUs:
  checkExpectedSharding(graph, partialGradOfLossWrtInput, getIpuSet(shard.size()));
  checkExpectedSharding(graph, gradOfLossWrtWeights, getIpuSet(shard.size()));

  // We return the partial input grads directly (we are assuming the
  // preceding op was sharded.copy_to_all or similar that will reduce
  // the grads itself):
  outputs.push_back(partialGradOfLossWrtInput);
  outputs.push_back(gradOfLossWrtWeights);
  return prog;
}

} // end extern "C"
