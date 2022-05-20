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
#include <popops/SelectScalarFromRows.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/TensorCloneMethod.hpp>

#include <gcl/Collectives.hpp>

#include <sstream>

#include "utils.hpp"
#include "common.hpp"
#include "sharded_utils.hpp"

extern "C" {

///
/// sharded_log_softmax
///
/// Used after a sharded_log_softmax layer in order to get the cross-entropy
///

void sharded_log_softmax_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs)
{
  allocating_indices.clear();
  is_stateless = true;
}

poplar::program::Program sharded_log_softmax(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix)
{
  auto input = inputs[0];
  auto prog = poplar::program::Sequence();

  // First compute the maximum logit for each batch item on each shard:
  auto shards = createIpuShards(graph);
  auto shardedLogits = getShardedTensorChunks(shards, 1, input);
  checkShardedTensorMapping(graph, shards.size(), shardedLogits);

  std::vector<poplar::Tensor> maxs;
  for (auto s = 0u; s < shards.size(); ++s) {
    auto logits = shardedLogits[s];
    auto max = popops::reduce(shards[s], logits, {1},
                          popops::ReduceParams(popops::Operation::MAX, false),
                          prog, debug_prefix + "/reduce_partial_max");
    // Expand partial results so we can concat them ready for all reduce:
    maxs.push_back(max.expand({0}));
  }

  // All reduce the partial maximums:
  auto interIpuMaxs = poplar::concat(maxs, 0);
  auto max = gcl::allReduceWithinReplica(graph, interIpuMaxs, gcl::CollectiveOperator::MAX, prog, "all_reduce_max");

  // Now we can compute the softmax numerators and partial denominator on each shard:
  auto maxSharded = getShardedTensorChunks(shards, 0, max);
  std::vector<poplar::Tensor> shardedDenominators;
  std::vector<poplar::Tensor> shardedTranslated;
  for (auto s = 0u; s < shards.size(); ++s) {
    auto logits = shardedLogits[s];
    auto max = maxSharded[s].squeeze({0}).expand({1}).broadcast(logits.dim(1), 1);
    // Lets check each slice is entirely on one shard:
    checkExpectedSharding(graph, max, {s});

    // Sub off the max for stable softmax:
    auto translated = popops::sub(shards[s], logits, max, prog,
                                  debug_prefix + "/partial_sub_max");
    auto numerators = popops::exp(shards[s], translated, prog,
                                 debug_prefix + "partial_exp");
    auto partialDenominator = popops::reduce(shards[s], numerators, numerators.elementType(), {1},
	                             popops::Operation::ADD, prog, debug_prefix + "reduce_partial_denom");
    prog.add(poplar::program::WriteUndef(numerators));
    shardedTranslated.push_back(translated.expand({0})); 
    shardedDenominators.push_back(partialDenominator.expand({0}));
  }

  // All reduce the partial denominators to get global denominator on every IPU:
  auto interIpuDenom = poplar::concat(shardedDenominators, 0);
  auto denom = gcl::allReduceWithinReplica(graph, interIpuDenom,
                                           gcl::CollectiveOperator::ADD,
                                           prog, debug_prefix + "all_reduce_denominator");

  // Final calculation of log softmax on each shard (hopefully Poplar
  // doesn't do anything stupid with sharded elementwise ops):
  auto translatedInput = poplar::concat(shardedTranslated, 0);
  auto logsum = popops::map(graph,
                            popops::expr::Cast(popops::expr::Log(popops::expr::_1), input.elementType()),
                            {denom}, prog, "log_denominator").expand({2}).broadcast(translatedInput.dim(2), 2);
  popops::subInPlace(graph, translatedInput, logsum, prog, debug_prefix + "subtract_logsum");
  auto result = translatedInput.dimShuffle({1, 0, 2}).reshape(input.shape());

  outputs.push_back(result);
  return prog;
}

void sharded_log_softmax_grad_metadata(std::vector<std::int64_t>& allocating_indices,
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
  input_to_output_tensor_aliasing.insert(
    std::make_pair(0, 0)
  );
}

poplar::program::Program sharded_log_softmax_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix)
{
  // For forward pass s = softmax(x), y = log(s) the VJP sequence is:
  //   dL/dx = dL/dy * dy/ds * ds/dx
  //         = dL/dy * (1/s) * (diag(s) - s'*s)
  //         where dL/dy = gradients[0] and (the row vector) s = exp(fwd_outputs[0])
  // We can fuse the division by the activations so that
  // (1/s) * (diag(s) - s'*s) becomes (I - s'*s/s) which destroys
  // the outer-product structure and we can form the log-softmax Jacobian
  // with just a broadcast of the softmax activations (s).
  //
  // TODO: This fusion/optimisation means that the matrix formed by broadcasting is
  // only sharded in the columns of the RHS matrix which is what we want, however
  // dL/dx (gradients[0]) is also sharded by columns and we do not want to exchange
  // dL/dx between IPUs if it is large (e.g. as it will be in embedding projections
  // for large vocab language models). To avoid this we can note that if the operation
  // following log-softmax is cross-entropy then dL/dx has only one non-zero (per grad)
  // so we can ensure that single element is already on the correct IPU using a custom
  // cross-entropy layer.
  poplar::program::Sequence prog;

  // Recover the softmax activations s = softmax(x) (fwd_outputs[0] = y = log(s)):
  auto s = popops::exp(graph, fwd_outputs[0], prog, debug_prefix + "/recover_softmax_acts");
  auto dType = s.elementType();

  // We want to compute dL/dx = dL/dy * (I - S) so first form S which is just a broadcast
  // of the activations:
  auto S = s.expand({1}).broadcast(s.dim(1), 1);
  auto dLdy = gradients[0].expand({1});

  // Instead of adding identity matrix to the diagonal of -S to form (I - S) we can
  // transform the equation to do everything in one accumulating matmul as follows:
  // dL/dx = dL/dy * (I - S) = dL/dy - (dL/dy * S)
  auto dLdx = dLdy; // dL/dx can just be an alias for the incoming gradient.

  // Do the batch of VJPs and squeeze the grad vectors into a matrix to return them:
  auto matmul_opts = poplar::OptionFlags();
  if (!attributes.empty()) {
    matmul_opts = getMatmulOptions(attributes);
  }
  poplin::matMulGroupedAcc(graph, dLdy, -1.f, dLdy, S, prog,
                           debug_prefix + "/batch_vjp", matmul_opts, getPlanningCache());
  outputs.push_back(dLdx.squeeze({1}));

  std::cerr << "BWD debugging op '" << debug_prefix
            << "':\n  Tensor: " << dLdy.getDebugStr()
            << " shape: " << dLdy.shape()
            << " sharding: " << getIPUMapping(graph, dLdy)
            << "':\n  Tensor: " << fwd_outputs[0].getDebugStr()
            << " shape: " << fwd_outputs[0].shape()
            << " sharding: " << getIPUMapping(graph, fwd_outputs[0])
            << "':\n  Tensor: " << fwd_inputs[0].getDebugStr()
            << " shape: " << fwd_inputs[0].shape()
            << " sharding: " << getIPUMapping(graph, fwd_inputs[0])
            << "':\n  Tensor: " << dLdx.getDebugStr()
            << " shape: " << dLdx.shape()
            << " sharding: " << getIPUMapping(graph, dLdx) << "\n";
  return prog;
}

///
/// sharded_take_last
///
/// Used after a sharded_log_softmax layer in order to get the cross-entropy
///

void sharded_take_last_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs)
{
  allocating_indices.clear();
  is_stateless = true;
  input_to_output_tensor_aliasing.insert(
    std::make_pair(0, 0)
  );
}

poplar::program::Program sharded_take_last(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix)
{
  // We expect indices to be pre-sharded across all IPUs:
  if (inputs[1].rank() != 3) {
    std::stringstream ss;
    ss << "Indices in op '" << debug_prefix << "' are expected to be a sharded copy over "
    "all IPUs and therefore should have rank 3. (Try using sharded.copy_to_all())" << "\n";
    throw std::logic_error(ss.str());
  }
  auto shards = createIpuShards(graph);
  checkExpectedSharding(graph, inputs[1], getIpuSet(shards.size()));

  auto prog = poplar::program::Sequence();

  // Input is sharded in the axis that we need to gather from so we must explcitly
  // restrict gathers to shards so that we do not exchange the vocab dimension
  // between IPUs:
  auto inputSharded = getShardedTensorChunks(shards, 1, inputs[0]);
  auto partitionSizes = getPartitionSizes(inputSharded, 1);
  poplar::Tensor indices;
  std::vector<poplar::Tensor> masks;
  std::tie(indices, masks) = adjustIndicesAndCreateMasks(
    graph, partitionSizes, inputs[1], prog, debug_prefix + "/adjust_indices");

  std::vector<poplar::Tensor> results;
  for (auto s = 0u; s < shards.size(); ++s) {
    auto shardSuffixStr = std::to_string(s);
    auto shardInput = inputSharded[s];

    // TODO: We manually serialise the slicing over rows as
    // multiSlice goes horribly OOM otherwise but this is very
    // slow:
    std::vector<poplar::Tensor> scalars;
    for (auto r = 0u; r < shardInput.dim(0); ++r) {
      auto rowSuffix = std::to_string(r);
      auto row = shardInput[r];
      auto rowIndex = indices[s][r].expand({1});

      // Copy the row to a sliceable tensor:
      auto sliceableRow = popops::createSliceableTensor(
        shards[s], row.elementType(), row.shape(), {0}, {1}, {}, {});
      prog.add(poplar::program::Copy(row, sliceableRow, false));

      auto scalar = popops::multiSlice(shards[s], sliceableRow, rowIndex, {0}, {1}, prog,
                                       {}, {},
                                       debug_prefix + "/take_along_row_" + rowSuffix + "_shard_" + shardSuffixStr);
      scalars.push_back(scalar);
    }

    auto shardSlice = poplar::concat(scalars);

    // Need to mask the result at invalid indices:
    auto mask = masks[s];
    auto zeros = zerosLike(shards[s], shardSlice, prog);
    popops::selectInPlace(shards[s], shardSlice, zeros, mask, prog,
                                 debug_prefix + "/apply_shard_mask_" + shardSuffixStr);

    results.push_back(shardSlice.expand({0}));
  }

  // Reduce the partial masked gathered elements across shards to get the final result onto
  // single IPU for final loss reduction stage.
  auto partials = poplar::concat(results, 0);
  auto result = popops::reduce(graph, partials, partials.elementType(), {0},
                               popops::Operation::ADD, prog, debug_prefix + "reduce_partial_denom");
  outputs.push_back(result);

  return prog;
}

void sharded_take_last_grad_metadata(std::vector<std::int64_t>& allocating_indices,
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
  input_to_output_tensor_aliasing.insert(
    std::make_pair(0, 0)
  );
}

poplar::program::Program sharded_take_last_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix)
{
  auto gradient = gradients[0];
  auto indices = fwd_inputs[1][0];
  auto x = fwd_inputs[0];
  auto dLdx = graph.clone(x);

  // multiUpdateAdd is faster than multiUpdate so create a scale factor of 1:
  auto scale = graph.addConstant(dLdx.elementType(), {}, 1.f);
  graph.setTileMapping(scale, 0);

  // TODO: This is very inefficient for large "vocab" dimension as it doesn't respect
  // sharding at all (but if you are concerned with performance you should be using the
  // fused BWD pass that comes from using `sharded_log_softmax_cross_entropy()`).
  auto prog = poplar::program::Sequence();
  auto unsignedIndices = popops::cast(graph, indices, poplar::UNSIGNED_INT, prog, debug_prefix + "/cast_indices");
  popops::fill(graph, dLdx, prog, 0.f, debug_prefix + "/init_zero");
  for (auto i = 0u; i < x.dim(0); ++i) {
    popops::multiUpdateAdd(graph, dLdx[i].expand({1}), gradient[i].expand({0, 0}), unsignedIndices[i].expand({0}),
                           scale, {0}, {1}, prog, {}, {},
                           debug_prefix + "/scatter_gradients_" + std::to_string(i));
  }

  outputs.push_back(dLdx);

  return prog;
}

///
/// sharded_log_softmax_cross_entropy
///

void sharded_log_softmax_cross_entropy_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs)
{
  allocating_indices.clear();
  is_stateless = true;
}

void sharded_log_softmax_cross_entropy_grad_metadata(std::vector<std::int64_t>& allocating_indices,
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

poplar::program::Program sharded_log_softmax_cross_entropy(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix)
{
  std::vector<poplar::Tensor> logSoftmaxOutputs;
  poplar::program::Sequence prog;
  prog.add(sharded_log_softmax(graph, inputs, logSoftmaxOutputs, attributes, debug_prefix + "/log_softmax"));
  popops::negInPlace(graph, logSoftmaxOutputs[0], prog, debug_prefix + "/neg_log_softmax");

  // Pass the label indices forward to take_last:
  logSoftmaxOutputs.push_back(inputs[1]);

  poplar::program::Sequence takeLastProg;
  takeLastProg.add(sharded_take_last(graph, logSoftmaxOutputs, outputs, attributes, debug_prefix + "/take_last"));
  outputs.push_back(logSoftmaxOutputs[0]); // We also stash -ve log-softmax for the backwards pass.
  prog.add(takeLastProg);
  return prog;
}

// Fused backwards pass is much simpler and more efficient:
poplar::program::Program sharded_log_softmax_cross_entropy_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix)
{
  // The fwd_inputs for sharded_take_last were -log(softmax()) forwards outputs which we stashed additionally:
  auto log_softmax_output = fwd_outputs[1];
  auto indices = fwd_inputs[1];
  auto shards = createIpuShards(graph);
  poplar::program::Sequence prog;

  // In the forward op we negated the sharded_log_softmax_grad result
  // but we saved the +ve activations so negate the activations:
  popops::negInPlace(graph, log_softmax_output, prog);
  auto s = popops::exp(graph, log_softmax_output, prog, debug_prefix + "/recover_softmax_acts");

  // TODO: Not sure this happens in parallel? (Check it happens in parallel on each shard):
  for (auto i = 0u; i < log_softmax_output.dim(0); ++i) {
    auto scale = gradients[0][i];
    popops::mulInPlace(graph, s[i], scale, prog, debug_prefix + "/scale_softmax_acts");
  }

  // Finally we need to scatter-subtract the incoming gradient from what we have computed:
  // The incoming gradient is not sharded as it was reduced for the final loss. The stashed activations
  // are sharded across all IPUs in the dimension we need to update (which could be very large). So like
  // the FWD pass we want to split the scatter operation across shards to avoid any communication of the
  // large dimension.
  // 1. Broadcast the incoming gradient to all shards. This grad is relatively small as it is result of
  //    sparse op (it is going to be scattered into the largest axis of the outgoing grad tensor).
  auto gradient = gradients[0];
  poplar::Tensor gradientShardedCopies;
  prog.add(copyToAll(graph, gradient, gradientShardedCopies, debug_prefix + "/broadcast_grad_to_shards"));

  // 2. Each shard now contains all the grads and all the indices, and a part of the activations 's'. We only want to
  // update indices in s that reside on each shard, so first we need to make adjustments to the indices.
  auto activationsSharded = getShardedTensorChunks(shards, 1, s);
  auto partitionSizes = getPartitionSizes(activationsSharded, 1);
  poplar::Tensor adjustedIndices;
  std::vector<poplar::Tensor> masks;
  std::tie(adjustedIndices, masks) = adjustIndicesAndCreateMasks(
    graph, partitionSizes, indices, prog, debug_prefix + "/adjust_indices");

  // 3. Use adjusted indices to update each shard's partition of s in parallel:
  for (auto c = 0u; c < shards.size(); ++c) {
    auto shardSuffixStr = std::to_string(c);
    auto activation = activationsSharded[c];
    auto grad = gradientShardedCopies[c];
    auto minusOne = graph.addConstant(activation.elementType(), {}, -1.f);
    graph.setTileMapping(minusOne, getFirstTile(graph, activationsSharded[c]));

    // TODO: this is slow to avoid OOM (same as fwds pass):
    for (auto i = 0u; i < activation.dim(0); ++i) {
      auto row = activation[i];
      auto updateable = popops::createSliceableTensor(
        shards[c], row.elementType(), row.shape(), {0}, {1}, {}, {});
      prog.add(poplar::program::Copy(row, updateable, false));

      popops::multiUpdateAdd(shards[c], updateable.expand({1}), grad[i].expand({0, 0}), adjustedIndices[c][i].expand({0}),
                             minusOne, {0}, {1}, prog, {}, {},
                             debug_prefix + "/scatter_gradients_" + shardSuffixStr + "_" + std::to_string(i));
      // Copy result back:
      prog.add(poplar::program::Copy(updateable, row, false));
    }
  }

  outputs.push_back(s);
  return prog;
}

} // end extern "C"
