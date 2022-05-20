// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popops/Fill.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>

#include <sstream>

#include "utils.hpp"
#include "common.hpp"

extern "C" {

/// Distribute to all copies the input tensor to all shards.
/// The input tensor must be on a single shard or an error will be thrown.
/// In the backwards pass it is assumed that the next downstream op will compute
/// partial gradients on each shard so the backwards pass for this op is simply to
/// reduce them back onto the original shard.
///
/// Note: If the nature of the downstream op makes it more efficient to return the
/// total grad on one shard, then that op should do this broadcast in its own
/// forward pass (i.e. this op is not relevant for that use case as it stands).
void copy_to_all_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs)
{
  allocating_indices = {0};
  is_stateless = true;
}

poplar::Tensor copy_to_all_allocator(
  poplar::Graph& graph, std::uint32_t operand,
  const std::vector<size_t>& shape,
  poplar::Type type,
  const std::string& attributes,
  const std::string& prefix) {

  auto shards = createIpuShards(graph);

  if (operand == 0) {
    // Input should be entirely on a single IPU so map all to IPU 0
    // (but usually it will come from output of another stage):
    auto t = shards[0].addVariable(type, shape);
    poputil::mapTensorLinearly(shards[0], t);
    return t;
  }

  throw std::logic_error("Tensor allocation requested for unexpected operand.");

  // Unreachable
  return poplar::Tensor();
}

// Function that does work of copy to all but with API
// simplified compared to TF custom op API (so it can be
// called neatly by other custom ops):
poplar::program::Program copyToAll(
  poplar::Graph& graph,
  poplar::Tensor input,
  poplar::Tensor& output,
  const std::string& debug_prefix) {

  auto prog = poplar::program::Sequence();

  // Copy to all shards, expand first dim so it acts as the shard/IPU index:
  auto shards = createIpuShards(graph);
  auto inputIpus = getIPUMapping(graph, input);
  if (inputIpus.size() != 1) {
    std::stringstream ss;
    ss << "In op '" << debug_prefix << "' the input tensor '" << input.getDebugStr()
       << "' is already sharded on IPUs " << inputIpus;
    printTileMapping(graph, input, input.getDebugStr());
    throw std::logic_error(ss.str());
  }
  auto inputIpu = *inputIpus.begin();
  std::vector<poplar::Tensor> copies;
  auto inputTileMapping = shards[inputIpu].getTileMapping(input);
  for (auto i = 0; i < shards.size(); ++i) {
    if (i == inputIpu) {
      copies.push_back(input.expand({0}));
    } else {
      auto t = shards[inputIpu].clone(input);
      shards[i].setTileMapping(t, inputTileMapping);
      copies.push_back(t.expand({0}));
      prog.add(poplar::program::Copy(input, t));
    }
  }

  output = poplar::concat(copies, 0);
  return prog;
}

poplar::program::Program copy_to_all(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debug_prefix)
{
  poplar::Tensor output;
  auto prog = copyToAll(graph, inputs[0], output, debug_prefix);
  outputs.push_back(output);
  return prog;
}

void copy_to_all_grad_metadata(std::vector<std::int64_t>& allocating_indices,
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

poplar::program::Program copy_to_all_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& attributes,
    const std::string& debug_prefix)
{
  auto gradOfLossWrtInput = graph.clone(fwd_inputs[0]);
  graph.setTileMapping(gradOfLossWrtInput, graph.getTileMapping(fwd_inputs[0]));
  poplar::program::Sequence prog;
  popops::reduceWithOutput(graph, gradients[0], gradOfLossWrtInput, {0},
                           popops::ReduceParams(popops::Operation::ADD, false), prog,
                           debug_prefix + "/dLdI_accumulate_partials");
  outputs.push_back(gradOfLossWrtInput);
  // dL/dInput should now be entirely on the original IPU (note that we
  // checked the input is on a single IPU in forward pass):
  checkExpectedSharding(graph, gradOfLossWrtInput, getIPUMapping(graph, fwd_inputs[0]));
  return prog;
}

} // end extern C
