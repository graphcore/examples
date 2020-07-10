// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <poputil/exceptions.hpp>
#include <popsparse/FullyConnected.hpp>
#include "utils.hpp"

extern "C" {

/// We are using a stateless op which requires
/// API level 1 or higher.
int32_t custom_op_api_level = 1;

popsparse::dynamic::PlanningCache sparse_matmul_cache;

/// Meta data function sets properties of the forward op.
void Build_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::uint32_t& num_inplace,
                    bool& is_elementwise,
                    bool& is_stateless,
                    std::uint32_t num_inputs) {
  allocating_indices.clear();
  num_inplace = 0;
  is_elementwise = false;
  is_stateless = true;
}

/// This op is for allocation. At compile time it creates the tensors we
/// need for a sparse matmul of a particular shape but it adds no ops to the
/// compute graph so there will be no associated runtime op in the graph:
poplar::program::Program Build(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs, const std::string& debug_prefix) {

  auto args = inputs[0];
  auto lhs = inputs[1];
  auto batchSize = lhs.dim(0);
  auto inputSize = lhs.dim(1);
  auto outputSize = args.dim(0);
  auto sparsityFactor = args.dim(1) / float(outputSize*inputSize);
  auto numGroups = args.dim(2);

  poplar::OptionFlags options = getSparseMulDefaultOptions();

  using namespace popsparse::dynamic;

  SparsityParams sparsityParams(SparsityType::Element, SparsityStructure::Unstructured);
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);
  auto dataType = inputs[1].elementType();

  const SparseTensor rhs = createFullyConnectedWeights(
        graph, dataType, params, "sparse_fc_weights", options, &sparse_matmul_cache);

  auto dense_lhs = createFullyConnectedInput(
      graph, dataType, params, "dense_lhs", options, &sparse_matmul_cache);

  // Can't return the meta-info as uint16 because GC-TF doesn't
  // support it, only 16-bit type is fp16 so we reinterpret
  // cast here.
  //
  // The meta-info Tensor needs to be cast back before access e.g.
  // using numpy.ndarray.view on the host or poplar::Tensor::reinterpret
  // again on the device:
  outputs.push_back(rhs.getMetaInfoTensor().reinterpret(poplar::HALF));
  outputs.push_back(rhs.getNzValuesTensor());
  outputs.push_back(dense_lhs);

  poplar::program::Sequence prog;
  return prog;
}

} // end extern "C"
