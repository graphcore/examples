// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <poplar/Graph.hpp>
#include <string>
#include <limits.h>
#include <stdlib.h>
#include <fstream>
#include <boost/property_tree/json_parser.hpp>

// Fwd declarations:
namespace poplin {
  namespace matmul {
    class PlanningCache;
  }
}

const auto sparse_defaults = R"SIC(
  {
    "availableMemoryProportion": 0.6,
    "doGradAPass": "true",
    "doGradWPass": "true",
    "metaInfoBucketOversizeProportion": 0.1,
    "partialsType": "float"
  }
)SIC";

const auto dense_defaults = R"SIC(
  {
    "availableMemoryProportion": 0.1,
    "partialsType": "float"
  }
)SIC";

struct SparseArgs {
  std::size_t block_size;
  std::size_t batch_size;
  std::size_t input_size;
  std::size_t output_size;
  std::size_t max_non_zeros;
  std::size_t group_size;
  poplar::Type data_type;
  std::string matmul_options;
  std::string dense_grad_matmul_options;
  std::string pooling_type;
  float embedding_grad_scale;
  bool debug_printing;
};

poplar::Type type_from_string(const std::string& dtype);

float computeDensity(const SparseArgs& args);

SparseArgs read_json_args(const std::string& attributes);

poplar::OptionFlags getSparseMulDefaultOptions(const std::string& jsonOptions);
poplar::OptionFlags getDenseGradMulDefaultOptions(const std::string& jsonOptions);

poplar::Tensor pool(
    poplar::Graph& graph,
    std::string poolingType,
    std::size_t blockSize,
    const poplar::Tensor gradient,
    poplar::program::Sequence &prog,
    const std::string& debug_prefix);

poplar::Tensor serializedMatmul(
  poplar::Graph& graph, poplar::program::Sequence& prog,
  poplar::Tensor& A, poplar::Tensor& B,
  std::size_t inSplits, std::size_t outSplits,
  const std::string& debug_prefix,
  bool enableSerialization,
  const poplar::OptionFlags& options,
  poplin::matmul::PlanningCache*);

poplar::Tensor topKIndices(
    poplar::Graph& graph,
    unsigned k,
    const poplar::Tensor gradient,
    poplar::program::Sequence &prog,
    const std::string& debug_prefix);
