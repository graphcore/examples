// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <poplar/Graph.hpp>

#include <string>
#include <iostream>
#include <set>

struct Attributes {
  std::size_t rows_lhs;
  std::size_t cols_lhs;
  std::size_t cols_rhs;
  std::string matmul_options;
};

struct GatherAttributes {
  std::size_t feature_count;
  std::size_t feature_dim;
  std::size_t output_count;
  float gradient_scale;
  std::string slice_options;
};

__attribute__((visibility("hidden")))
GatherAttributes readJsonGatherAttributes(const std::string& attributes);

__attribute__((visibility("hidden")))
void checkGatherTensorDimensions(const GatherAttributes& attr,
                           const poplar::Tensor& features,
                           const poplar::Tensor& intIndices);

__attribute__((visibility("hidden")))
Attributes readJsonAttributes(const std::string& attributes);

__attribute__((visibility("hidden")))
poplar::OptionFlags getMatmulOptions(const std::string& jsonOptions);

__attribute__((visibility("hidden")))
poplar::OptionFlags getSliceOptions(const std::string& jsonOptions);

__attribute__((visibility("hidden")))
poplar::Type typeFromString(const std::string& dtype);

std::vector<poplar::Graph> createIpuShards(poplar::Graph& graph);

std::set<std::size_t> getIpuSet(std::size_t);

std::set<std::size_t> getIPUMapping(const poplar::Graph& g, const poplar::Tensor& t);

void printTileMapping(poplar::Graph& g, poplar::Tensor& t, const std::string& id);

std::vector<poplar::Interval> getShardIntervals(std::size_t numShards, std::size_t dimSize);

poplar::Tensor allocateShardedWeightMatrix(
  std::vector<poplar::Graph>& shards, const Attributes& attr, poplar::Type type,
  const std::string& prefix);

std::vector<poplar::Tensor> getShardedTensorChunks(
  std::vector<poplar::Graph>& shards, std::size_t axis, poplar::Tensor t);

void checkShardedTensorMapping(poplar::Graph& graph, std::size_t numShards,
                               const std::vector<poplar::Tensor>& shardedWeights);

void checkOkForSharding(std::size_t numShards, Attributes& attr);

void checkExpectedSharding(poplar::Graph& g, poplar::Tensor t, std::set<std::size_t>&& expectedIpus);
void checkExpectedSharding(poplar::Graph& g, poplar::Tensor t, std::initializer_list<std::size_t>&& expectedIpus);

std::size_t getFirstTile(poplar::Graph& graph, poplar::Tensor tensor);

void addToDiagonal(poplar::Graph& g, poplar::Tensor m, poplar::Tensor v,
                   poplar::program::Sequence& prog, const std::string& debugStr);

std::pair<poplar::Tensor, std::vector<poplar::Tensor>>
adjustIndicesAndCreateMasks(poplar::Graph& graph,
                            std::vector<std::size_t>& partitionSizes,
                            poplar::Tensor indices,
                            poplar::program::Sequence& prog,
                            const std::string& debug_prefix);

poplar::Tensor zerosLike(poplar::Graph& g, poplar::Tensor t, poplar::program::Sequence& p);

std::vector<std::size_t> getPartitionSizes(const std::vector<poplar::Tensor>& t, std::size_t partitionedAxis);
template <typename T>
inline std::ostream& operator << (std::ostream& s, const std::vector<T>& v) {
  for (auto& d : v) {
    s << d << " ";
  }
  return s;
}

template <typename T>
inline std::ostream& operator << (std::ostream& os, const std::set<T>& s) {
  for (auto& k : s) {
    os << k << " ";
  }
  return os;
}
