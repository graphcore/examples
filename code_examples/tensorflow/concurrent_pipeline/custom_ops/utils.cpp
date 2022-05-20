// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "utils.hpp"
#include "common.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popops/Fill.hpp>

#include <boost/property_tree/json_parser.hpp>

GatherAttributes readJsonGatherAttributes(const std::string& attributes) {
  GatherAttributes args;
  try {
    std::stringstream json(attributes);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json, pt);
    args.feature_count = pt.get<std::size_t>("feature_count");
    args.feature_dim = pt.get<std::size_t>("feature_dim");
    args.output_count = pt.get<std::size_t>("output_count");
    args.gradient_scale = pt.get<float>("gradient_scale");
    args.slice_options = pt.get<std::string>("slice_options");
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(
      "Error reading custom op's JSON attributes: ") + e.what()
      + "\nJSON input: " + attributes);
  }

  return args;
}

void checkGatherTensorDimensions(const GatherAttributes& attr,
                           const poplar::Tensor& features,
                           const poplar::Tensor& intIndices) {
  if (features.rank() != 2) {
    throw std::logic_error("Features must be a matrix.");
  }

  if (features.dim(0) != attr.feature_count) {
    throw std::logic_error("Feature count in tensor input does not match JSON attributes.");
  }

  if (features.dim(1) != attr.feature_dim) {
    throw std::logic_error("Feature dimension in tensor input does not match JSON attributes.");
  }

  if (intIndices.dim(0) != attr.output_count) {
    throw std::logic_error("Indices/output count in tensor input does not match JSON attributes.");
  }
}

Attributes readJsonAttributes(const std::string& attributes) {
  Attributes args;
  try {
    std::stringstream json(attributes);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json, pt);
    args.rows_lhs = pt.get<std::size_t>("batch_size");
    args.cols_lhs = pt.get<std::size_t>("input_size");
    args.cols_rhs = pt.get<std::size_t>("output_size");
    args.matmul_options = pt.get<std::string>("matmul_options");
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(
      "Error reading custom op's JSON attributes: ") + e.what()
      + "\nJSON input: " + attributes);
  }

  return args;
}

static const auto matmulDefaults = R"SIC(
  {
    "availableMemoryProportion": 0.6,
    "partialsType": "float"
  }
)SIC";

static const auto sliceDefaults = R"SIC(
  {
    "availableMemoryProportion": 0.6
  }
)SIC";

poplar::OptionFlags getPoplarOptionsFromString(const std::string& jsonOptions, const std::string& defaults) {
  poplar::OptionFlags options;
  poplar::readJSON(defaults, options);

  // Overwrite defaults with contents of config string:
  if (!jsonOptions.empty()) {
    poplar::readJSON(jsonOptions, options);
  }

  return options;
}

poplar::OptionFlags getMatmulOptions(const std::string& jsonOptions) {
  return getPoplarOptionsFromString(jsonOptions, matmulDefaults);
}

poplar::OptionFlags getSliceOptions(const std::string& jsonOptions) {
  return getPoplarOptionsFromString(jsonOptions, sliceDefaults);
}

poplar::Type typeFromString(const std::string& dtype) {
  static const std::map<std::string, poplar::Type> types = {
    {"<dtype: 'float16'>", poplar::HALF},
    {"<dtype: 'float32'>", poplar::FLOAT}
  };

  try {
    return types.at(dtype);
  } catch (const std::exception& e) {
    throw std::runtime_error("Conversion to Poplar type not supported for: " + dtype);
  }
}

std::map<std::size_t, poplar::Interval> getShardInfo(const poplar::Graph& graph) {
  std::map<std::size_t, poplar::Interval> info;
  const auto numIpus = graph.getTarget().getNumIPUs();
  const auto tilesPerIpu = graph.getTarget().getTilesPerIPU();
  std::size_t nextShardStartTile = 0;
  for (auto i = 0u; i < numIpus; ++i) {
    auto nextShardEndTile = nextShardStartTile + tilesPerIpu;
    info.insert(
      std::make_pair(i, poplar::Interval(nextShardStartTile, nextShardEndTile))
    );
    nextShardStartTile = nextShardEndTile;
  }
  return info;
}

std::vector<poplar::Graph> createIpuShards(poplar::Graph& graph) {
  std::vector<poplar::Graph> shard;
  std::size_t nextShardStartTile = 0;
  //std::cerr << "Target num IPUs (num shards): " << graph.getTarget().getNumIPUs() << "\n";
  //std::cerr << "Tiles per IPU: " << graph.getTarget().getTilesPerIPU() << "\n";
  if (graph.getTarget().getNumIPUs() == 1) {
    throw std::logic_error("You cannot use sharded ops with only 1 IPU.");
  }
  for (auto s = 0u; s < graph.getTarget().getNumIPUs(); ++s) {
    auto nextShardEndTile = nextShardStartTile + graph.getTarget().getTilesPerIPU();
    shard.push_back(graph.createVirtualGraph(nextShardStartTile, nextShardEndTile));
    //std::cerr << "Created virtual graph for tiles: [" << nextShardStartTile << ", " << nextShardEndTile << ")\n";
    nextShardStartTile = nextShardEndTile;
  }
  return shard;
}

// Return the set of IPU ids for given number of shards.
std::set<std::size_t> getIpuSet(std::size_t shardCount) {
  std::set<std::size_t> ipus;
  std::vector<int> v(shardCount);
  std::iota(v.begin(), v.end(), 0);
  ipus.insert(v.begin(), v.end());
  return ipus;
}

inline poplar::Interval intersect(const poplar::Interval& a, const poplar::Interval& b) {
  auto begin = std::max(a.begin(), b.begin());
  auto end = std::min(a.end(), b.end());
  if (end < begin) {
    end = begin;
  }
  return poplar::Interval(begin, end);
}

inline bool intersects(const poplar::Interval& a, const poplar::Interval& b) {
  return intersect(a, b).size();
}

poplar::Interval getTileInterval(const poplar::Graph& g, const poplar::Tensor& t) {
  auto mapping = g.getTileMapping(t);
  auto min = std::numeric_limits<std::size_t>::max();
  auto max = std::numeric_limits<std::size_t>::min();
  if (mapping.empty()) {
    throw std::runtime_error("Called getTileInterval() on tensor with no tile mapping.");
  }
  for (auto t = 0u; t < mapping.size(); ++t) {
    if (!mapping[t].empty()) {
      if (t < min) { min = t; }
      if (t > max) { max = t; }
    }
  }
  return poplar::Interval(min, max + 1);
}

std::set<std::size_t> getIPUMapping(const poplar::Graph& g, const poplar::Tensor& t) {
  auto shardInfo = getShardInfo(g);
  auto tileInterval = getTileInterval(g, t);

  std::set<std::size_t> shards;
  for (const auto& s : shardInfo) {
    // If any interval in the Tensor's mapping overlaps the tiles for
    // a particular shard then record it as being on that shard:
    auto intersection = intersect(tileInterval, s.second);
    if (intersection.size()) {
      shards.insert(s.first);
    }
  }

  return shards;
}

void printTileMapping(poplar::Graph& g, poplar::Tensor& t, const std::string& id) {
  auto mapping = g.getTileMapping(t);
  std::cerr << "Tile mapping '" << id << "'\n";
  for (auto t = 0u; t < mapping.size(); t += 1) {
    if (!mapping[t].empty()) {
      std::cerr << "  Tile " << t << " intervals: " << mapping[t] << "\n";
    }
  }
}

// Returns the bounds of the intervals that result from splitting a dimension
// of size `dimSize` into `numShards` partitions (i.e. use this to consistently
// calculate intervals that shard a dimension of particular size across a
// number of IPUs).
std::vector<poplar::Interval> getShardIntervals(std::size_t numShards, std::size_t dimSize) {
  auto sliceSizePerShard = dimSize / numShards;
  // Calculate column partitioning in case the number of columns
  // doesn't divide the number of IPUs exactly:
  std::vector<std::size_t> colCount(numShards, sliceSizePerShard);
  // Put any remainders all on shard 0:
  colCount[0] += dimSize % numShards;
  std::vector<poplar::Interval> intervals;

  auto begin = 0u;
  for (auto c = 0u; c < colCount.size(); ++c) {
    auto end = begin + colCount[c];
    intervals.emplace_back(begin, end);
    begin = end;
  }
  return intervals;
}

/// Allocate the RHS (weights). The weights need to be explicitly sharded over all IPUs
/// by columns (because this implementation is targeting tied embedding projection matrices).
poplar::Tensor allocateShardedWeightMatrix(
  std::vector<poplar::Graph>& shards, const Attributes& attr, poplar::Type type,
  const std::string& prefix)
{
  const auto intervals = getShardIntervals(shards.size(), attr.cols_rhs);
  const auto matmulOpts = getMatmulOptions(attr.matmul_options);
  std::size_t startCol = 0u;
  std::vector<poplar::Tensor> matrices;
  for (auto c = 0u; c < intervals.size(); ++c) {
    auto& shard = shards[c];
    auto rhsColumns = intervals[c].size();
    auto tensor = poplin::createMatMulInputRHS(shard, type,
                                        {attr.rows_lhs, attr.cols_lhs},
                                        {attr.cols_lhs, rhsColumns},
                                        prefix + "/rhs", matmulOpts, getPlanningCache());
    matrices.push_back(tensor);
  }

  return poplar::concat(matrices, 1);
}

// Slice up the specified axis of the tensor across the given shards (IPUs).
std::vector<poplar::Tensor> getShardedTensorChunks(std::vector<poplar::Graph>& shards,
                                        std::size_t axis, poplar::Tensor weights) {
  std::vector<poplar::Tensor> slices;
  const auto intervals = getShardIntervals(shards.size(), weights.dim(axis));

  for (const auto& i : intervals) {
    slices.push_back(weights.slice(i.begin(), i.end(), axis));
  }

  return slices;
}

void checkShardedTensorMapping(poplar::Graph& graph, std::size_t numShards,
                               const std::vector<poplar::Tensor>& shardedTensor) {
  // Check slices are correct:
  if (shardedTensor.size() != numShards) {
    throw std::runtime_error("No. of weight matrix slices does not match no. of shards.");
  }
  for (auto& t : shardedTensor) {
    auto ipus = getIPUMapping(graph, t);
    if (ipus.size() != 1) {
      std::stringstream ss;
      ss << "Tensor slice: '" << t.getDebugStr()
         << "' is sharded on multiple IPUs: " << ipus
        << " (each slice must be on a single IPU).";
      throw std::runtime_error(ss.str());
    }
  }
}

void checkOkForSharding(std::size_t numShards, Attributes& attr) {
  // The columns must divide the number of shards so that the
  // LHS can be laid out exactly the same way for all shards:
  if (attr.cols_rhs % numShards) {
    throw std::runtime_error("No. of RHS columns must be multiple of no. of shards (" + std::to_string(numShards) + ")");
  }
}

void checkExpectedSharding(poplar::Graph& g, poplar::Tensor t, std::set<std::size_t>&& expectedIpus) {
  auto m = getIPUMapping(g, t);
  auto e = std::set<std::size_t>(std::move(expectedIpus));
  if (m != e) {
    std::stringstream ss;
    ss << "Sharding (" << m << ") of tensor " << t.getDebugStr()
       << " does not match expected (" << e << ").";
    throw std::logic_error(ss.str());
  }
}

void checkExpectedSharding(poplar::Graph& g, poplar::Tensor t, std::initializer_list<std::size_t>&& expectedIpus) {
  checkExpectedSharding(g, t, std::set<std::size_t>(std::move(expectedIpus)));
}

std::size_t getFirstTile(poplar::Graph& g, poplar::Tensor t) {
  auto m = g.getTileMapping(t);
  for (auto i = 0u; i < m.size(); ++i) {
    if (!m[i].empty()) {
      return i;
    }
  }

  throw std::runtime_error("Tensor '" + t.getDebugStr() + "' has no tile mapping in this graph.");
}

// Add a vector to the diagonal of the matrix in place.
void addToDiagonal(poplar::Graph& g, poplar::Tensor m, poplar::Tensor v,
                   poplar::program::Sequence& prog, const std::string& debugStr) {
  if (m.shape().size() != 2 || m.dim(0) != m.dim(1)) {
    throw std::logic_error("First tensor in addToDiagonal must be a square matrix.");
  }
  if (v.shape().size() != 1) {
    throw std::logic_error("Second tensor in addToDiagonal mut be a vector.");
  }
  if (m.dim(0) != v.dim(0)) {
    throw std::logic_error("First tensor in addToDiagonal must be a matrix.");
  }

  auto length = v.dim(0);
  // Slice out a view of the diagonal:
  std::vector<poplar::Tensor> diagonalElements(length);
  for (auto i = 0u; i < length; ++i) {
    diagonalElements[i] = m.slice({i, i}, {i+1, i+1}); // Using indexing gives segfault
  }

  auto diagM = poplar::concat(diagonalElements, 1).squeeze({0});
  popops::addInPlace(g, diagM, v, prog, debugStr);
}

std::pair<poplar::Tensor, std::vector<poplar::Tensor>>
adjustIndicesAndCreateMasks(poplar::Graph& graph,
                            std::vector<std::size_t>& partitionSizes,
                            poplar::Tensor indices,
                            poplar::program::Sequence& prog,
                            const std::string& debug_prefix) {
  // IPU XLA supports int but not unsigned int so we need to cast here:
  auto unsignedIndices = popops::cast(graph, indices, poplar::UNSIGNED_INT, prog, debug_prefix + "/cast");
  auto shards = createIpuShards(graph);
  std::vector<poplar::Tensor> masks;
  for (auto s = 0u; s < shards.size(); ++s) {
    auto shardSuffixStr = std::to_string(s);
    // Create shifted indices on each partition where we have subtracted
    // the partition offset from the labels.
    auto size = partitionSizes[s];
    auto zero = shards[s].addConstant(unsignedIndices.elementType(), {}, 0u);
    auto partitionSize = shards[s].addConstant(unsignedIndices.elementType(), {}, size);
    auto partitionStart = shards[s].addConstant(unsignedIndices.elementType(), {}, s * size);
    auto firstTileOfInput = getFirstTile(shards[s], unsignedIndices[s]);
    shards[s].setTileMapping(partitionStart, firstTileOfInput);
    shards[s].setTileMapping(zero, firstTileOfInput);
    shards[s].setTileMapping(partitionSize, firstTileOfInput);
    popops::subInPlace(shards[s], unsignedIndices[s], partitionStart, prog, debug_prefix + "/shift_" + shardSuffixStr);

    // From the shifted labels create a mask on each shard to remove labels < 0 or >= labels-per-shard
    // (we need this because IPU gathers do not return 0 for invalid indices).
    namespace pe = popops::expr;
    auto betweenExpr = pe::And(pe::Gte(pe::_1, pe::_2), pe::Lt(pe::_1, pe::_3));
    auto operands = {unsignedIndices[s], zero, partitionSize};
    auto mask = popops::map(shards[s], betweenExpr, operands, prog, debug_prefix + "/generate_mask_" + shardSuffixStr);
    masks.push_back(mask);
  }

  return std::make_pair(unsignedIndices, masks);
}

poplar::Tensor zerosLike(poplar::Graph& g, poplar::Tensor t, poplar::program::Sequence& p) {
  auto z = g.clone(t);
  popops::fill(g, z, p, 0.f);
  return z;
}

std::vector<std::size_t> getPartitionSizes(const std::vector<poplar::Tensor>& t, std::size_t partitionedAxis) {
  std::vector<std::size_t> partitionSizes;
  for (const auto& p : t) {
    partitionSizes.push_back(p.dim(partitionedAxis));
  }
  return partitionSizes;
}
