// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <exception>
#include <iostream>
#include <sstream>

#include <limits>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Zero.hpp>

namespace {
enum class Operation {
  Gather,
  ScatterSum,
  ScatterMax,
};

struct Settings {
  Operation operation;
  poplar::Type type;
  unsigned nGroups;
  unsigned tableSize;
  unsigned embeddingSize;
  unsigned lookupSize;
  poplar::OptionFlags options;
  popops::SlicePlan plan;
};

std::istream &operator>>(std::istream &in, Operation &operation) {
  std::string name;
  in >> name;
  if (name == "gather") {
    operation = Operation::Gather;
  } else if (name == "scatter_sum") {
    operation = Operation::ScatterSum;
  } else if (name == "scatter_max") {
    operation = Operation::ScatterMax;
  } else {
    in.setstate(std::ios::badbit);
  }
  return in;
}

std::istream &operator>>(std::istream &in, poplar::Type &type) {
  std::string name;
  in >> name;
  if (name == "float") {
    type = poplar::FLOAT;
  } else if (name == "half") {
    type = poplar::HALF;
  } else {
    in.setstate(std::ios::badbit);
  }
  return in;
}

Settings parseAndPlan(const std::string &spec, poplar::Graph &graph) {
  Settings settings;
  std::string version;
  std::istringstream in(spec);

  in >> version >> settings.operation >> settings.type >> settings.nGroups >>
      settings.tableSize >> settings.embeddingSize >> settings.lookupSize;

  if (version != "v0" || !in.eof() || in.fail()) {
    std::ostringstream msg;
    msg << "Bad attribute '" << spec
        << "', expected 'v0 <operation> \"float\"|\"half\" <nGroups> "
           "<tableSize> "
           "<embeddingSize> <lookupSize>'";
    throw std::invalid_argument(msg.str());
  }

  settings.plan = popops::embedding::plan(
      graph, settings.type, settings.nGroups, settings.tableSize,
      settings.embeddingSize, {settings.lookupSize}, settings.options);
  return settings;
}
} // namespace

extern "C" {
int32_t custom_op_api_level = 5;
}

extern "C" void Build_metadata(
    std::vector<std::int64_t> &allocating_indices,
    std::vector<std::int64_t> & /*replica_identical_output_indices*/,
    std::map<std::int64_t, std::int64_t> & /*input_to_output_tensor_aliasing*/,
    bool & /*is_elementwise*/, bool &is_stateless, bool &is_hashable,
    std::uint32_t /*num_inputs*/
) {
  allocating_indices = {0, 1};
  is_stateless = true;
  is_hashable = true;
}

extern "C" poplar::Tensor
Build_allocator(poplar::Graph &graph, std::uint32_t operand,
                const std::vector<size_t> & /*shape*/, poplar::Type /*type*/,
                const std::string &attributes, const std::string &debugPrefix) {
  auto settings = parseAndPlan(attributes, graph);
  if (operand == 0 && settings.operation == Operation::Gather) {
    return popops::createGroupedSliceableTensor(
        graph, settings.type, settings.nGroups,
        {settings.tableSize, settings.embeddingSize}, {0}, {1}, settings.plan,
        settings.options, debugPrefix);
  }
  if (operand == 0 && settings.operation == Operation::ScatterSum) {
    return popops::createGroupedSliceTensor(
               graph, settings.type, settings.nGroups,
               {settings.tableSize, settings.embeddingSize}, {0}, {1},
               settings.lookupSize, settings.plan, settings.options,
               debugPrefix)
        .squeeze({2});
  }
  if (operand == 0 && settings.operation == Operation::ScatterMax) {
    return popops::createGroupedSliceTensor(
               graph, settings.type, settings.nGroups,
               {settings.tableSize, settings.embeddingSize}, {0}, {1},
               settings.lookupSize, settings.plan, settings.options,
               debugPrefix)
        .squeeze({2});
  }
  if (operand == 1) {
    return popops::createGroupedIndicesTensor(
               graph, settings.nGroups, {0}, settings.lookupSize, settings.plan,
               settings.options, debugPrefix)
        .squeeze({2});
  }
  std::ostringstream msg;
  msg << "Bad operand index '" << operand << "', expected [0, 2).";
  throw std::invalid_argument(msg.str());
}

extern "C" poplar::program::Program
Build(poplar::Graph &graph, const std::vector<poplar::Tensor> &inputs,
      std::vector<poplar::Tensor> &outputs, const std::string &attributes,
      const std::string &debugPrefix) {
  auto settings = parseAndPlan(attributes, graph);
  poplar::program::Sequence program;

  // Need to reinterpret, as GC-XLA has no uint32
  auto indices = inputs[1].expand({2}).reinterpret(poplar::UNSIGNED_INT);

  if (settings.operation == Operation::Gather) {
    auto params = inputs[0];
    auto output =
        popops::groupedMultiSlice(graph, params, indices, {0}, {1}, program,
                                  settings.plan, settings.options, debugPrefix);
    outputs = {output.squeeze({2})};
  }
  if (settings.operation == Operation::ScatterSum) {
    auto data = inputs[0].expand({2});
    auto output = popops::createGroupedSliceableTensor(
        graph, settings.type, settings.nGroups,
        {settings.tableSize, settings.embeddingSize}, {0}, {1}, settings.plan,
        settings.options, debugPrefix);
    popops::zero(graph, output, program);
    auto scale = graph.addConstant(poplar::FLOAT, {}, 1.0f);
    graph.setTileMapping(scale, 0);
    popops::groupedMultiUpdateAdd(graph, output, data, indices, scale, {0}, {1},
                                  program, settings.plan, settings.options,
                                  debugPrefix);
    outputs = {output};
  }
  if (settings.operation == Operation::ScatterMax) {
    auto data = inputs[0].expand({2});
    auto output = popops::createGroupedSliceableTensor(
        graph, settings.type, settings.nGroups,
        {settings.tableSize, settings.embeddingSize}, {0}, {1}, settings.plan,
        settings.options, debugPrefix);

    popops::fill<float>(graph, output, program,
                        std::numeric_limits<float>::has_infinity
                            ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::lowest(),
                        debugPrefix);
    popops::groupedMultiUpdateMax(graph, output, data, indices, {0}, {1},
                                  program, settings.plan, settings.options,
                                  debugPrefix);

    // do masking at python level for simplicity

    outputs = {output};
  }
  return program;
}
