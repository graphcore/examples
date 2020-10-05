// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <poplar/Graph.hpp>
#include <string>
#include <limits.h>
#include <stdlib.h>
#include <fstream>
#include <boost/property_tree/json_parser.hpp>

const auto defaults = R"SIC(
  {
    "availableMemoryProportion": 0.6,
    "doGradAPass": "true",
    "doGradWPass": "true",
    "metaInfoBucketOversizeProportion": 0.1
  }
)SIC";

inline
poplar::Type type_from_string(const std::string& dtype) {
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

struct SparseArgs {
  std::size_t batch_size;
  std::size_t input_size;
  std::size_t output_size;
  std::size_t max_non_zeros;
  std::size_t group_size;
  poplar::Type data_type;
  std::string matmul_options;
};

inline
SparseArgs read_json_args(const std::string& attributes) {
  SparseArgs args;
  try {
    std::stringstream json_args(attributes);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json_args, pt);
    args.batch_size = pt.get<std::size_t>("batch_size");
    args.input_size = pt.get<std::size_t>("input_size");
    args.output_size = pt.get<std::size_t>("output_size");
    args.max_non_zeros = pt.get<std::size_t>("max_non_zeros");
    args.group_size = pt.get<std::size_t>("group_size");
    args.data_type = type_from_string(pt.get<std::string>("data_type"));
    args.matmul_options = pt.get<std::string>("matmul_options");
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(
      "Error reading custom op's JSON attributes: ") + e.what()
      + "\nJSON input: " + attributes);
  }

  return args;
}

inline
poplar::OptionFlags getSparseMulDefaultOptions(const std::string& jsonOptions) {
  poplar::OptionFlags options;
  poplar::readJSON(defaults, options);

  // Overwrite defaults with contents of config string:
  if (!jsonOptions.empty()) {
    poplar::readJSON(jsonOptions, options);
  }

  return options;
}
