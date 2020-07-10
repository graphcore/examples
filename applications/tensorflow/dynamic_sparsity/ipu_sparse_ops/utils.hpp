// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <poplar/Graph.hpp>
#include <string>
#include <limits.h>
#include <stdlib.h>
#include <fstream>
#include <experimental/filesystem>

const auto defaults = R"SIC(
  {
    "availableMemoryProportion": "1.0",
    "doGradAPass": "true",
    "doGradWPass": "true",
    "metaInfoBucketOversizeProportion": "0.1"
  }
)SIC";

inline
poplar::OptionFlags getSparseMulDefaultOptions(
    const std::string& jsonConfigFile="sparse_matmul_options.json") {

  poplar::OptionFlags options;
  poplar::readJSON(defaults, options);

  // Overwrite defaults with contents of config file if it exists:
  if (!jsonConfigFile.empty() &&
      std::experimental::filesystem::exists(jsonConfigFile)) {
    std::ifstream f(jsonConfigFile);
    std::string configString((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    poplar::readJSON(configString, options);
  }

  return options;
}
