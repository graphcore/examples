// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "bsutils.hpp"

#include <poputil/exceptions.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>
#include <stdlib.h>
#include <boost/property_tree/json_parser.hpp>
#include <spdlog/sinks/ansicolor_sink.h>
#include <spdlog/sinks/file_sinks.h>

BsMatMulArgs parseBsMatMulJsonArgs(const std::string& attributes) {
  std::string errorPrefix = "Error reading custom op's JSON attributes: ";
  BsMatMulArgs args;
  try {
    std::stringstream json_args(attributes);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json_args, pt);

    auto nodeIter = pt.get_child("dim");
    std::vector<int> dimsTmp;
    for (auto& item : nodeIter) {
      dimsTmp.push_back(item.second.get_value<int>());
    }
    if (dimsTmp.size() != 3) {
      throw poputil::poplibs_error(errorPrefix + "'dim' node must be an array of 3 elements.\n" +
        "JSON input: " + attributes);  
    }
    for (int i = 0; i < 3; ++i) {
      args.dim[i] = dimsTmp[i];
    }

    nodeIter = pt.get_child("block_size");
    std::vector<int> blockSizeTmp;
    for (auto& item : nodeIter) {
      blockSizeTmp.push_back(item.second.get_value<int>());
    }
    if (blockSizeTmp.size() != 3) {
      throw poputil::poplibs_error(errorPrefix + "'block_size' node must be an array of 3 elements.\n" +
        "JSON input: " + attributes);
    }
    for (int i = 0; i < 3; ++i) {
      args.blockSize[i] = blockSizeTmp[i];
    }

    std::string sparsityStr = pt.get<std::string>("sparsity_mask");
    for (auto iter = sparsityStr.begin(); iter != sparsityStr.end(); ++iter) {
      args.sparsityMask.push_back(*iter - '0');
    }

    args.transposedRhs = pt.get("transposed_rhs", args.transposedRhs);

    std::string dataTypeStr = pt.get("data_type", std::string(args.dataType.toString()));
    if (dataTypeStr == "half") {
      args.dataType = poplar::HALF;
    } else if (dataTypeStr == "float") {
      args.dataType = poplar::FLOAT;
    } else {
      throw poputil::poplibs_error(errorPrefix + "'data_type' must be 'float' or 'half'.\n" +
        "JSON input: " + attributes);
    }

    std::string partialDataTypeStr = pt.get("partial_data_type", std::string(args.partialDataType.toString()));
    if (partialDataTypeStr == "half") {
      args.partialDataType = poplar::HALF;
    } else if (partialDataTypeStr == "float") {
      args.partialDataType = poplar::FLOAT;
    } else {
      throw poputil::poplibs_error(errorPrefix + "'partial_data_type' must be 'float' or 'half'.\n" +
        "JSON input: " + attributes);
    }

    args.innerGroupSize = pt.get("inner_group_size", args.innerGroupSize);
    if (args.innerGroupSize < 0) {
      throw poputil::poplibs_error(errorPrefix + "incorrect 'inner_group_size' parameter.\n" +
        "JSON input: " + attributes);
    }

    args.partitionMethod = pt.get("partition_method", args.partitionMethod);

    args.memoryCycleRatio = pt.get("memory_cycle_ratio", args.memoryCycleRatio);
  } catch (const std::exception& e) {
    throw std::runtime_error(errorPrefix + e.what() + "\n" +
      "JSON input: " + attributes);
  }
  return args;
}

BsSoftmaxArgs parseBsSoftmaxJsonArgs(const std::string& attributes) {
  std::string errorPrefix = "Error reading custom op's JSON attributes: ";
  BsSoftmaxArgs args;
  try {
    std::stringstream json_args(attributes);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json_args, pt);

    auto nodeIter = pt.get_child("dim_dense");
    for (auto& item : nodeIter) {
      args.dimDense.push_back(item.second.get_value<int>());
    }
    if (args.dimDense.size() < 2) {
      throw poputil::poplibs_error(errorPrefix + "'dim_dense' node must be an array of at least 2 elements.\n" +
        "JSON input: " + attributes);
    }

    nodeIter = pt.get_child("block_size");
    std::vector<int> blockSizeTmp;
    for (auto& item : nodeIter) {
      blockSizeTmp.push_back(item.second.get_value<int>());
    }
    if (blockSizeTmp.size() != 2) {
      throw poputil::poplibs_error(errorPrefix + "'block_size' node must be an array of 2 elements.\n" +
        "JSON input: " + attributes);
    }
    for (int i = 0; i < 2; ++i) {
      args.blockSize[i] = blockSizeTmp[i];
    }

    std::string sparsityStr = pt.get<std::string>("sparsity_mask");
    for (auto iter = sparsityStr.begin(); iter != sparsityStr.end(); ++iter) {
      args.sparsityMask.push_back(*iter - '0');
    }

    nodeIter = pt.get_child("subblock_mask_type");
    for (auto& item : nodeIter) {
      popsparse::experimental::SubBlockMask subBlockMaskType = static_cast<popsparse::experimental::SubBlockMask>(item.second.get_value<int>());
      if (subBlockMaskType < popsparse::experimental::SubBlockMask::None || subBlockMaskType > popsparse::experimental::SubBlockMask::ZeroLowerTriangle) {
        throw poputil::poplibs_error(errorPrefix + "Incorrect 'subblock_mask_type'.\n" +
          "JSON input: " + attributes);
      }
      args.subBlockMaskType.push_back(subBlockMaskType);
    }

    args.innerGroupSize = pt.get("inner_group_size", args.innerGroupSize);
    if (args.innerGroupSize < 0) {
      throw poputil::poplibs_error(errorPrefix + "incorrect 'inner_group_size' parameter.\n" +
        "JSON input: " + attributes);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(errorPrefix + e.what() + "\n" +
      "JSON input: " + attributes);
  }
  return args;
}

spdlog::logger* createLogger() {
  static std::shared_ptr<spdlog::logger> spLogger;
  if (spLogger) {
    return spLogger.get();
  }

  spdlog::sink_ptr sink;

  auto POPSPARSE_LOG_DEST = std::getenv("POPSPARSE_LOG_DEST");
  auto POPSPARSE_LOG_LEVEL = std::getenv("POPSPARSE_LOG_LEVEL");
  std::string logLevel = POPSPARSE_LOG_LEVEL ? POPSPARSE_LOG_LEVEL : "OFF";
  std::string logDest = POPSPARSE_LOG_DEST ? POPSPARSE_LOG_DEST : "stderr";
  if (logDest == "stdout") {
    auto colouredSink =
        std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    sink = colouredSink;
  } else if (logDest == "stderr") {
    auto colouredSink =
        std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();
    sink = colouredSink;
  } else {
    try {
      sink =
          std::make_shared<spdlog::sinks::simple_file_sink_mt>(logDest, true);
    } catch (const spdlog::spdlog_ex &e) {
      std::cerr << "Error opening log file: " << e.what() << std::endl;
      throw;
    }
  }

  spLogger = std::make_unique<spdlog::logger>("POPSPARSE.TF", sink);
  if (logLevel == "TRACE") {
    spLogger->set_level(spdlog::level::trace);
  } else if (logLevel == "DEBUG") {
    spLogger->set_level(spdlog::level::debug);
  } else if (logLevel == "INFO") {
    spLogger->set_level(spdlog::level::info);
  } else if (logLevel == "WARN") {
    spLogger->set_level(spdlog::level::warn);
  } else if (logLevel == "ERR") {
    spLogger->set_level(spdlog::level::err);
  } else if (logLevel == "OFF" || logLevel == "") {
    spLogger->set_level(spdlog::level::off);
  } else {
    throw ::poputil::poplibs_error(
      "Unknown POPSPARSE_LOG_LEVEL '" + logLevel +
      "'. Valid values are TRACE, DEBUG, INFO, WARN, ERR and OFF.");
  }
  spLogger->set_pattern("%T.%e %t PL:%n [%L] %v");

  return spLogger.get();
}

