// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#pragma once

#include <poplar/Type.hpp>
#include <popsparse/experimental/BlockSparse.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <vector>
#include <string>

struct BsMatMulArgs {
  std::array<int, 3> dim;
  std::array<int, 3> blockSize;
  std::vector<unsigned char> sparsityMask;
  bool transposedRhs = false;
  poplar::Type dataType = poplar::FLOAT;
  poplar::Type partialDataType = poplar::FLOAT;
  int innerGroupSize = 0;
  std::string partitionMethod = "strip";
  float memoryCycleRatio = 1.0f;
};

BsMatMulArgs parseBsMatMulJsonArgs(const std::string& attributes);

struct BsSoftmaxArgs {
  std::vector<int> dimDense;
  std::array<int, 2> blockSize;
  std::vector<unsigned char> sparsityMask;
  std::vector<popsparse::experimental::SubBlockMask> subBlockMaskType;
  int innerGroupSize = 0;
};

BsSoftmaxArgs parseBsSoftmaxJsonArgs(const std::string& attributes);

spdlog::logger* createLogger();