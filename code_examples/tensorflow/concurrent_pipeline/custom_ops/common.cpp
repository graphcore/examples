// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "common.hpp"

extern "C" {

/// The API level must be comaptible with the version of the Poplar SDK
/// that you are using. Please check the docs for your SDK version:
int32_t custom_op_api_level = 5;

poplin::matmul::PlanningCache* getPlanningCache() {
  thread_local poplin::matmul::PlanningCache threads_cache;
  return &threads_cache;
}

}
