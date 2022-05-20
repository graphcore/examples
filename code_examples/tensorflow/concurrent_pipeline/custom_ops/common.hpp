// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#pragma once

#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>

extern "C" {
  poplin::matmul::PlanningCache* getPlanningCache();
}
