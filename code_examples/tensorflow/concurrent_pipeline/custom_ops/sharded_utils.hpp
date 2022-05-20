// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#pragma once

extern "C" {

poplar::program::Program copyToAll(
  poplar::Graph& graph,
  poplar::Tensor input,
  poplar::Tensor& output,
  const std::string& debug_prefix);

}
