// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>

class SumVertex : public poplar::Vertex {
public:
  // Fields
  poplar::Input<poplar::Vector<float>> in;
  poplar::Output<float> out;

  // Compute function
  bool compute() {
    *out = 0;
    for (const auto &v : in) {
      *out += v;
    }
    return true;
  }
};
