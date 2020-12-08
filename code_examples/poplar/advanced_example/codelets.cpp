// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>

using namespace poplar;

class VectorAdd : public Vertex {
public:
    Input<Vector<float>> x;
    Input<Vector<float>> y;
    Output<Vector<float>> z;

    bool compute() {
        for (auto i = 0u; i < x.size(); ++i) {
            z[i] = x[i] + y[i];
        }
        return true;
    }
};
