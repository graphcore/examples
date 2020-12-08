// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

using namespace poplar;

template <typename FPType>
class VectorAdd : public Vertex {
public:
    Vector<Input<Vector<FPType>>> x;
    Vector<Input<Vector<FPType>>> y;
    Vector<Output<Vector<FPType>>> z;

    bool compute() {
        for (unsigned i = 0; i < x.size(); ++i) {
            for (unsigned j = 0; j != x[i].size(); ++j) {
                z[i][j] = x[i][j] + y[i][j];
            }
        }
        return true;
    }
};

template class VectorAdd<float>;
template class VectorAdd<half>;
