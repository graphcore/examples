// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/opidentifier.hpp>

namespace popart {

  namespace CustomOperators {
    const OperatorIdentifier bsMatMul = {"ai.graphcore", "BSMatMul", 1};
    const OperatorIdentifier BsSoftmax = {"ai.graphcore", "BsSoftmax", 1};

  } // namespace CustomOperators

  namespace CustomGradOperators {
    const OperatorIdentifier bsMatMulGrad = {"ai.graphcore", "BSMatMulGrad", 1};
    const OperatorIdentifier BsSoftmaxGrad = {"ai.graphcore", "BsSoftmaxGrad", 1};
  }	// namespace CustomGradOperators

} // namespace popart

// Include all source files below
#include "bsmatmul.cpp"  // contains both op and opx
#include "bssoftmax.cpp" // contains op
#include "bssoftmaxx.cpp" // contains opx
