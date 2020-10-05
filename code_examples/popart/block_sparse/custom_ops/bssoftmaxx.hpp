// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef GUARD_NEURALNET_BSSOFTMAXX_HPP
#define GUARD_NEURALNET_BSSOFTMAXX_HPP

#include <popart/names.hpp>

namespace popart {
namespace popx {

class BsSoftmaxOpx : public Opx {
public:
  BsSoftmaxOpx(Op *, Devicex *);
  ~BsSoftmaxOpx() override = default;
  void grow(poplar::program::Sequence &) const final;
};

class BsSoftmaxGradOpx : public Opx {
public:
  BsSoftmaxGradOpx(Op *, Devicex *);
  ~BsSoftmaxGradOpx() override = default;
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
