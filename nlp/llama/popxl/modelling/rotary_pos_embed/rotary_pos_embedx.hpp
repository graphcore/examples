// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ROTARYPOSEMBEDX_HPP
#define GUARD_NEURALNET_ROTARYPOSEMBEDX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>
#include <vector>

namespace popart {
namespace popx {

class RotaryPosEmbedOpx : public Opx {
public:
  RotaryPosEmbedOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const;
};

class RotaryPosEmbedGradOpx : public Opx {
public:
  RotaryPosEmbedGradOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const;
};

} // namespace popx
} // namespace popart

#endif
