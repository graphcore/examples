// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STRIDEDSLICE_HPP
#define GUARD_NEURALNET_STRIDEDSLICE_HPP

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

#include "common.hpp"

namespace popart {

class RotaryPosEmbedOp : public Op {
public:
  RotaryPosEmbedOp(const OperatorIdentifier &_opid, uint32_t rotary_dim_,
                   const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() final;

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  static RotaryPosEmbedOp *
  createOpInGraph(popart::Graph &graph, const InMapType &in,
                  const OutMapType &out, uint32_t rotary_dim_,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<RotaryPosEmbedOp>(in, out, RotaryPosEmbed,
                                                     rotary_dim_, settings);
  }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  uint32_t rotary_dim = 0;
};

class RotaryPosEmbedGradOp : public Op {
public:
  RotaryPosEmbedGradOp(const RotaryPosEmbedOp &op);

  void setup() final;
  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  uint32_t rotary_dim = 0;
};

} // namespace popart

#endif
