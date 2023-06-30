// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <popart/graphcoreoperators.hpp>
#include <popart/names.hpp>
#include <string>
#include <vector>

#include <memory>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include "common.hpp"
#include "rotary_pos_embed.hpp"

namespace popart {

/////////////////////////////////////////////////////////////
////// Fwd op

RotaryPosEmbedOp::RotaryPosEmbedOp(const OperatorIdentifier &_opid,
                                   uint32_t rotary_dim_,
                                   const Op::Settings &settings_)
    : Op(_opid, settings_), rotary_dim{rotary_dim_} {
  if ((rotary_dim % 2) != 0) {
    throw error("RotaryPosEmbedOp::RotaryPosEmbedOp rotary_dim must be a "
                "multiple of 2");
  }
}

std::unique_ptr<Op> RotaryPosEmbedOp::clone() const {
  return std::make_unique<RotaryPosEmbedOp>(*this);
}

std::vector<std::unique_ptr<Op>> RotaryPosEmbedOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<RotaryPosEmbedGradOp>(*this));
  return result;
}

void RotaryPosEmbedOp::setup() {
  auto xInfo = inInfo(0);
  auto cosInfo = inInfo(1);
  auto sinInfo = inInfo(2);

  // check expected shapes
  if (xInfo.rank() != 4) {
    throw error(
        "RotaryPosEmbedOp::setup x should have rank 4 (batch, heads, seq, hh)");
  }
  if (cosInfo.rank() != 3 || sinInfo.rank() != 3) {
    throw error("RotaryPosEmbedOp::setup trig functions should have rank 3 "
                "(1 or batch, seq, hh/2)");
  }
  if ((rotary_dim % 2) != 0) {
    throw error("RotaryPosEmbedOp::setup rotary dim must be a multiple of 2");
  }

  // x rotated
  outInfo(0) = xInfo;
}

void RotaryPosEmbedOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  os.appendAttribute("rotary_dim", rotary_dim);
  Op::appendOutlineAttributes(os);
}

/////////////////////////////////////////////////////////////
////// Grad op

RotaryPosEmbedGradOp::RotaryPosEmbedGradOp(const RotaryPosEmbedOp &op)
    : Op(RotaryPosEmbedGrad, op.getSettings()), rotary_dim{op.rotary_dim} {}

const std::map<int, int> &RotaryPosEmbedGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

const std::vector<GradInOutMapper> &
RotaryPosEmbedGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GradOut},
      {1, 1, GradOpInType::In},
      {2, 2, GradOpInType::In}};
  return inInfo;
}

void RotaryPosEmbedGradOp::setup() { outInfo(0) = inInfo(0); }

std::unique_ptr<Op> RotaryPosEmbedGradOp::clone() const {
  return std::make_unique<RotaryPosEmbedGradOp>(*this);
}

void RotaryPosEmbedGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  os.appendAttribute("rotary_dim", rotary_dim);
  Op::appendOutlineAttributes(os);
}

} // namespace popart
