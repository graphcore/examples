// Copyright 2019 Graphcore Ltd.
#include <popart/op.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>

#include <poplin/ConvUtil.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier AttemptRegroup = {"ai.graphcore", "AttemptRegroup", 1};
} // namespace CustomOperators

class AttemptRegroupOp : public popart::Op {
public:
  AttemptRegroupOp(const popart::Op::Settings &settings_)
      : popart::Op(CustomOperators::AttemptRegroup, settings_) {}

  void setup() final { outInfo(0) = inInfo(0); }

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<AttemptRegroupOp>(*this);
  }

  static popart::InIndex getInTensorIndex() { return 0; }
  static popart::InIndex getRefTensorIndex() { return 1; }
  static popart::OutIndex getRegroupedOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class AttemptRegroupOpx : public popart::popx::Opx
{
public:
  AttemptRegroupOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex) {
    verifyOp<AttemptRegroupOp>(op, CustomOperators::AttemptRegroup);
    auto attemptRegroup_op = dynamic_cast<AttemptRegroupOp *>(op_p);
  }

  popart::popx::InputCreatorType getInputCreatorType(popart::InIndex index) const {
    return popart::popx::Opx::getInputCreatorType(index);
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto out = poplin::regroupIfBeneficial(
        graph(),
        getInTensor(AttemptRegroupOp::getInTensorIndex()),
        getInTensor(AttemptRegroupOp::getRefTensorIndex()),
        prog,
        debugPrefix("regroup"));
    setOutTensor(AttemptRegroupOp::getRegroupedOutIndex(), out);
  }
};

static popart::popx::OpxCreator<AttemptRegroupOpx>
    attemptRegroupOpxCreator(CustomOperators::AttemptRegroup);
