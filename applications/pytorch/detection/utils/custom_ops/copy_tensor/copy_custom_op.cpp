// Copyright (c) 2021 Graphcore Ltd. All rights reserved.


#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poputil/TileMapping.hpp>


namespace CustomOperators {
const popart::OperatorIdentifier CopyTensorId = {"ai.graphcore", "CopyTensor", 1};
}
namespace CustomGradOperators {
const popart::OperatorIdentifier CopyTensorGradId = {"ai.graphcore", "CopyTensorGrad", 1};
}

class CopyTensorOp;
class CopyTensorOpx;
class CopyTensorGradOpx;

class CopyTensorGradOp : public popart::Op {
public:
  CopyTensorGradOp(const CopyTensorOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<CopyTensorGradOp>(*this);
  }
  void setup() final { outInfo(0) = inInfo(0); };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const;

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const;

  bool requiresRandomSeed() const override { return false; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  // Implementation defined below
  void appendAttributes(popart::OpSerialiserBase &os) const override;

  // Implementation defined below
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;
};

class CopyTensorOp : public popart::Op {
public:
  CopyTensorOp(const popart::OperatorIdentifier &_opid,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<CopyTensorOp>(*this);
  }

  void setup() final { outInfo(0) = inInfo(0); }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new CopyTensorGradOp(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }
};

namespace {
using popart::OpDefinition;
using popart::DataType;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
      CopyTensorOpDef({OpDefinition::Inputs({{"input", T}}),
                      OpDefinition::Outputs({{"output", T}}),
                      OpDefinition::Attributes({})});


static popart::OpCreator<CopyTensorOp> CopyTensorOpCreator(
      popart::OpDefinitions({{CustomOperators::CopyTensorId, CopyTensorOpDef}}),
      true);
}

class CopyTensorOpx : public popart::popx::Opx {
public:
  CopyTensorOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<CopyTensorOp>(
        op, {CustomOperators::CopyTensorId});
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op = getOp<CopyTensorOp>();
    poplar::Tensor input_ = getInTensor(0);

    poplar::Tensor input;
    input = graph().addVariable(input_.elementType(), input_.shape());
    poputil::mapTensorLinearly(graph(), input, 0, 10);
    prog.add(poplar::program::Copy(input_, input));
    setOutTensor(0, input);
  }
};

class CopyTensorGradOpx : public popart::popx::Opx {
public:
  CopyTensorGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<CopyTensorGradOp>(op, {CustomGradOperators::CopyTensorGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op = getOp<CopyTensorGradOp>();
    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor input_ = getInTensor(1);

    poplar::Tensor input;
    input = graph().addVariable(input_.elementType(), input_.shape());
    poputil::mapTensorLinearly(graph(), input, 0, 10);
    prog.add(poplar::program::Copy(input_, input));

    setOutTensor(0, input);
  }
};

CopyTensorGradOp::CopyTensorGradOp(const CopyTensorOp &fwdOp)
    : popart::Op(CustomGradOperators::CopyTensorGradId, fwdOp.settings) {}

const std::vector<popart::GradInOutMapper> &
CopyTensorGradOp::gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {{0, 0, popart::GradOpInType::GradOut}, {0, 0, popart::GradOpInType::In}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &CopyTensorGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

void CopyTensorGradOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
}

void CopyTensorGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
}

static popart::popx::OpxCreator<CopyTensorOpx> 
    CopyTensorOpxCreator({CustomOperators::CopyTensorId});

static popart::popx::OpxCreator<CopyTensorGradOpx>
    CopyTensorGradOpxCreator({CustomGradOperators::CopyTensorGradId});
