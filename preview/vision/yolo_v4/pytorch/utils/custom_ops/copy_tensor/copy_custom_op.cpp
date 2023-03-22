// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/operators.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>
#include <poputil/TileMapping.hpp>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
namespace popart {
namespace CustomOperators {
const OperatorIdentifier CopyTensorId = {"ai.graphcore", "CopyTensor", 1};
}
class CopyTensorOp : public Op {
public:
  CopyTensorOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
      : Op(_opid, settings_) {}
  std::unique_ptr<Op> clone() const final {
    return std::make_unique<CopyTensorOp>(*this);
  }

  void setup() final { outInfo(0) = inInfo(0); };

  std::vector<std::unique_ptr<Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(std::make_unique<popart::IdentityGradOp>(settings));
    return upops;
  }
  const std::vector<GradInOutMapper> &gradInputInfo() const {
    static const std::vector<GradInOutMapper> inInfo = {
        {0, 0, GradOpInType::GradOut}};
    return inInfo;
  };
  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  };
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  bool requiresRandomSeed() const override { return false; }
};
namespace {
static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition CopyTensorOpDef({OpDefinition::Inputs({{"input", T}}),
                                     OpDefinition::Outputs({{"output", T}}),
                                     OpDefinition::Attributes({})});
static OpCreator<CopyTensorOp> CopyTensorOpCreator(
    OpDefinitions({{CustomOperators::CopyTensorId, CopyTensorOpDef}}), true);
} // namespace
class CopyTensorOpx : public popx::PopOpx {
public:
  CopyTensorOpx(Op *op, popx::Devicex *devicex) : popx::PopOpx(op, devicex) {
    verifyOp<CopyTensorOp>(op, {CustomOperators::CopyTensorId});
  }
  void grow(snap::program::Sequence &prog) const final {
    auto op = getOp<CopyTensorOp>();
    snap::Tensor input = getInTensor(0);
    poplar::Tensor output = graph().getPoplarGraph().addVariable(
        input.elementType(), input.shape());
    poputil::mapTensorLinearly(graph().getPoplarGraph(), output, 0, 10);
    snap::Tensor snap_output{output, graph()};
    snap::program::Copy copyProg(getInTensor(0), snap_output, false,
                                 debugContext("CustomCopy"));
    prog.add(copyProg);
    setOutTensor(0, snap_output);
  }

  popart::popx::InputCreatorType getInputCreatorType(InIndex index) const {
    return popart::popx::InputCreatorType::CanUnwind;
  }
  snap::Tensor unwindTensorLayout(snap::Tensor tensor, InIndex,
                                  OutIndex) const {
    return tensor;
  }

  view::RegMap unwindRegion(InIndex, OutIndex) const {
    return [](const view::Region &r) { return view::Regions(1, r); };
  }
};
static popx::OpxCreator<CopyTensorOpx>
    CopyTensorOpxCreator({CustomOperators::CopyTensorId});
} // namespace popart
