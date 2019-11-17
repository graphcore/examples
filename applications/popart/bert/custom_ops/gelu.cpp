// Copyright 2019 Graphcore Ltd.
#include <cmath>
#include <memory>
#include <iostream>

#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/half.hpp>
#include <popart/ir.hpp>

#include <popnn/NonLinearity.hpp>
#include <poplin/ConvUtil.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier Gelu = {"ai.graphcore", "Gelu", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
  const popart::OperatorIdentifier GeluGrad = {"ai.graphcore", "GeluGrad", 1};
} // namespace CustomGradOperators

class GeluOp;
class GeluGradOp;
class GeluOpx;
class GeluGradOpx;

class GeluGradOp : public popart::Op {
public:

  GeluGradOp(const popart::Op &fwdOp)
    : popart::Op(CustomGradOperators::GeluGrad, fwdOp.settings) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<GeluGradOp>(*this);
  }

  void setup() final {
    outInfo(0) = inInfo(0);
  };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {0, 0, popart::GradOpInType::GRADOUT},
        {1, 0, popart::GradOpInType::IN}};
    return inInfo;
  }

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }

  bool requiresRandomSeed() const override { return true; }
  popart::InIndex getSeedInIndex() const override { return 6; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

class GeluOp : public popart::Op {
public:

  GeluOp(const popart::OperatorIdentifier &_opid,
         const popart::Op::Settings &settings_)
    : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<GeluOp>(*this);
  }

  void setup() final {
    outInfo(0) = inInfo(0);
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new GeluGradOp(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return true; }
  popart::InIndex getSeedInIndex() const override { return 2; }
};

static popart::OpCreator<GeluOp> geluOpCreator(CustomOperators::Gelu);

class GeluOpx : public popart::popx::Opx {
public:
  GeluOpx(popart::Op *op, popart::popx::Devicex *devicex)
    : popart::popx::Opx(op, devicex) {
      verifyOp<GeluOp>(op, CustomOperators::Gelu);
  }

  void grow(poplar::program::Sequence &prog) const final {
      poplar::Tensor input = getInTensor(0);
      auto nonLinearityOutput =
          popnn::nonLinearity(graph(), popnn::NonLinearityType::GELU,
                               input, prog, debugPrefix("gelu"));
      setOutTensor(0, nonLinearityOutput);
  }
};

class GeluGradOpx : public popart::popx::Opx {
public:
  GeluGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<GeluGradOp>(op, CustomGradOperators::GeluGrad);
  }

  void grow(poplar::program::Sequence &prog) const final {
      poplar::Tensor grad = getInTensor(0);
      poplar::Tensor input = getInTensor(1);

      auto gradRearranged =
          poplin::regroupIfBeneficial(graph(), grad, input, prog, debugPrefix("regroup"));

      auto inputGrad =
          popnn::nonLinearityInputGradient(graph(), popnn::NonLinearityType::GELU,
                                    input, gradRearranged, prog, debugPrefix("geluGrad"));

      setOutTensor(0, inputGrad);
  }
};

static popart::popx::OpxCreator<GeluOpx> GeluOpxCreator(CustomOperators::Gelu);
static popart::popx::OpxCreator<GeluGradOpx> GeluGradOpxCreator(CustomGradOperators::GeluGrad);
