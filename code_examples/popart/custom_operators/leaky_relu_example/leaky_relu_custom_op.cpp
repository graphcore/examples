// Copyright 2020 Graphcore Ltd.
//
// This example demonstrates how to create a custom operator for PopART, in this
// case a Leaky ReLU op that returns `x` for any element `x >= 0` and `x *
// alpha` for any element `x < 0`, where `alpha` is provided as a scalar
// attribute to the operator.
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace CustomOperators {
const popart::OperatorIdentifier LeakyRelu_1 = {"ai.onnx", "LeakyRelu", 1};
const popart::OperatorIdentifier LeakyRelu_6 = {"ai.onnx", "LeakyRelu", 6};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier LeakyReluGrad_1 = {"ai.onnx", "LeakyReluGrad",
                                                    1};
const popart::OperatorIdentifier LeakyReluGrad_6 = {"ai.onnx", "LeakyReluGrad",
                                                    6};
} // namespace CustomGradOperators

class LeakyReluOp;
class LeakyReluOpx;
class LeakyReluGradOpx;

class LeakyReluGradOp : public popart::Op {
public:
  LeakyReluGradOp(const LeakyReluOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<LeakyReluGradOp>(*this);
  }
  void setup() final { outInfo(0) = inInfo(0); };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const;

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const;

  bool requiresRandomSeed() const override { return false; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  float getAlpha() const { return alpha; }

  // Implementation defined below
  void appendAttributes(popart::OpSerialiserBase &os) const override;

  // Implementation defined below
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;

private:
  float alpha;
};

class LeakyReluOp : public popart::Op {
public:
  LeakyReluOp(const popart::OperatorIdentifier &_opid, float _alpha,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), alpha(_alpha) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<LeakyReluOp>(*this);
  }

  void setup() final { outInfo(0) = inInfo(0); }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new LeakyReluGradOp(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }

  // Attributes
  float getAlpha() const { return alpha; }

private:
  float alpha;
};

namespace {

static popart::OpDefinition leakyReluOpDef({});

static popart::OpCreator<LeakyReluOp> leakyReluOpCreator(
    popart::OpDefinitions({{CustomOperators::LeakyRelu_1, leakyReluOpDef},
                           {CustomOperators::LeakyRelu_6, leakyReluOpDef}}),
    [](const popart::OperatorIdentifier &_opid,
       const popart::Op::Settings &settings,
       const popart::Attributes &attr) -> std::unique_ptr<popart::Op> {
      // default epsilon is 10**(-2)
      float alpha =
          attr.getAttribute<popart::Attributes::Float>("alpha", 1e-2f);
      auto leakyRelu = new LeakyReluOp(_opid, alpha, settings);

      return std::unique_ptr<popart::Op>(leakyRelu);
    },
    true);
} // namespace

namespace pe = popops::expr;

class LeakyReluOpx : public popart::popx::Opx {
public:
  LeakyReluOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluOp>(
        op, {CustomOperators::LeakyRelu_1, CustomOperators::LeakyRelu_6});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<LeakyReluOp>();

    poplar::Tensor input = getInTensor(0);

    float alpha = op.getAlpha();

    // x < 0.0f ? alpha * x : x
    auto expression = pe::Select(pe::Mul(pe::Const(alpha), pe::_1), pe::_1,
                                 pe::Lt(pe::_1, pe::Const(0.0f)));

    popops::mapInPlace(graph(), expression, {input}, prog,
                       debugPrefix("LeakyRelu"), poplar::OptionFlags());

    setOutTensor(0, input);
  }
};

class LeakyReluGradOpx : public popart::popx::Opx {
public:
  LeakyReluGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluGradOp>(op, {CustomGradOperators::LeakyReluGrad_1,
                                   CustomGradOperators::LeakyReluGrad_6});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<LeakyReluGradOp>();

    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    float alpha = op.getAlpha();

    // (grad * (x < 0.0f ? alpha : 1))
    pe::Mul expression = pe::Mul(pe::Select(pe::Const(alpha), pe::Const(1.0f),
                                            pe::Lt(pe::_2, pe::Const(0.0f))),
                                 pe::_1);

    auto output =
        popops::map(graph(), expression, {grad, input}, prog,
                    debugPrefix("LeakyReluGrad"), poplar::OptionFlags());

    setOutTensor(0, output);
  }
};

LeakyReluGradOp::LeakyReluGradOp(const LeakyReluOp &fwdOp)
    : popart::Op(CustomGradOperators::LeakyReluGrad_6, fwdOp.settings),
      alpha(fwdOp.getAlpha()) {}

const std::vector<popart::GradInOutMapper> &
LeakyReluGradOp::gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut}, {1, 0, popart::GradOpInType::In}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &LeakyReluGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

void LeakyReluGradOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

void LeakyReluGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("alpha", getAlpha());
}

static popart::popx::OpxCreator<LeakyReluOpx> LeakyReluOpxCreator(
    {CustomOperators::LeakyRelu_1, CustomOperators::LeakyRelu_6});
static popart::popx::OpxCreator<LeakyReluGradOpx>
    LeakyReluGradOpxCreator({CustomGradOperators::LeakyReluGrad_1,
                             CustomGradOperators::LeakyReluGrad_6});
