// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// This implements a custom popart operator for RNNTLoss
//

#include <iostream>
#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include "rnnt_utils.hpp"
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>

// The first thing to do is to provide an identifier that PopART can use later
// to address the operator.
namespace Onnx {
namespace CustomOperators {
const popart::OperatorIdentifier RNNTLoss = {"com.acme", "RNNTLoss", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier RNNTLossGrad = {"com.acme", "RNNTLossGrad", 1};
} // namespace CustomGradOperators
} // namespace Onnx

// For training with a custom Op, four classes need to be implemented,
// one for each of:
// {forward, gradient} x {Op, Opx}.
//
// If only inference is required, then two classes need to be implemented:
// {forward} x {Op, Opx}.
//
// The Op is a poplar/hardware agnostic description of the computation.
// the Opx is the poplar implementation of the Op.
//
// We do training in this example, so the four classes implemented are:
//
class RNNTLossOp;
class RNNTLossGradOp;
class RNNTLossOpx;
class RNNTLossGradOpx;

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

// The gradient Op
class RNNTLossGradOp : public popart::Op {
public:
  RNNTLossGradOp(const popart::Op &fwdOp)
      : popart::Op(Onnx::CustomGradOperators::RNNTLossGrad,
                   fwdOp.getSettings()) {}

  std::unique_ptr<Op> clone() const final {
    return make_unique<RNNTLossGradOp>(*this);
  }

  // The output popart Tensor has the same inputInfo and numerical type
  // (i.e. the same TensorInfo) as the input Tensor. This function is
  // required for inputInfo/type inference
  //
  virtual void setup() { outInfo(0) = inInfo(1); }

  // function describing the inputs and output(s) of RNNTLossGradOp
  // The Gradient Op which we are implementing (RNNTLossGradOp) has 2 inputs.
  // The input at index 0 is:
  // the gradient of the 0'th output Tensor of the RNNTLossOp.
  // The input at index 1 is :
  // the 0'th output Tensor of the RNNTLossOp.
  // Supposing the RNNTLossOp has input Tensor T0 and output Tensor T1,
  //
  //   input at index 0 (T0)
  //          |
  //        RNNTLossOp
  //          |
  //   output at index 0 (T1)
  //
  // Then the picture described by the map below looks like,
  //
  //
  //    input at index 0 (gradient of T1)
  //         |   input at index 1 (T1)
  //         |     |
  //         |     |
  //        RNNTLossGradOp
  //            |
  //            |
  //   output at index 0 (gradient of T0)
  //
  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {0, 0,
         popart::GradOpInType::GradOut},    // Gradient of the input to the op
        {1, 0, popart::GradOpInType::In},   // compactedlogProbs
        {2, 1, popart::GradOpInType::In},   // inputLengths
        {3, 2, popart::GradOpInType::In},   // labelLengths
        {4, 1, popart::GradOpInType::Out},  // non-reduced loss
        {5, 2, popart::GradOpInType::Out},  // logAlpha
        {6, 3, popart::GradOpInType::Out}}; // logBeta
    return inInfo;
  }

  // The Grad Op only has one output, at index 0. The output at index 0
  // is the gradient of the input at index 0 of the RNNTLossOp
  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

// The forward Op
class RNNTLossOp : public popart::Op {
public:
  RNNTLossOp(const popart::OperatorIdentifier &_opid,
             const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  // same comment as for RNNTLossGradOp, for running shape/type inference
  // "statically"
  void setup() final {
    auto logProbInfo = inInfo(0);
    auto batchSize = logProbInfo.dim(0);
    auto T = logProbInfo.dim(1);
    auto U = logProbInfo.dim(2);
    auto inputLengthsInfo = inInfo(1);
    assert(inputLengthsInfo.dim(0) == batchSize);
    auto labelLengthsInfo = inInfo(2);
    assert(labelLengthsInfo.dim(0) == batchSize);
    // first output is the reduced loss
    const popart::ReductionType reduction = popart::ReductionType::Mean;
    const auto outType = popart::DataType::FLOAT; // correct ?
    switch (reduction) {
    case popart::ReductionType::Sum:
    case popart::ReductionType::Mean:
      outInfo(0).set(outType, {});
      break;

    case popart::ReductionType::NoReduction:
      outInfo(0).set(outType, {batchSize});
      break;
    }
    // second output is the full loss
    outInfo(1).set(outType, {batchSize});
    // third is alpha
    outInfo(2).set(popart::DataType::FLOAT, {batchSize, T, U});
    // fourth is beta
    outInfo(3).set(popart::DataType::FLOAT, {batchSize, T, U});
  }

  std::unique_ptr<Op> clone() const final {
    return make_unique<RNNTLossOp>(*this);
  }

  // There is only one Gradient Op for RNNTLossOp, a RNNTLossGradOp
  // It is possible to have multiple Gradient Ops
  // (Conv has 2 in popart, one for weights and one for activations)
  //
  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new RNNTLossGradOp(*this));
    return upops;
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

// describe the inputs and outputs that are supported by the operation
static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};
static popart::OpDefinition::DataTypes T1 = {popart::DataType::FLOAT};

static popart::OpDefinition::DataTypes T2 = {popart::DataType::INT32};

static popart::OpDefinition
    RNNTLossOpDef({popart::OpDefinition::Inputs({{"probs", T},
                                                 {"input_lengths", T2},
                                                 {"label_lengths", T2}}),
                   popart::OpDefinition::Outputs({{"loss", T1},
                                                  {"full_loss", T1},
                                                  {"log_alpha", T1},
                                                  {"log_beta", T1}}),
                   popart::OpDefinition::Attributes({})});

static popart::OpCreator<RNNTLossOp>
    RNNTLossOpCreator({{Onnx::CustomOperators::RNNTLoss, RNNTLossOpDef}});

using namespace popops::expr;

// forward Opx (poplar implementation of the forward Op)
class RNNTLossOpx : public popart::popx::Opx {
public:
  RNNTLossOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    // not strictly necessary, we check that op is castable to a RNNTLossOp *.
    verifyOp<RNNTLossOp>(op, Onnx::CustomOperators::RNNTLoss);
    graph().addCodelets("custom_ops/rnnt_loss/codelet.cpp"); // add codelets to the graph
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto compactedLogProbs = getInTensor(0);
    auto batchSize = compactedLogProbs.dim(0);
    auto T = compactedLogProbs.dim(1);
    auto U = compactedLogProbs.dim(2);
    auto input_lengths = getInTensor(1);
    auto label_lengths = getInTensor(2);
    poplar::Tensor compactedFinal =
        graph().addVariable(compactedLogProbs.elementType(),
                            {batchSize, U, T * 2}, "compactedLogProbs");
    poputil::mapTensorLinearly(graph(), compactedFinal, 1, T * 2);
    compactedFinal =
        compactedFinal.reshape({batchSize, U, T, 2}).dimShuffle({0, 2, 1, 3});
    prog.add(poplar::program::Copy(compactedLogProbs, compactedFinal));
    prog.add(poplar::program::WriteUndef(compactedLogProbs));
    auto logAlpha =
        graph().addVariable(poplar::FLOAT, {batchSize, T, U}, "alpha");
    graph().setTileMapping(
        logAlpha,
        graph().getTileMapping(compactedFinal.dimShuffle({3, 0, 1, 2})[0]));
    auto logBeta =
        graph().addVariable(poplar::FLOAT, {batchSize, T, U}, "beta");
    graph().setTileMapping(logBeta, graph().getTileMapping(logAlpha));
    alpha_beta(graph(), prog, compactedFinal, input_lengths, label_lengths,
               logAlpha, logBeta);
    auto loss = losses(graph(), prog, logBeta);

    // For now we support only Mean Reduction
    const popart::ReductionType reduction =
        popart::ReductionType::Mean;
    setOutTensor(1, loss);
    popops::ReduceParams params(popops::Operation::ADD);
    poplar::Tensor tmp;
    switch (reduction) {
    case popart::ReductionType::Sum:
      tmp = reduce(graph(), loss, {0}, params, prog,
                   debugContext("LossReduction"));

      setOutTensor(0, tmp);
      break;
    case popart::ReductionType::Mean: {
      poplar::Tensor scale = graph().addConstant(
          poplar::FLOAT, {},
          poplar::ArrayRef<float>({1.0F / static_cast<float>(batchSize)}),
          "scale");
      popops::ReduceParams scaleParams(popops::Operation::ADD, false, scale);
      graph().setTileMapping(scale, 0);

      tmp = reduce(graph(), loss, {0}, scaleParams, prog,
                   debugContext("lossReduction"));
      setOutTensor(0, tmp);
    } break;
    case popart::ReductionType::NoReduction:
      setOutTensor(0, loss);
      break;
    }

    setOutTensor(2, logAlpha);
    setOutTensor(3, logBeta);
  }
};

// backward Opx (poplar implementation of the backward Op)
class RNNTLossGradOpx : public popart::popx::Opx {
public:
  RNNTLossGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<RNNTLossGradOp>(op, Onnx::CustomGradOperators::RNNTLossGrad);
  }

  // Create the gradient poplar::Tensor, which is
  // 3 * input_to_RNNTLoss**2 * gradient_of_RNNTLoss_output
  void grow(poplar::program::Sequence &prog) const final {
    poplar::Tensor gradOut = getInTensor(0);
    poplar::Tensor compactedLogProbs = getInTensor(1);
    poplar::Tensor inputLengths = getInTensor(2);
    poplar::Tensor labelLengths = getInTensor(3);
    poplar::Tensor loss = getInTensor(4);
    poplar::Tensor logAlpha = getInTensor(5);
    poplar::Tensor logBeta = getInTensor(6);
    const size_t batchSize = compactedLogProbs.dim(0);
    const size_t T = compactedLogProbs.dim(1);
    const size_t U = compactedLogProbs.dim(2);
    poplar::Tensor compactedGrads = graph().addVariable(
        poplar::FLOAT, {batchSize, T, U, 2}, "compactedGradients");
    poputil::mapTensorLinearly(graph(), compactedGrads, 1, 2);
    grads(graph(), prog, compactedLogProbs, inputLengths, labelLengths,
          logAlpha, logBeta, loss, compactedGrads);

    popops::mulInPlace(graph(), compactedGrads, gradOut, prog);
    if (compactedGrads.elementType() != compactedLogProbs.elementType()) {
      setOutTensor(0, popops::cast(graph(), compactedGrads,
                                   compactedLogProbs.elementType(), prog));
    } else {
      setOutTensor(0, compactedGrads);
    }
  }
};

static popart::popx::OpxCreator<RNNTLossOpx>
    RNNTLossOpxCreator(Onnx::CustomOperators::RNNTLoss);
static popart::popx::OpxCreator<RNNTLossGradOpx>
    RNNTLossGradOpxCreator(Onnx::CustomGradOperators::RNNTLossGrad);
