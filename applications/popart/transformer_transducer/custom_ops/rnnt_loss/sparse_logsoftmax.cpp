// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>
#include <random>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>

#include "logsoftmax.hpp"
#include "rnnt_utils.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popnn/Loss.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

namespace Onnx {
namespace CustomOperators {
const popart::OperatorIdentifier SparseLogSoftmax = {"com.acme",
                                                     "SparseLogSoftmax", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier SparseLogSoftmaxGrad = {
    "com.acme", "SparseLogSoftmaxGrad", 1};
} // namespace CustomGradOperators
} // namespace Onnx

class SparseLogSoftmaxOp;
class SparseLogSoftmaxGradOp;
class SparseLogSoftmaxOpx;
class SparseLogSoftmaxGradOpx;

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

// The gradient Op
class SparseLogSoftmaxGradOp : public popart::Op {
public:
  SparseLogSoftmaxGradOp(const popart::Op &fwdOp)
      : popart::Op(Onnx::CustomGradOperators::SparseLogSoftmaxGrad,
                   fwdOp.getSettings()) {}

  std::unique_ptr<Op> clone() const final {
    return make_unique<SparseLogSoftmaxGradOp>(*this);
  }

  // for running shape/type inference "statically"
  virtual void setup() {
    // auto compactedGradients = getInTensor(0);
    // auto logProbs = getInTensor(1);
    // auto input_lengths = getInTensor(2);
    // auto labels = getInTensor(3);
    // auto label_lengths = getInTensor(4);
    outInfo(0) = inInfo(1);
  }

  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {0, 0,
         popart::GradOpInType::GradOut},   // Gradient of the input to the op
        {1, 0, popart::GradOpInType::In},  // logProbs
        {2, 1, popart::GradOpInType::In},  // labels
        {3, 2, popart::GradOpInType::In}}; // labelLengths
    return inInfo;
  }

  // The Grad Op only has one output, at index 0. The output at index 0
  // is the gradient of the input at index 0 of the CubeOp
  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class SparseLogSoftmaxOp : public popart::Op {
public:
  SparseLogSoftmaxOp(const popart::OperatorIdentifier &_opid,
                     const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  // Configure the output popart Tensor
  void setup() final {
    auto logProbInfo = inInfo(0);
    auto batchSize = logProbInfo.dim(0);
    auto T = logProbInfo.dim(1);
    auto U = logProbInfo.dim(2);
    auto labelsInfo = inInfo(1);
    assert(labelsInfo.dim(0) == batchSize);
    assert(labelsInfo.dim(1) == U - 1);
    auto labelLengthsInfo = inInfo(2);
    assert(labelLengthsInfo.dim(0) == batchSize);
    outInfo(0).set(logProbInfo.dataType(), {batchSize, T, U, 2});
  }

  std::unique_ptr<Op> clone() const final {
    return make_unique<SparseLogSoftmaxOp>(*this);
  }
  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new SparseLogSoftmaxGradOp(*this));
    return upops;
  }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

// describe the inputs and outputs that are supported by the operation
static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};

static popart::OpDefinition::DataTypes T2 = {popart::DataType::INT32};

static popart::OpDefinition sparseLogSoftmaxOpDef(
    {popart::OpDefinition::Inputs(
         {{"probs", T}, {"labels", T2}, {"label_lengths", T2}}),
     popart::OpDefinition::Outputs({{"output", T}}),
     popart::OpDefinition::Attributes({})});

static popart::OpCreator<SparseLogSoftmaxOp> sparseLogSoftmaxOpCreator(
    {{Onnx::CustomOperators::SparseLogSoftmax, sparseLogSoftmaxOpDef}});

// The forward Opx (poplar implementation of the forward Op)
class SparseLogSoftmaxOpx : public popart::popx::Opx {

public:
  SparseLogSoftmaxOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    // Not strictly necessary, we check that op is castable to a CopmactProbsOp
    // *.
    verifyOp<SparseLogSoftmaxOp>(op, Onnx::CustomOperators::SparseLogSoftmax);
    graph().addCodelets("custom_ops/rnnt_loss/codelet.cpp"); // add codelets to the graph
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto logProbs = getInTensor(0);
    auto alphabet = logProbs.dim(3);
    auto labels = getInTensor(1);
    auto label_lengths = getInTensor(2);
    poputil::mapTensorLinearly(graph(), logProbs, 1, alphabet);
    auto compacted = logSoftmaxRnnt(graph(), logProbs, labels, label_lengths,
                                    prog, "SparseLogSoftmaxFwd");

    setOutTensor(0, compacted);
  }
};

class SparseLogSoftmaxGradOpx : public popart::popx::Opx {
public:
  SparseLogSoftmaxGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<SparseLogSoftmaxGradOp>(
        op, Onnx::CustomGradOperators::SparseLogSoftmaxGrad);
  }
  void grow(poplar::program::Sequence &prog) const final {
    auto compactedGradients = getInTensor(0);
    auto logProbs = getInTensor(1);
    size_t alphabet = logProbs.dim(3);
    auto labels = getInTensor(2);
    auto label_lengths = getInTensor(3);
    poputil::mapTensorLinearly(graph(), logProbs, 1, alphabet);
    poputil::mapTensorLinearly(graph(), compactedGradients, 1, 2);
    poplar::Tensor gradients =
        logSoftmaxRnntGrad(graph(), logProbs, compactedGradients, labels,
                           label_lengths, prog, "SparseLogSoftmaxGrad");
    setOutTensor(0, gradients);
  }
};

static popart::popx::OpxCreator<SparseLogSoftmaxOpx>
    sparseLogSoftmaxOpxCreator(Onnx::CustomOperators::SparseLogSoftmax);
static popart::popx::OpxCreator<SparseLogSoftmaxGradOpx>
    sparseLogSoftmaxGradOpxCreator(
        Onnx::CustomGradOperators::SparseLogSoftmaxGrad);
