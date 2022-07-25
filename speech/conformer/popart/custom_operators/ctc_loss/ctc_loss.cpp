// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

//
// This custom operator implements the CTC loss
//

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <cstdlib>

#include "ctcloss_utils.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/ProfileValue.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>

#include <popart/op.hpp>
#include <popart/op/loss.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popnn/Loss.hpp>

#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/Scatter.hpp>
#include <popops/Zero.hpp>

namespace CustomOperators {
const popart::OperatorIdentifier CtcLoss = {"com.acme", "CtcLoss", 1};
} // namespace CustomOperators

namespace CustomGradOperators {
const popart::OperatorIdentifier CtcLossGrad = {"com.acme", "CtcLossGrad", 1};
} // namespace CustomGradOperators

class CtcLossOp;
class CtcLossGradOp;
class CtcLossOpx;
class CtcLossGradOpx;

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace popops::expr;

// The gradient Op
class CtcLossGradOp : public popart::Op {
public:
  CtcLossGradOp(const CtcLossOp &fwdOp);

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<CtcLossGradOp>(*this);
  }

  virtual void setup() { outInfo(0) = inInfo(4); }

  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {0, 0, popart::GradOpInType::GradOut}, // Gradient of the input to the
                                               // CTC Loss
        {1, 0, popart::GradOpInType::Out}, // reduced negative log-likelihood
        {2, 1, popart::GradOpInType::Out}, // log-alpha+beta
        {3, 2, popart::GradOpInType::Out}, // sequence
        {4, 0, popart::GradOpInType::In},  // probabilities
        {5, 1, popart::GradOpInType::In},  // targets
        {6, 2, popart::GradOpInType::In},  // input lengths
        {7, 3, popart::GradOpInType::In},  // target lengths
        {8, 3, popart::GradOpInType::Out}, // non-reduced nll loss
    };
    return inInfo;
  }

  // The Grad Op has one output at index 0. The output at index 0 is the
  // gradient of the loss at nongradient index 0.
  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  unsigned int getBlank() const { return blank; }
  popart::ReductionType getReductionType() const { return reduction; }
  bool getPartial32() const { return partial32; }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("partial32", getPartial32());
    os.appendAttribute("blank", getBlank());
    os.appendAttribute("reduction", static_cast<int>(getReductionType()));
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("partial32", getPartial32());
    os.appendAttribute("blank", getBlank());
    os.appendAttribute("reduction", static_cast<int>(getReductionType()));
  }

private:
  unsigned int blank;
  bool partial32;
  popart::ReductionType reduction;
};

namespace {} // namespace

class CtcLossOp : public popart::LossOp {

public:
  CtcLossOp(const popart::OperatorIdentifier &_opid, const bool partial32_,
            const unsigned int blank_, const popart::ReductionType reduction_,
            const popart::Op::Settings &settings_)
      : popart::LossOp(_opid, settings_, reduction_), partial32{partial32_}, blank(blank_),
        reduction(reduction_) {}

  // Configure the output popart Tensor
  void setup() final {
    auto probInfo = inInfo(0);
    auto targetInfo = inInfo(1);
    auto inputLengths = inInfo(2);
    auto targetLengths = inInfo(3);

    auto batchSize = probInfo.dim(0);
    auto maxInputLength = probInfo.dim(1);
    // auto numLabels = probInfo.dim(3);
    auto maxTargetLength = targetInfo.dim(1);
    auto sequenceLength = 2 * maxTargetLength + 1;

    assert(batchSize == targetInfo.dim(0));
    assert(batchSize == inputLengths.dim(0));
    assert(batchSize == targetLengths.dim(0));
    auto outType = partial32 ? popart::DataType::FLOAT : probInfo.dataType();

    switch (reduction) {
    case popart::ReductionType::Sum:
    case popart::ReductionType::Mean:
      outInfo(0).set(outType, {});
      break;

    case popart::ReductionType::NoReduction:
      outInfo(0).set(outType, {batchSize});
      break;
    }
	const auto realInputLength =
        (maxInputLength % 2) == 0 ? maxInputLength : maxInputLength + 1;
    outInfo(1).set(outType, {realInputLength, sequenceLength, batchSize});
    outInfo(2).set(targetInfo.dataType(), {batchSize, sequenceLength});
    outInfo(3).set(outType, {batchSize});
  }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("partial32", getPartial32());
    os.appendAttribute("blank", getBlank());
    os.appendAttribute("reduction", static_cast<int>(getReductionType()));
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("partial32", getPartial32());
    os.appendAttribute("blank", getBlank());
    os.appendAttribute("reduction", static_cast<int>(getReductionType()));
  }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<CtcLossOp>(*this);
  }

  // CtcLossOp has only one GradOp.
  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new CtcLossGradOp(*this));
    return upops;
  }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  unsigned int getBlank() const { return blank; }
  bool getPartial32() const { return partial32; }
  popart::ReductionType getReductionType() const { return reduction; }

private:
  bool partial32;
  unsigned int blank;
  popart::ReductionType reduction;
};

namespace {

// static popart::OpDefinition ctcLossOpDef({});
static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};
static popart::OpDefinition::DataTypes T2 = {popart::DataType::UINT32};
static popart::OpDefinition::DataTypes T3 = {popart::DataType::BOOL};

static popart::OpDefinition
    ctcLossOpDef({popart::OpDefinition::Inputs({{"input", T},
                                                {"targets", T2},
                                                {"input_length", T2},
                                                {"target_length", T2}}),
                  popart::OpDefinition::Outputs({{"loss", T},
                                                 {"alpha", T},
                                                 {"sequence", T2},
                                                 {"reverse", T3},
                                                 {"logBeta", T}}),
                  popart::OpDefinition::Attributes({})});

static popart::OpCreator<CtcLossOp> ctcLossOpCreator(
    popart::OpDefinitions({{CustomOperators::CtcLoss, ctcLossOpDef}}),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
      // default blank is 0
      unsigned int blank = static_cast<unsigned int>(
          oci.attributes.getAttribute<popart::Attributes::Int>("blank", 0));
      // by default, mean reduction.
      popart::ReductionType reduction = static_cast<popart::ReductionType>(
          oci.attributes.getAttribute<popart::Attributes::Int>("reduction", 1));
      bool partial32 = static_cast<bool>(
          oci.attributes.getAttribute<popart::Attributes::Int>("partial32", 0));
      auto ctcLoss =
          new CtcLossOp(oci.opid, partial32, blank, reduction, oci.settings);

      return std::unique_ptr<popart::Op>(ctcLoss);
    },
    true);
} // namespace

// The forward Opx (poplar implementation of the forward Op)

class CtcLossOpx : public popart::popx::Opx {
  unsigned numTiles_;
  unsigned numWorkers_;

public:
  CtcLossOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    // Not strictly necessary, we check that op is castable to a CtcLossOp *.
    verifyOp<CtcLossOp>(op, CustomOperators::CtcLoss);
    const auto &target = graph().getTarget();
    numTiles_ = target.getTilesPerIPU();
    numWorkers_ = target.getNumWorkerContexts();
    graph().addCodelets(std::string(std::getenv("CTC_LOSS_CODELET_PATH")).c_str()); // add codelets to the graph
  }

  void grow(Sequence &prog) const final {
    auto op = getOp<CtcLossOp>();

    auto probs = getInTensor(0);
    auto probInfo = inInfo(0);

    auto targets = getInTensor(1);
    auto inputLengths = getInTensor(2);
    auto targetLengths = getInTensor(3);

    const std::size_t batchSize = probs.dim(0);
    // const std::size_t maxInputLength = probs.dim(1);
    // const std::size_t numLabels = probs.dim(3);
    const std::size_t maxTargetLength = targets.dim(1);
    const std::size_t sequenceLength = 2 * maxTargetLength + 1;

    // unsigned int BLANK = op.getBlank();
    popart::ReductionType reduction = op.getReductionType();
    bool partial32 = op.getPartial32();

    assert(batchSize == targets.dim(0));
    poplar::Tensor lpp = probs.dimShuffle({1, 2, 0});
    const size_t inputLength = lpp.dim(0);
    Tensor sequence, sequenceBs, diff, reverse, logAlpha, loss, batchIndices,
        sliceIndices;
    std::tie(sequence, sequenceBs, diff, reverse, logAlpha, loss, batchIndices,
             sliceIndices) =
        ctcLossPrepare(graph(), targets, targetLengths, probs.elementType(),
                       prog, partial32, inputLength);

    loss = computeAlphaBeta(graph(), lpp, sequence, diff, reverse, inputLengths,
                            targetLengths, batchIndices, sliceIndices, logAlpha,
                            loss, prog);
    Tensor negLogLikelihood = nllLoss(graph(), loss, prog, batchSize);
    setOutTensor(3, cloneNcopy(prog, negLogLikelihood));

    auto outType = partial32 ? popart::DataType::FLOAT : probInfo.dataType();
    auto poplarOutType = partial32 ? poplar::FLOAT : lpp.elementType();

    ReduceParams params(Operation::ADD);
    Tensor tmp;
    switch (reduction) {
    case popart::ReductionType::Sum:
      tmp = reduce(graph(), negLogLikelihood, {0}, params, prog,
                   debugContext("CtcLoss"));

      setOutTensor(0, popops::cast(graph(), tmp, poplarOutType, prog));
      break;
    case popart::ReductionType::Mean: {
      Tensor scale = graph().addConstant(
          poplar::FLOAT, {},
          ArrayRef<float>({1.0F / static_cast<float>(batchSize)}), "scale");
      ReduceParams scaleParams(Operation::ADD, false, scale);
      graph().setTileMapping(scale, 0);
      Tensor lengths = popops::cast(graph(), targetLengths,
                                    negLogLikelihood.elementType(), prog);

      popops::mapInPlace(graph(), _1 / _2, {negLogLikelihood, lengths}, prog);

      tmp = reduce(graph(), negLogLikelihood, {0}, scaleParams, prog,
                   debugContext("CtcLoss"));
      setOutTensor(0, popops::cast(graph(), tmp, poplarOutType, prog));
    } break;
    case popart::ReductionType::NoReduction:
      setOutTensor(
          0, popops::cast(graph(), negLogLikelihood, poplarOutType, prog));
      break;
    }

    setOutTensor(1, logAlpha);
    setOutTensor(2, sequenceBs);
  }
};

class CtcLossGradOpx : public popart::popx::Opx {
  unsigned numTiles_;
  unsigned numWorkers_;

public:
  CtcLossGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<CtcLossGradOp>(op, CustomGradOperators::CtcLossGrad);
    const auto &target = graph().getTarget();
    numTiles_ = target.getTilesPerIPU();
    numWorkers_ = target.getNumWorkerContexts();
  }

  // Create the gradient Tensor
  void grow(Sequence &prog) const final {

    const CtcLossGradOp &op = getOp<CtcLossGradOp>();
    auto reductionType = op.getReductionType();

    Tensor gradOut = getInTensor(0);
    Tensor reducedNll = getInTensor(1);
    Tensor logAlpha = getInTensor(2);
    Tensor sequence = getInTensor(3);
    Tensor probs = getInTensor(4);
    Tensor targets = getInTensor(5);
    Tensor inputLengths = getInTensor(6);
    Tensor targetLengths = getInTensor(7);
    Tensor negLogLikelihood = getInTensor(8);

    // const std::size_t maxInputLength = logAlpha.dim(0);
    const std::size_t sequenceLength = logAlpha.dim(1);
    const std::size_t batchSize = logAlpha.dim(2);
    // const std::size_t numLabels = probs.dim(3);

    // const unsigned BLANK = 0;
    poplar::Tensor lpp = probs.dimShuffle({1, 2, 0});

    Tensor grad =
        computeGrad(graph(), logAlpha, sequence, inputLengths, targetLengths,
                    negLogLikelihood, lpp, gradOut, prog);
    // To match PyTorch implementation, we now need reduce the grads by the
    // target length and batch size
    if (reductionType == popart::ReductionType::Mean) {
      auto len = popops::cast(graph(), targetLengths, grad.elementType(), prog);

      poplar::Tensor bsTensor =
          graph().addConstant(grad.elementType(), {}, batchSize);
      graph().setTileMapping(bsTensor, 0);
      popops::mulInPlace(graph(), len, bsTensor, prog);
      popops::divInPlace(graph(), grad, len.reshape({1, 1, len.shape()[0]}),
                         prog, debugContext("rescaleLossTgtlenafter"));
    }

    auto output = grad.dimShuffle({2, 0, 1});

    setOutTensor(0, popops::cast(graph(), output,
                                 probs.elementType(), prog));
  }
};

CtcLossGradOp::CtcLossGradOp(const CtcLossOp &fwdOp)
    : popart::Op(CustomGradOperators::CtcLossGrad, fwdOp.getSettings()),
      blank(fwdOp.getBlank()), reduction(fwdOp.getReductionType()) {}

static popart::popx::OpxCreator<CtcLossOpx>
    ctcLossOpxCreator(CustomOperators::CtcLoss);

static popart::popx::OpxCreator<CtcLossGradOpx>
    ctcLossGradOpxCreator(CustomGradOperators::CtcLossGrad);

int main(int argc, char **argv) { return 0; }

// -------------- cppimport --------------
// clang-format off
/*
<%
cfg['sources'] = ['ctcloss_utils.cpp']
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wsign-compare', '-shared']
cfg['libraries'] = ['popart', 'poplar', 'popops', 'poputil', 'popnn']
setup_pybind11(cfg)
%>
*/
