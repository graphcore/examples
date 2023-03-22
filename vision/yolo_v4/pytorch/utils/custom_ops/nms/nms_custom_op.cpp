// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <popops/Cast.hpp>

#include "nms.hpp"

// The first thing to do is to provide an identifier that PopART can use later
// to address the operator.
namespace Onnx {
namespace CustomOperators {
const popart::OperatorIdentifier Nms = {"ai.graphcore", "Nms", 1};
} // namespace CustomOperators
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
class NmsOp;
class NmsOpx;

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

// The forward Op
class NmsOp : public popart::Op {
public:
  NmsOp(const popart::OperatorIdentifier &_opid, const float threshold,
        const float scoreThreshold, const uint32_t numDetections, float sigma,
        bool useGather, const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), threshold_{threshold},
        scoreThreshold_{scoreThreshold},
        numDetections_{numDetections}, sigma_{sigma}, useGather_{useGather} {}

  // same comment as for NmsGradOp, for running shape/type inference
  // "statically"
  virtual void setup() {
    auto scoresInfo = inInfo(0);
    auto boxesInfo = inInfo(1);
    assert(scoresInfo.rank() == 3);
    assert(boxesInfo.rank() == 3);
    const uint32_t batchSize = scoresInfo.dim(0);
    const uint32_t N = scoresInfo.dim(1);

    outInfo(0).set(popart::DataType::INT32, {batchSize, numDetections_});
    outInfo(1).set(scoresInfo.dataType(), {batchSize, numDetections_});
    outInfo(2).set(boxesInfo.dataType(), {batchSize, numDetections_, 4});
    outInfo(3).set(popart::DataType::INT32, {batchSize, numDetections_});
    outInfo(4).set(popart::DataType::INT32, {batchSize});
  }

  std::unique_ptr<Op> clone() const final { return make_unique<NmsOp>(*this); }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("threshold", getThreshold());
    os.appendAttribute("scoreThreshold", getScoreThreshold());
    os.appendAttribute("sigma", getSigma());
    os.appendAttribute("numDetections", getNumDetections());
    os.appendAttribute("useGather", getUseGather());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("threshold", getThreshold());
    os.appendAttribute("scoreThreshold", getScoreThreshold());
    os.appendAttribute("sigma", getSigma());
    os.appendAttribute("numDetections", getNumDetections());
    os.appendAttribute("useGather", getUseGather());
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  float getThreshold() const { return threshold_; }
  float getScoreThreshold() const { return scoreThreshold_; }
  float getSigma() const { return sigma_; }
  uint32_t getNumDetections() const { return numDetections_; }
  bool getUseGather() const { return useGather_; }

private:
  float threshold_;
  float scoreThreshold_;
  uint32_t numDetections_;
  float sigma_;
  bool useGather_;
};

// describe the inputs and outputs that are supported by the operation
static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};
static popart::OpDefinition::DataTypes T2 = {popart::DataType::INT32,
                                             popart::DataType::UINT32};
static popart::OpDefinition::DataTypes T3 = {popart::DataType::INT32};

static popart::OpDefinition nmsOpDef(
    {popart::OpDefinition::Inputs({{"scores", T}, {"boxes", T}}),
     popart::OpDefinition::Outputs(
         {{"indices", T3}, {"scores", T}, {"boxes", T}, {"classes", T2}}),
     popart::OpDefinition::Attributes({{"threshold", {"*"}},
                                       {"scoreThreshold", {"*"}},
                                       {"numDetections", {"*"}},
                                       {"useGather", {"*"}}})});

static popart::OpCreator<NmsOp> nmsOpCreator(
    popart::OpDefinitions({{Onnx::CustomOperators::Nms, nmsOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      float threshold = info.attributes.getAttribute<popart::Attributes::Float>(
          "threshold", 0.5f);
      float sigma = info.attributes.getAttribute<popart::Attributes::Float>(
          "sigma", 0.0f);
      float scoreThreshold =
          info.attributes.getAttribute<popart::Attributes::Float>(
              "scoreThreshold", -std::numeric_limits<float>::max());
      uint32_t numDetections =
          info.attributes.getAttribute<popart::Attributes::Int>("numDetections",
                                                                100);
      int useGather =
          info.attributes.getAttribute<popart::Attributes::Int>("useGather", 0);
      return std::make_unique<NmsOp>(info.opid, threshold, scoreThreshold,
                                     numDetections, sigma, useGather != 0,
                                     info.settings);
    },
    true);

// forward Opx (poplar implementation of the forward Op)
class NmsOpx : public popart::popx::Opx {
public:
  NmsOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    // not strictly necessary, we check that op is castable to a NmsOp *.
    verifyOp<NmsOp>(op, Onnx::CustomOperators::Nms);
    std::string file_path("/utils/custom_ops/nms/codelet.cpp");
    std::string detection_path(std::getenv("PYTORCH_APPS_DETECTION_PATH"));
    graph().addCodelets(detection_path +
                        file_path); // add codelets to the graph
  }

  void grow(poplar::program::Sequence &prog) const final {
    // Nms the input. We create a poplar::Tensor of name outId(0)
    auto op = getOp<NmsOp>();
    float threshold = op.getThreshold();
    float scoreThreshold = op.getScoreThreshold();
    float sigma = op.getSigma();
    bool useGather = op.getUseGather();
    uint32_t numDetections = op.getNumDetections();
    const auto &scores = getInTensor(0);
    const auto &boxes = getInTensor(1);
    poplar::Tensor indicesAns, scoresAns, boxesAns, classesAns, lengthsAns;
    std::tie(indicesAns, scoresAns, boxesAns, classesAns, lengthsAns) =
        nmsMulti(graph(), prog, scores, boxes, threshold, numDetections,
                 scoreThreshold, sigma, useGather);
    setOutTensor(0, indicesAns);
    setOutTensor(1, scoresAns);
    setOutTensor(2, boxesAns);
    setOutTensor(3, classesAns);
    setOutTensor(4, lengthsAns);
  }
};

static popart::popx::OpxCreator<NmsOpx>
    nmsOpxCreator(Onnx::CustomOperators::Nms);
