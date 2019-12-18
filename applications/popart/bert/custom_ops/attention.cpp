// Copyright 2019 Graphcore Ltd.
#include <cmath>
#include <memory>
#include <iostream>

#include <math.h>

#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/half.hpp>
#include <popart/ir.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/MatMul.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popnn/NonLinearity.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier Attention = {"ai.graphcore", "Attention", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
  const popart::OperatorIdentifier AttentionGrad = {"ai.graphcore", "AttentionGrad", 1};
} // namespace CustomGradOperators

class AttentionOp;
class AttentionGradOp;
class AttentionOpx;
class AttentionGradOpx;

class AttentionGradOp : public popart::Op {
public:
  unsigned heads;
  unsigned sequence_length;
  float available_memory_proportion;
  bool use_dropout;
  uint32_t dropout_modifier;
  float dropout_ratio;

  AttentionGradOp(const popart::Op &fwdOp, 
                  const unsigned heads,
                  const unsigned sequence_length,
                  const float available_memory_proportion,
                  const bool use_dropout,
                  const uint32_t dropout_modifier,
                  const float dropout_ratio)
    : popart::Op(CustomGradOperators::AttentionGrad, fwdOp.settings), 
      heads(heads),
      sequence_length(sequence_length),
      available_memory_proportion(available_memory_proportion),
      use_dropout(use_dropout),
      dropout_modifier(dropout_modifier),
      dropout_ratio(dropout_ratio) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<AttentionGradOp>(*this);
  }

  void setup() final {
    auto input = inInfo(0);
    auto outer = input.dim(0);
    auto inner = (input.dim(1) * 3);
    outInfo(0) = {input.dataType(), {outer, inner}};
  };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    if (use_dropout) {
      static const std::vector<popart::GradInOutMapper> inInfo = {
          {0, 0, popart::GradOpInType::GRADOUT},
          {1, 1, popart::GradOpInType::OUT},
          {2, 2, popart::GradOpInType::OUT},
          {3, 3, popart::GradOpInType::OUT},
          {4, 4, popart::GradOpInType::OUT},
          {5, 5, popart::GradOpInType::OUT},
          {getSeedInIndex(), 2, popart::GradOpInType::IN}};
      return inInfo;
    } else {
      static const std::vector<popart::GradInOutMapper> inInfo = {
          {0, 0, popart::GradOpInType::GRADOUT},
          {1, 1, popart::GradOpInType::OUT},
          {2, 2, popart::GradOpInType::OUT},
          {3, 3, popart::GradOpInType::OUT},
          {4, 4, popart::GradOpInType::OUT},
          {5, 5, popart::GradOpInType::OUT}};
      return inInfo;
    }
  }

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }

  // This function is only called on fwd ops
  bool requiresRandomSeed() const override { return use_dropout; }
  popart::InIndex getSeedInIndex() const override { return 6; }

  float getDropoutRatio() const { return dropout_ratio; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

class AttentionOp : public popart::Op {
public:
  unsigned heads;
  unsigned sequence_length;
  float available_memory_proportion;
  bool use_dropout;
  uint32_t dropout_modifier;
  float dropout_ratio;

  AttentionOp(const popart::OperatorIdentifier &_opid,
              const popart::Op::Settings &settings_,
              const unsigned heads,
              const unsigned sequence_length,
              const float available_memory_proportion,
              const bool use_dropout,
              const uint32_t dropout_modifier,
              const float dropout_ratio)
    : popart::Op(_opid, settings_),
      heads(heads),
      sequence_length(sequence_length),
      available_memory_proportion(available_memory_proportion),
      use_dropout(use_dropout),
      dropout_modifier(dropout_modifier),
      dropout_ratio(dropout_ratio) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<AttentionOp>(*this);
  }

  void setup() final {
    // [batch_size*sequence_length, 3*heads*head_size]
    auto input = inInfo(0);
    auto batch_size = input.dim(0) / sequence_length;
    auto head_size = ((input.nelms() / heads) / 3) / (batch_size * sequence_length);
    // Z
    outInfo(0) = {input.dataType(), {batch_size * sequence_length, heads * head_size}};
    // Q
    outInfo(1) = {input.dataType(), {batch_size * heads, sequence_length, head_size}};
    // K
    outInfo(2) = {input.dataType(), {batch_size * heads, head_size, sequence_length}};
    // V
    outInfo(3) = {input.dataType(), {batch_size * heads, sequence_length, head_size}};
    // selfAtten
    outInfo(4) = {input.dataType(), {batch_size * heads, sequence_length, sequence_length}};
    // reference for dropout
    outInfo(5) = {input.dataType(), {batch_size * heads, sequence_length, sequence_length}};
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new AttentionGradOp(*this, 
      heads, sequence_length, available_memory_proportion, 
      use_dropout, dropout_modifier, dropout_ratio));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  float getDropoutRatio() const { return dropout_ratio; }

  bool requiresRandomSeed() const override { return use_dropout; }
  popart::InIndex getSeedInIndex() const override { return 2; }
};

static popart::OpDefinition attentionOpDef({});

static popart::OpCreator<AttentionOp> attentionOpCreator(
  popart::OpDefinitions({{CustomOperators::Attention, attentionOpDef}}),
  [](const popart::OperatorIdentifier &_opid,
     const popart::Op::Settings &settings,
     const popart::Attributes &attr = {}) -> std::unique_ptr<popart::Op> {
    int64_t heads = attr.getAttribute<popart::Attributes::Int>("heads", 12);
    int64_t sequence_length = attr.getAttribute<popart::Attributes::Int>("sequence_length", 128);
    int64_t dropout_modifier = attr.getAttribute<popart::Attributes::Int>("dropout_modifier", 0);
    float dropout_ratio = attr.getAttribute<popart::Attributes::Float>("dropout_ratio", 0.15);
    float available_memory_proportion = attr.getAttribute<popart::Attributes::Float>("available_memory_proportion", -1);
    return std::unique_ptr<AttentionOp>(new AttentionOp(
      _opid, settings, 
      heads, sequence_length, available_memory_proportion,
      dropout_modifier != -1, dropout_modifier,
      dropout_ratio));
  }, true);

class AttentionOpx : public popart::popx::Opx {
public:
  AttentionOpx(popart::Op *op, popart::popx::Devicex *devicex)
    : popart::popx::Opx(op, devicex) {
      verifyOp<AttentionOp>(op, CustomOperators::Attention);
  }

  void grow(poplar::program::Sequence &prog) const final {
    poplar::Tensor input = getInTensor(0);
    poplar::Tensor mask = getInTensor(1);
    auto inputInfo = inInfo(0);
    auto op = dynamic_cast<AttentionOp*>(op_p);
    unsigned heads = op->heads;
    unsigned sequence_length = op->sequence_length;
    unsigned batch_size = inputInfo.dim(0) / sequence_length;
    unsigned head_size = ((inputInfo.nelms() / heads) / 3) / (batch_size * sequence_length);

    poplar::Tensor query, key, value;

    // input is [B * S, 3 * An * As]

    auto combinedDim = input.rank() - 1;
    auto sliceSize = head_size * heads;
    query = input.slice(0, sliceSize, combinedDim)
            .reshape({batch_size, sequence_length, heads, head_size})
            .dimShuffle({0, 2, 1, 3}) //  [B,An,S,As];
            .reshape({batch_size * heads, sequence_length, head_size});
    key = input.slice(sliceSize, 2 * sliceSize, combinedDim)
            .reshape({batch_size, sequence_length, heads, head_size})
            .dimShuffle({0, 2, 3, 1}) //  [B,An,As,S]; Key is transposed!
            .reshape({batch_size * heads, head_size, sequence_length});
    value = input.slice(2 * sliceSize, 3 * sliceSize, combinedDim)
            .reshape({batch_size, sequence_length, heads, head_size})
            .dimShuffle({0, 2, 1, 3}) //  [B,An,S,As];
            .reshape({batch_size * heads, sequence_length, head_size});

    setOutTensor(1, query);
    setOutTensor(2, key);
    setOutTensor(3, value);

    poplar::OptionFlags mmOpts;
    if (op->available_memory_proportion > 0) {
      mmOpts.set("availableMemoryProportion", std::to_string(op->available_memory_proportion));
    }

    auto selfAtten = poplin::matMulGrouped(
      graph(), query, key, prog, input.elementType(), debugPrefix("QK"),
      mmOpts, &dv_p->matmulCache); // [B,An,S,S]

    // Scale the Self Attention Matrix by the head size : Makes Large Difference in SoftMax
    float fscale = 1 / sqrt(float(head_size));
    auto pScale = graph().addConstant(selfAtten.elementType(), {1}, fscale);
    poputil::mapTensorLinearly(graph(), pScale);
    // Use inplace multiplication as it is more efficient
    popops::mulInPlace(graph(), selfAtten, pScale, prog, debugPrefix("Scale"));

    popops::addInPlace(graph(), selfAtten, mask, prog, debugPrefix("ApplyMask"));

    popnn::nonLinearityInPlace(
      graph(), popnn::NonLinearityType::SOFTMAX_STABLE,
      selfAtten,
      prog, debugPrefix());

    poplar::Tensor reference = graph().clone(selfAtten);
    if (op->use_dropout && op->getIr().canTrain()) {
      double dropoutProbability = 1. - static_cast<double>(op->getDropoutRatio());
      double scale = 1. / (1. - static_cast<double>(op->getDropoutRatio()));

      auto dropout = poprand::dropout(graph(),
                                      &getInTensor(op->getSeedInIndex()),
                                      op->dropout_modifier,
                                      selfAtten,
                                      reference,
                                      dropoutProbability,
                                      scale,
                                      prog,
                                      debugPrefix("dropout"));
    }

    selfAtten = selfAtten.reshape({batch_size * heads, sequence_length, sequence_length}); // [B*An,S,S]
    setOutTensor(4, selfAtten);
    setOutTensor(5, reference);

    auto z = poplin::matMulGrouped(
      graph(), selfAtten, value, prog, input.elementType(), debugPrefix("Z"),
      mmOpts, &dv_p->matmulCache); // [B*An,S,As]

    z = z
      .reshape({batch_size, heads, sequence_length, head_size}) // [B,An,S,As]
      .dimShuffle({0,2,1,3}) // [An,S,B,As]
      .reshape({batch_size * sequence_length, heads * head_size});

    setOutTensor(0, z);
  }
};

class AttentionGradOpx : public popart::popx::Opx {
public:
  AttentionGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<AttentionGradOp>(op, CustomGradOperators::AttentionGrad);
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op = dynamic_cast<AttentionGradOp *>(op_p);
  
    poplar::Tensor zGrad   = getInTensor(0),
                   query   = getInTensor(1),
                   key     = getInTensor(2),
                   value   = getInTensor(3),
                   softmax = getInTensor(4),
                   ref     = getInTensor(5);

    auto outputInfo = outInfo(0);
    unsigned heads = op->heads;
    unsigned sequence_length = op->sequence_length;
    unsigned batch_size = outputInfo.dim(0) / sequence_length;
    unsigned head_size = ((outputInfo.nelms() / heads) / 3) / (batch_size * sequence_length);

    zGrad = zGrad
            .reshape({batch_size, sequence_length, heads, head_size}) // [B,S,An,As]
            .dimShufflePartial({1, 2}, {2, 1})  // [B,An,S,As]
            .reshape({batch_size * heads, sequence_length, head_size});
    value = value.dimShufflePartial({1, 2}, {2, 1});

    poplar::OptionFlags mmOpts;
    if (op->available_memory_proportion > 0) {
      mmOpts.set("availableMemoryProportion", std::to_string(op->available_memory_proportion));
    }

    auto softmaxGrad = poplin::matMulGrouped(
        graph(), zGrad, value, prog, zGrad.elementType(), debugPrefix("ZGrad"),
        mmOpts, &dv_p->matmulCache);
    softmax = softmax.reshape(softmaxGrad.shape());

    if (op->use_dropout && op->getIr().canTrain()) {
      double dropoutProbability = 1. - static_cast<double>(op->getDropoutRatio());
      double scale = 1. / (1. - static_cast<double>(op->getDropoutRatio()));
      auto dropout = poprand::dropout(graph(),
                                      &getInTensor(op->getSeedInIndex()),
                                      op->dropout_modifier,
                                      softmax,
                                      ref,
                                      dropoutProbability,
                                      scale,
                                      prog,
                                      debugPrefix("dropout"));
    }

    auto selfAttenGrad = popnn::nonLinearityInputGradient(
      graph(), popnn::NonLinearityType::SOFTMAX_STABLE, 
      softmax, softmaxGrad, 
      prog, debugPrefix());

    double scale = 1 / sqrt(double(head_size));
    float fscale = (float)scale;
    auto pScale = graph().addConstant(selfAttenGrad.elementType(), {1}, fscale);
    poputil::mapTensorLinearly(graph(), pScale);
    // Use inplace multiplication as it is more efficient
    popops::mulInPlace(graph(), selfAttenGrad, pScale, prog, debugPrefix("Scale"));

    // now group tensors
    auto groupedLhs = poplar::concat({selfAttenGrad,
                                      selfAttenGrad.dimShufflePartial({1, 2}, {2, 1}),
                                      softmax.dimShufflePartial({1, 2}, {2, 1})});

    auto groupedRhs = poplar::concat({key.dimShufflePartial({1, 2}, {2, 1}), 
                                      query, 
                                      zGrad});

    // use explicit copy to reduce rearrange cost
    auto groupedLhsRearranged =
        poplin::createMatMulGroupedInputLHS(graph(),
                                            groupedLhs.elementType(),
                                            groupedLhs.elementType(),
                                            groupedLhs.shape(),
                                            groupedRhs.shape(),
                                            debugPrefix("concat/lhsplaceholder"),
                                            {},
                                            &dv_p->matmulCache);

    auto groupedRhsRearranged =
        poplin::createMatMulGroupedInputRHS(graph(),
                                            groupedLhs.elementType(),
                                            groupedLhs.elementType(),
                                            groupedLhs.shape(),
                                            groupedRhs.shape(),
                                            debugPrefix("concat/rhsplaceholder"),
                                            {},
                                            &dv_p->matmulCache);

    prog.add(poplar::program::Copy(groupedLhs, groupedLhsRearranged));
    prog.add(poplar::program::Copy(groupedRhs, groupedRhsRearranged));

    auto groupedGrad = poplin::matMulGrouped(
      graph(), groupedLhsRearranged, groupedRhsRearranged, 
      prog, groupedLhsRearranged.elementType(), debugPrefix("QKVGrad"),
        mmOpts, &dv_p->matmulCache);
    
    unsigned numGroups = heads * batch_size;
    auto queryGrad = groupedGrad.slice(0, numGroups)
                .reshape({batch_size, heads, sequence_length, head_size}) // [B,An,S,As]
                .dimShufflePartial({1, 2}, {2, 1}) // [An,B,S,As]
                .reshape({batch_size * sequence_length, heads * head_size}); // [1, An,B*S,As]
    auto keyGrad = groupedGrad.slice(numGroups, 2 * numGroups)
                .reshape({batch_size, heads, sequence_length, head_size}) // [B,An,S,As]
                .dimShufflePartial({1, 2}, {2, 1}) // [An,B,S,As]
                .reshape({batch_size * sequence_length, heads * head_size}); // [1, An,B*S,As]
    auto valueGrad = groupedGrad.slice(2 * numGroups, 3 * numGroups)
                .reshape({batch_size, heads, sequence_length, head_size}) // [B,An,S,As]
                .dimShufflePartial({1, 2}, {2, 1}) // [An,B,S,As]
                .reshape({batch_size * sequence_length, heads * head_size}); // [1, An,B*S,As]

    auto outGrad = poplar::concat({queryGrad, keyGrad, valueGrad}, 1);

    setOutTensor(0, outGrad);
  }
};

static popart::popx::OpxCreator<AttentionOpx> attentionOpxCreator(CustomOperators::Attention);
static popart::popx::OpxCreator<AttentionGradOpx> attentionGradOpxCreator(CustomGradOperators::AttentionGrad);
