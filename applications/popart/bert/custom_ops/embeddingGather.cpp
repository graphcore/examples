// Copyright 2019 Graphcore Ltd.
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

#include <popart/error.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Cast.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>

#include "embeddingGather.hpp"

// This is a simplified clone of the standard PopART Gather, but instead of using popops::multiSlice,
// it explicitly uses the popops::gather method. multiSlice uses tensor introspection for placing the compute
// so assumes that the dictionary input has been created by popops::createGatherInput. However, this op needs to support
// the tied embedding & projection so in this case the input has been layed out by popops::createMatMulInputRHS.
// popops::gather instead adds a copy before execution to ensure the compute and memory is balanced.

// WARNING: This is a specialised op and assumes the dictionary input has been provided transposed.

EmbeddingGatherOp::EmbeddingGatherOp(const popart::OperatorIdentifier &_opid,
                                     const popart::Op::Settings &settings_,
                                     const MatMulSplit split)
    : popart::Op(_opid, settings_), split(split) {}

std::unique_ptr<popart::Op> EmbeddingGatherOp::clone() const
{
    return std::make_unique<EmbeddingGatherOp>(*this);
}

std::vector<std::unique_ptr<popart::Op>> EmbeddingGatherOp::getGradOps()
{
    std::vector<std::unique_ptr<popart::Op>> result;
    result.push_back(std::make_unique<EmbeddingGatherGradOp>(*this));
    return result;
}

void EmbeddingGatherOp::setup()
{
    auto data_shape = inShape(dataInIndex());
    const auto indices_shape = inShape(indicesInIndex());

    // Use the computed shape with the data input type
    outInfo(outIndex()) =
        popart::TensorInfo(inInfo(dataInIndex()).dataType(), {indices_shape[0], data_shape[0]});
}

EmbeddingGatherGradOp::EmbeddingGatherGradOp(const EmbeddingGatherOp &op)
    : popart::Op(CustomGradOperators::EmbeddingGatherGrad, op.getSettings()) {}

std::unique_ptr<popart::Op> EmbeddingGatherGradOp::clone() const
{
    return std::make_unique<EmbeddingGatherGradOp>(*this);
}

const std::vector<popart::GradInOutMapper> &EmbeddingGatherGradOp::gradInputInfo() const
{
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {gradInIndex(), EmbeddingGatherOp::outIndex(), popart::GradOpInType::GRADOUT},
        {indicesInIndex(), EmbeddingGatherOp::indicesInIndex(), popart::GradOpInType::IN},
        {dataInIndex(), EmbeddingGatherOp::dataInIndex(), popart::GradOpInType::IN}};

    return inInfo;
}

const std::map<int, int> &EmbeddingGatherGradOp::gradOutToNonGradIn() const
{
    static const std::map<int, int> outInfo = {
        {gradOutIndex(), EmbeddingGatherOp::dataInIndex()}};

    return outInfo;
}

void EmbeddingGatherGradOp::setup() { outInfo(gradOutIndex()) = inInfo(dataInIndex()); }

static popart::OpDefinition embeddingGatherOpDef({});

static popart::OpCreator<EmbeddingGatherOp> EmbeddingGatherOpCreator(
    popart::OpDefinitions({{CustomOperators::EmbeddingGather,
      embeddingGatherOpDef}}),
    [](const popart::OperatorIdentifier &_opid,
       const popart::Op::Settings &settings,
       const popart::Attributes &attr) -> std::unique_ptr<popart::Op> {
        std::vector<int64_t> split = attr.getAttribute<popart::Attributes::Ints>("split", {2, 1});
        if (split.size() != 2)
            throw popart::error("Split must be 2 ints. {dim, factor}");
        return std::unique_ptr<popart::Op>(new EmbeddingGatherOp(_opid, settings,
                                                                {static_cast<unsigned>(split[0]), 
                                                                 static_cast<unsigned>(split[1])}));
    },
    true);

// Start OpX

EmbeddingGatherOpx::EmbeddingGatherOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex)
{
    verifyOp<EmbeddingGatherOp>(op, CustomOperators::EmbeddingGather);

    // We always want this op to layout its inputs
    inputCreatorPriority = std::numeric_limits<double>::max();
}

popart::popx::InputCreatorType EmbeddingGatherOpx::getInputCreatorType(int index0) const 
{
    return index0 == EmbeddingGatherOp::dataInIndex() ? popart::popx::InputCreatorType::CANCREATE
                                                      : popart::popx::Opx::getInputCreatorType(index0);
}

bool EmbeddingGatherOpx::createsEquiv(int, const popart::popx::Opx *, int) const { return false; }

std::vector<popart::TensorId> EmbeddingGatherOpx::mustExistBeforeCreate(int) const { return {}; }

poplar::Tensor EmbeddingGatherOpx::createInput(int index,
                                               const std::string &name) const
{
    if (index != EmbeddingGatherOp::dataInIndex())
        throw popart::error("EmbeddingGatherOpx::createInput Cannot create input {}", index);
    assert(index == 0);

    auto inputInfo = inInfo(1);
    auto weightInfo = inInfo(0);
    auto op = dynamic_cast<EmbeddingGatherOp *>(op_p);
    auto split = op->split;

    bool create_transposed = true;

    // Create W
    unsigned splitInputSize = inputInfo.dim(0);
    unsigned splitInternalSize = weightInfo.dim(0);
    unsigned splitChannels = weightInfo.dim(1);

    if (split.dim == 0)
        splitInputSize /= split.factor;
    else if (split.dim == 1)
        splitInternalSize /= split.factor;
    else
        splitChannels /= split.factor;

    std::vector<std::size_t> lhsShape = {splitInputSize, splitInternalSize};
    std::vector<std::size_t> rhsShape = {splitInternalSize, splitChannels};

    if (create_transposed)
    {
        lhsShape[1] = splitChannels;
        std::swap(rhsShape[0], rhsShape[1]);
    }
    auto type = popart::popx::popType(weightInfo);
    poplar::Tensor weight;
    if (split.factor > 1)
    {
        weight = poplin::createMatMulInputRHS(graph(),
                                              type,
                                              lhsShape,
                                              rhsShape,
                                              name + "/weights/split/0",
                                              {{"fullyConnectedPass", "TRAINING_FWD"}},
                                              &dv_p->matmulCache);
        auto weightsToClone = weight;
        if (split.factor != 1 && split.dim != 0)
        {
            for (unsigned s = 1; s != split.factor; ++s)
            {
                auto w = graph().clone(weightsToClone, name + "/weights/split/" + std::to_string(s));
                weight =
                    concat(weight, w, (split.dim - 1) ^ (create_transposed ? 1 : 0));
            }
        }
    }
    else
    {
        weight = popops::createGatherInput(graph(),
                                           type,
                                           rhsShape,
                                           static_cast<unsigned>(0),
                                           popops::GatherParams{},
                                           name);
    }

    return create_transposed ? weight.transpose() : weight;
}

void EmbeddingGatherOpx::grow(poplar::program::Sequence &prog) const
{
    const auto indicesShape = inShape(EmbeddingGatherOp::indicesInIndex());
    const auto outputShape =
        popart::vXtoY<int64_t, std::size_t>(outShape(EmbeddingGatherOp::outIndex()));

    auto indices = getInTensor(EmbeddingGatherOp::indicesInIndex());
    auto data = getInTensor(EmbeddingGatherOp::dataInIndex());

    // If there are no indices, return an empty tensor of the appropriate
    // shape
    if (indices.numElements() == 0)
    {
        auto result = graph().addVariable(
            data.elementType(), outputShape, debugPrefix("result"));

        setOutTensor(EmbeddingGatherOp::outIndex(), result);
    }
    else
    {
        auto result = popops::gather(graph(),
                                     data.transpose(),
                                     popops::cast(graph(), indices, poplar::UNSIGNED_INT, prog),
                                     0,
                                     prog,
                                     popops::GatherParams(),
                                     debugPrefix());

        setOutTensor(EmbeddingGatherOp::outIndex(), result);
    }
}

EmbeddingGatherGradOpx::EmbeddingGatherGradOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex)
{
    verifyOp<EmbeddingGatherGradOp>(op, CustomGradOperators::EmbeddingGatherGrad);
}

void EmbeddingGatherGradOpx::grow(poplar::program::Sequence &prog) const
{
    auto update = getInTensor(EmbeddingGatherGradOp::gradInIndex());
    auto indices = getInTensor(EmbeddingGatherGradOp::indicesInIndex());
    auto data = getInTensor(EmbeddingGatherGradOp::dataInIndex());

    auto result = graph().clone(data.transpose());
    popops::zero(graph(), result, prog, debugPrefix("zero"));

    if (result.numElements() == 0 || update.numElements() == 0 || indices.numElements() == 0)
    {
        setOutTensor(EmbeddingGatherGradOp::gradOutIndex(), result);
        return;
    }

    auto scale = graph().addConstant(
        update.elementType(), {}, 1.0f, debugPrefix("const_1"));
    graph().setTileMapping(scale, 0);

    update = update.expand({1});
    indices = indices.expand({1});

    // TODO: Use popops::embedding::plan

    // Accumulate the updates into the target
    popops::multiUpdateAdd(graph(),
                           result,
                           update,
                           popops::cast(graph(), indices, poplar::UNSIGNED_INT, prog),
                           scale,
                           {0},
                           {1},
                           prog,
                           popops::SlicePlan(),
                           poplar::OptionFlags(),
                           debugPrefix());

    setOutTensor(EmbeddingGatherGradOp::gradOutIndex(), result.transpose());
}

static popart::popx::OpxCreator<EmbeddingGatherOpx> embeddingGatherOpxCreator(CustomOperators::EmbeddingGather);
static popart::popx::OpxCreator<EmbeddingGatherGradOpx> embeddingGatherGradOpxCreator(CustomGradOperators::EmbeddingGatherGrad);
