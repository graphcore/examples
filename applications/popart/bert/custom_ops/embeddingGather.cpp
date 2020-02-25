// Copyright 2019 Graphcore Ltd.
#include <algorithm>
#include <string>
#include <vector>
#include <memory>

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/ir.hpp>
#include <popart/graph.hpp>

#include <popart/error.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>
#include <popart/logging.hpp>

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
                                     const MatMulSplit split,
                                     const std::string embedding_tensor_id)
    : popart::Op(_opid, settings_), split(split), embedding_tensor_id(embedding_tensor_id) {}

std::unique_ptr<popart::Op> EmbeddingGatherOp::clone() const
{
    return std::make_unique<EmbeddingGatherOp>(*this);
}

std::vector<std::unique_ptr<popart::Op>> EmbeddingGatherOp::getGradOps()
{
    std::vector<std::unique_ptr<popart::Op>> result;
    result.push_back(std::make_unique<EmbeddingGatherGradOp>(*this, split, embedding_tensor_id));
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

EmbeddingGatherGradOp::EmbeddingGatherGradOp(const EmbeddingGatherOp &op,
                                             const MatMulSplit split,
                                             const std::string embedding_tensor_id)
    : popart::Op(CustomGradOperators::EmbeddingGatherGrad, op.getSettings()),
      split(split), embedding_tensor_id(embedding_tensor_id)
{
    // This prevents the op from being removed when using splitUpdate. Normally it would be as:
    // it's output is not consumed and it is not a subclass of VarUpdate.
    pruneable = split.factor <= 1;
}

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

void EmbeddingGatherGradOp::setup()
{
    outInfo(gradOutIndex()) = inInfo(dataInIndex());
}

static popart::OpDefinition embeddingGatherOpDef({});

static popart::OpCreator<EmbeddingGatherOp> EmbeddingGatherOpCreator(
    popart::OpDefinitions({{CustomOperators::EmbeddingGather,
                            embeddingGatherOpDef}}),
    [](const popart::OperatorIdentifier &_opid,
       const popart::Op::Settings &settings,
       const popart::Attributes &attr) -> std::unique_ptr<popart::Op> {
        std::vector<int64_t> split = attr.getAttribute<popart::Attributes::Ints>("split", {2, 1});
        std::string embedding_tensor_id = attr.getAttribute<popart::Attributes::String>("embedding_tensor_id", "");
        if (split.size() != 2)
            throw popart::error("Split must be 2 ints. {dim, factor}");
        return std::unique_ptr<popart::Op>(new EmbeddingGatherOp(_opid, settings,
                                                                 {static_cast<unsigned>(split[0]),
                                                                  static_cast<unsigned>(split[1])},
                                                                 embedding_tensor_id));
    },
    true);

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

    if (create_transposed) {
        lhsShape[1] = splitChannels;
        std::swap(rhsShape[0], rhsShape[1]);
    }
    auto type = popart::popx::popType(weightInfo);
    poplar::Tensor weight;
    if (split.factor > 1) {
        // TODO: Consider making this createGatherInput
        weight = poplin::createMatMulInputRHS(graph(),
                                              type,
                                              lhsShape,
                                              rhsShape,
                                              name + "/weights/split/0",
                                              {},
                                              &dv_p->matmulCache);
        auto weightsToClone = weight;
        if (split.factor != 1 && split.dim != 0) {
            for (unsigned s = 1; s != split.factor; ++s) {
                auto w = graph().clone(weightsToClone, name + "/weights/split/" + std::to_string(s));
                weight =
                    concat(weight, w, (split.dim - 1) ^ (create_transposed ? 1 : 0));
            }
        }
    } else {
        weight = popops::createGatherInput(graph(),
                                           type,
                                           rhsShape,
                                           static_cast<unsigned>(0),
                                           popops::GatherParams{},
                                           name);
    }

    return create_transposed ? weight.transpose() : weight;
}

poplar::Tensor EmbeddingGatherOpx::serialisedGather(const poplar::Tensor &data, const poplar::Tensor &indices, poplar::program::Sequence &prog) const {
    auto op = dynamic_cast<EmbeddingGatherOp *>(op_p);
    auto split = op->split;
    auto dType = data.elementType();
    auto vocabLength = data.shape()[0];

    if (vocabLength % split.factor != 0) {
        throw popart::error("Split Factor {} must be a multiple of vocab size {}.", split.factor, vocabLength); 
    }

    poplar::Tensor result = popops::createSliceableTensor(
        graph(), dType, {indices.shape()[0], data.shape()[1]}, {0}, {1}, 0, op->outId(EmbeddingGatherOp::outIndex()));
    popops::zero(graph(), result, prog, debugPrefix("Zero"));

    unsigned splitSize = vocabLength / split.factor;
    for (int i = 0; i < split.factor; ++i) {
        auto indicesSplit = popops::sub(graph(), indices, i * splitSize, prog, debugPrefix("indicesSplit"));
        auto mask = popops::lt(graph(), indicesSplit, splitSize, prog, debugPrefix("mask<size"));
        // If unsigned the subtraction will underflow values so checking >= 0 is not needed
        if (indicesSplit.elementType() != poplar::UNSIGNED_INT) {
            auto mask_gteq = popops::gteq(graph(), indicesSplit, 0U, prog, debugPrefix("mask_>=0"));
            popops::logicalAndInPlace(graph(), mask, mask_gteq, prog, debugPrefix("mask_AND"));
        }

        auto indicesMask = popops::cast(graph(), mask, indicesSplit.elementType(), prog, debugPrefix("mask_castInt"));
        popops::mulInPlace(graph(), indicesSplit, indicesMask, prog, debugPrefix("masked_indices"));

        auto resultSplit = popops::gather(graph(),
                                          data.slice(i * splitSize, (i + 1) * splitSize, 0),
                                          indicesSplit,
                                          0,
                                          prog,
                                          popops::GatherParams(),
                                          debugPrefix());

        auto gradMask = popops::cast(graph(), mask, dType, prog, debugPrefix("mask_castHalf"));
        popops::mulInPlace(graph(), resultSplit, gradMask.expand({1}), prog, debugPrefix("masked_result"));
        // Accumulate into result
        popops::addInPlace(graph(), result, resultSplit, prog);
    }

    return result;

}

void EmbeddingGatherOpx::grow(poplar::program::Sequence &prog) const
{
    bool serialiseGather = dynamic_cast<EmbeddingGatherOp *>(op_p)->split.factor > 1;

    const auto indicesShape = inShape(EmbeddingGatherOp::indicesInIndex());
    const auto outputShape =
        popart::vXtoY<int64_t, std::size_t>(outShape(EmbeddingGatherOp::outIndex()));

    auto indices = getInTensor(EmbeddingGatherOp::indicesInIndex());
    auto data = getInTensor(EmbeddingGatherOp::dataInIndex());

    // If there are no indices, return an empty tensor of the appropriate
    // shape
    if (indices.numElements() == 0) {
        auto result = graph().addVariable(
            data.elementType(), outputShape, debugPrefix("result"));
        setOutTensor(EmbeddingGatherOp::outIndex(), result);
    } else {
        poplar::Tensor result;
        if (serialiseGather) {
            result = serialisedGather(data.transpose(),
                                      popops::cast(graph(), indices, poplar::UNSIGNED_INT, prog),
                                      prog);
        } else {
            result = popops::gather(graph(),
                                    data.transpose(),
                                    popops::cast(graph(), indices, poplar::UNSIGNED_INT, prog),
                                    0,
                                    prog,
                                    popops::GatherParams(),
                                    debugPrefix());
        }
        setOutTensor(EmbeddingGatherOp::outIndex(), result);
    }
}

EmbeddingGatherGradOpx::EmbeddingGatherGradOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex)
{
    verifyOp<EmbeddingGatherGradOp>(op, CustomGradOperators::EmbeddingGatherGrad);
    inputCreatorPriority = std::numeric_limits<double>::max();
}

popart::popx::InputCreatorType EmbeddingGatherGradOpx::getInputCreatorType(int index0) const
{
    auto op = dynamic_cast<EmbeddingGatherGradOp *>(op_p);
    auto initialId = op->acclSliceInputFirstIndex();
    auto numSplits = op->split.factor;

    if (index0 < initialId || index0 >= initialId + numSplits)
    {
        return popart::popx::Opx::getInputCreatorType(index0);
    }

    return popart::popx::InputCreatorType::CANCREATE;
}

std::vector<popart::TensorId> EmbeddingGatherGradOpx::mustExistBeforeCreate(int) const { return {}; }

poplar::Tensor EmbeddingGatherGradOpx::createInput(int index, const std::string &name) const
{
    auto op = dynamic_cast<EmbeddingGatherGradOp *>(op_p);
    auto initialId = op->acclSliceInputFirstIndex();
    auto split = op->split;
    auto numSplits = split.factor;

    bool createTransposed = true;

    if (index < initialId || index >= initialId + numSplits) {
        throw popart::error("EmbeddingGatherGradOpx::createInput Cannot create input {}", index);
    }

    popart::logging::info("Creating accumulator tensor: {} [{}]", name, index);

    auto indicesInfo = inInfo(EmbeddingGatherGradOp::indicesInIndex());
    auto weightInfo = inInfo(index);
    auto type = popart::popx::popType(weightInfo);
    unsigned splitInternalSize = weightInfo.dim(0);
    unsigned splitChannels = weightInfo.dim(1);

    std::vector<std::size_t> lhsShape = indicesInfo.shape_szt();
    std::vector<std::size_t> rhsShape = {splitInternalSize, splitChannels};

    if (createTransposed) {
        lhsShape[1] = splitChannels;
        std::swap(rhsShape[0], rhsShape[1]);
    }

    auto weight = poplin::createMatMulInputRHS(graph(),
                                               type,
                                               lhsShape,
                                               rhsShape,
                                               name,
                                               {},
                                               &dv_p->matmulCache);
    return createTransposed ? weight.transpose() : weight;
}

void EmbeddingGatherGradOpx::tiedGradUpdate(poplar::program::Sequence &prog,
                                            const poplar::Tensor &update,
                                            const poplar::Tensor &indices,
                                            const poplar::Tensor &scale) const
{
    auto op = dynamic_cast<EmbeddingGatherGradOp *>(op_p);
    auto initialId = EmbeddingGatherGradOp::acclSliceInputFirstIndex();

    auto numSplits = op->split.factor;
    unsigned splitSize = getInTensor(initialId).shape()[1];

    std::vector<poplar::Tensor> slices;
    slices.reserve(numSplits);

    for (unsigned i = 0; i < numSplits; ++i)
    {
        auto acclInputIndex = initialId + i;
        auto acclTensor = getInTensor(acclInputIndex);
        slices.push_back(acclTensor);

        auto indicesSplit = popops::sub(graph(), indices, i * splitSize, prog, debugPrefix("indicesSplit"));

        // For this slice, perform an add using the maxed indices
        popops::multiUpdateAdd(graph(),
                               acclTensor.transpose(),
                               update,
                               indicesSplit,
                               scale,
                               {0},
                               {1},
                               prog,
                               popops::SlicePlan(),
                               poplar::OptionFlags(),
                               debugPrefix());
    }

    auto concatSlices = poplar::concat(slices, 1);
    setOutTensor(EmbeddingGatherGradOp::gradOutIndex(), concatSlices);
}

void EmbeddingGatherGradOpx::untiedGradUpdate(poplar::program::Sequence &prog,
                                              const poplar::Tensor &update,
                                              const poplar::Tensor &indices,
                                              const poplar::Tensor &scale) const
{
    auto data = getInTensor(EmbeddingGatherGradOp::dataInIndex());
    auto result = graph().clone(data.transpose());
    popops::zero(graph(), result, prog, debugPrefix("zero"));

    if (result.numElements() == 0 || update.numElements() == 0 || indices.numElements() == 0)
    {
        setOutTensor(EmbeddingGatherGradOp::gradOutIndex(), result);
        return;
    }

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

void EmbeddingGatherGradOpx::grow(poplar::program::Sequence &prog) const
{
    // If using tied weights, we want the dampening scale factor from the optimiser
    auto op = dynamic_cast<EmbeddingGatherGradOp *>(op_p);
    bool serialiseUpdate = op->split.factor > 1;

    auto update = getInTensor(EmbeddingGatherGradOp::gradInIndex());
    auto indices = getInTensor(EmbeddingGatherGradOp::indicesInIndex());
    update = update.expand({1});
    indices = indices.expand({1});

    auto dampeningScale = serialiseUpdate ? getScaledDampeningScalar(op->embedding_tensor_id) : 1.0f;
    auto scale = graph().addConstant(update.elementType(), {}, dampeningScale, debugPrefix("EmbeddingGather/scale"));
    graph().setTileMapping(scale, 0);

    if (serialiseUpdate) {
        if(!op->input->hasIndex(EmbeddingGatherGradOp::acclSliceInputFirstIndex())) {
            throw popart::error("Accumulator slices haven't been mapped to inputs. Has EmbeddingGatherPattern run?");
        }
        tiedGradUpdate(prog,
                       update,
                       popops::cast(graph(), indices, poplar::UNSIGNED_INT, prog),
                       scale);
    } else {
        untiedGradUpdate(prog,
                         update,
                         popops::cast(graph(), indices, poplar::UNSIGNED_INT, prog),
                         scale);
    }
}

// When const, dampening is currently stored as a scalar in the optimizer - there's no
// readily accessible tensor. This is being considered and may change in the future, but
// for now we will grab the optimizer and read it back.
float EmbeddingGatherGradOpx::getScaledDampeningScalar(const popart::TensorId tensorId, float defaultValue) const
{
    auto sgd = dynamic_cast<const popart::SGD *>(&(dv_p->ir().getOptimizer()));

    auto dampeningScalar = defaultValue;
    if (sgd != 0) {
        dampeningScalar = 1 - sgd->dampenings().get(tensorId).val();
        dampeningScalar /= sgd->lossScaling().val();
    }
    return dampeningScalar;
}


static popart::popx::OpxCreator<EmbeddingGatherOpx> embeddingGatherOpxCreator(CustomOperators::EmbeddingGather);
static popart::popx::OpxCreator<EmbeddingGatherGradOpx> embeddingGatherGradOpxCreator(CustomGradOperators::EmbeddingGatherGrad);
