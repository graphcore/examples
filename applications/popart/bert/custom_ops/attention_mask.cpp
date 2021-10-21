// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <popart/op.hpp>
#include <popart/shapeinference.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popops/Rearrange.hpp>

#include <poputil/TileMapping.hpp>

#include <random>

using namespace popart;
using namespace popart::popx;
using namespace popops::expr;
    
namespace CustomOperators {
    const popart::OperatorIdentifier AttentionMask = {"ai.graphcore", "AttentionMask", 1};
} // namespace CustomOperators

// An InplaceIdentityOp that doesn't return any grad ops. This allows you to disconnect the flow of gradients when creating the backwards pass
class AttentionMaskOp : public popart::Op {
public:

    poplar::Type dataType;

    AttentionMaskOp(const popart::OperatorIdentifier &_opid, 
                    const Op::Settings &settings_,
                    poplar::Type &dataTypeIn)
        : Op(_opid, settings_), dataType(dataTypeIn) {}

    void setup() final { 
    // input shape [B, S]
    Shape inShape = inInfo(0).shape();
    Shape refShape = inInfo(1).shape();

    // output shape [B, 1, S, S] 
    Shape outShape = {refShape.at(0), 1, refShape.at(2), refShape.at(3)};

    if (dataType == poplar::HALF)
        outInfo(0) = {"FLOAT16", outShape};
    else
        outInfo(0) = {"FLOAT", outShape};
    }

    std::unique_ptr<Op> clone() const final {
    return std::make_unique<AttentionMaskOp>(*this);
    }

    float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static popart::OpDefinition attentionMaskOpDef({});

static popart::OpCreator<AttentionMaskOp> attentionMaskOpCreator(
    popart::OpDefinitions({{CustomOperators::AttentionMask, attentionMaskOpDef}}),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
        std::string type = oci.attributes.getAttribute<Attributes::String>("dataType");
        poplar::Type dataType  = (type == "FLOAT") ? poplar::FLOAT : poplar::HALF;

        return std::unique_ptr<AttentionMaskOp>(
            new AttentionMaskOp(oci.opid, oci.settings, dataType));
    },
    true);

class AttentionMaskOpX : public popart::popx::Opx
{
public:
    AttentionMaskOpX(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex) {
    verifyOp<AttentionMaskOp>(op, CustomOperators::AttentionMask);
    }

    popart::popx::InputCreatorType getInputCreatorType(popart::InIndex) const {
    return popart::popx::InputCreatorType::CanUnwind;
    }

    poplar::Tensor unwindTensorLayout(poplar::Tensor tensor, popart::InIndex, popart::OutIndex) const {
    return tensor;
    }

    popart::view::RegMap unwindRegion(popart::InIndex, popart::OutIndex) const {
    return [this](const popart::view::Region &r) {
        return popart::view::Regions(1, r);
    };
    }

    void grow(poplar::program::Sequence &prog) const final {
    AttentionMaskOp &myOp = getOp<AttentionMaskOp>();

    poplar::Type dataType = myOp.dataType;
    poplar::Graph &graph = Opx::graph();

    // input tensor shape [B, S]
    poplar::Tensor seqIndex  = getInTensor(0);
    poplar::Tensor attentionMatrix  = getInTensor(1);
    std::size_t batchSize = attentionMatrix.dim(0);
    std::size_t seqLength = attentionMatrix.dim(3);
    seqIndex = seqIndex.reshape({batchSize, seqLength, 1});
    seqIndex = popops::cast(graph, seqIndex, dataType, prog, "input_mask_f");

    const auto dimOrdering = poputil::detectDimGroupings(graph, attentionMatrix);
    bool swapOrder = !dimOrdering.empty() && dimOrdering.front().first == 2;
    auto seqMask = swapOrder ?
            popops::sub(graph, seqIndex.dimShuffle({0, 2, 1}), seqIndex, prog, "maskVal").dimShuffle({0, 2, 1}):
            popops::sub(graph, seqIndex, seqIndex.dimShuffle({0, 2, 1}), prog, "maskVal");
    popops::absInPlace(graph, seqMask, prog);
    popops::tanhInPlace(graph, seqMask, prog);

    // Create constant tensor;
    std::mt19937 randomEngine;
    unsigned totalTile = graph.getTarget().getTilesPerIPU();
    std::uniform_int_distribution<> distrib(0, totalTile - 1);
    int tileForConst = distrib(randomEngine);
    poplar::Tensor minValue = graph.addConstant(dataType, {}, -10000.0);
    graph.setTileMapping(minValue, tileForConst);

    // Create log mask
    popops::mulInPlace(graph, seqMask, minValue, prog);
    seqMask = seqMask.reshape({batchSize, 1, seqLength, seqLength});
    setOutTensor(0, seqMask);
    }
};

static popart::popx::OpxCreator<AttentionMaskOpX>
    attentionMaskOpxCreator(CustomOperators::AttentionMask);

static popart::RegisterShapeInferenceFunction AttentionMaskShapeInfer(
    CustomOperators::AttentionMask,
            [](ShapeInferenceContext &ctx) {
            auto B = ctx.inInfo(1).shape().at(0);
            auto S = ctx.inInfo(1).shape().at(3);
            auto dtype = ctx.inInfo(1).data_type();
            ctx.outInfo(0) = {dtype, Shape({B, 1, S, S})};
    });