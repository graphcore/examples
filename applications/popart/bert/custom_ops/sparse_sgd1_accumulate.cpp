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
#include <popart/op/sgd1accumulate.hpp>
#include <popart/util.hpp>
#include <popart/logging.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Cast.hpp>
#include <poputil/TileMapping.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier SparseSGD1Accumulate = {"ai.graphcore", "SparseSGD1Accumulate", 1};
} // namespace CustomOperators

class SparseSGD1AccumulateOp;
class SparseSGD1AccumulateOpx;

// Cannot subclass popart::SGD1AccumulateOp as the constructor hard codes the OperatorIdentifier
class SparseSGD1AccumulateOp : public popart::VarUpdateWithUpdaterOp {
public:
    unsigned axis;
    const popart::OptimizerValue initDpsf1;

    SparseSGD1AccumulateOp(const popart::TensorId &varToUpdate,
                           popart::OptimizerValue dpsf1,
                           const unsigned axis,
                           const Op::Settings &opSettings)
        : popart::VarUpdateWithUpdaterOp(CustomOperators::SparseSGD1Accumulate,
                                         varToUpdate,
                                         opSettings),
          initDpsf1(dpsf1),
          axis(axis) {}

    static popart::InIndex getIndicesInIndex() { return 3; }
    static popart::InIndex getDpsf1InIndex() { return 2; }
    float getSubgraphValue() const final { return getLowSubgraphValue(); }

    std::unique_ptr<popart::Op>
    cloneWithNewName(const popart::TensorId &x) const {
        return std::make_unique<SparseSGD1AccumulateOp>(x, initDpsf1, axis, settings);
    }

    std::unique_ptr<popart::Op> clone() const {
        return std::make_unique<SparseSGD1AccumulateOp>(*this);
    }

    std::map<popart::InIndex, popart::TensorId> optimizerInputs() const {
        throw popart::error("Sparse SGD1 optimizer inputs not implemented");
    }

    void appendOutlineAttributes(popart::OpSerialiserBase &os) const {
        popart::Op::appendOutlineAttributes(os);

        if (initDpsf1.isConst()) {
            os.appendAttribute("const dampening scale factor", initDpsf1.val());
        }
    }
};

class SparseSGD1AccumulateOpx : public popart::popx::Opx {
public:
    unsigned axis;
    SparseSGD1AccumulateOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<SparseSGD1AccumulateOp>(op, CustomOperators::SparseSGD1Accumulate);
        inputCreatorPriority = std::numeric_limits<double>::max();

        auto _op = getOp<SparseSGD1AccumulateOp>();
        axis = _op.axis;
    }

    popart::popx::InputCreatorType getInputCreatorType(int index0) const {
        return index0 == SparseSGD1AccumulateOp::getVarToUpdateInIndex() ? 
            popart::popx::InputCreatorType::CANCREATE : popart::popx::Opx::getInputCreatorType(index0);
    }

    std::vector<popart::TensorId> mustExistBeforeCreate(int) const { return {}; }

    poplar::Tensor createInput(int index, const std::string &name) const {
        if (index != SparseSGD1AccumulateOp::getVarToUpdateInIndex()) {
            throw popart::error("SparseSGD1AccumulateOpx::createInput Cannot create input {}", index);
        }

        auto info = inInfo(SparseSGD1AccumulateOp::getVarToUpdateInIndex());
        const auto shape = info.shape_szt();

        // Perhaps should be a clone of the original weight tensor
        return popops::createGatherInput(graph(),
                                         popart::popx::popType(info),
                                         shape,
                                         static_cast<unsigned>(axis),
                                         popops::GatherParams{},
                                         name);
    }

    void grow(poplar::program::Sequence &prog) const
    {
        // If using tied weights, we want the dampening scale factor from the optimiser
        auto op = getOp<SparseSGD1AccumulateOp>();

        auto isConst = op.initDpsf1.isConst();

        auto accl = getInTensor(SparseSGD1AccumulateOp::getVarToUpdateInIndex());
        auto grad = getInTensor(SparseSGD1AccumulateOp::getUpdaterInIndex());
        auto indices = getInTensor(SparseSGD1AccumulateOp::getIndicesInIndex());
        auto dpsf = isConst ? 
            getConst(accl.elementType(), {}, op.initDpsf1.val(), "ConstSparseDPSF") :
            getInTensor(SparseSGD1AccumulateOp::getDpsf1InIndex());

        if (isConst && op.initDpsf1.val() == 0.0f) {
            throw popart::internal_error(
                "dpsf1 of 0 is not allowed, should have been caught in "
                "the Ir, dpsf1 of 0 could be caused by dampening of 1, which "
                "means the gradient is multiplied by 0 (no learning)");
        }

        grad = grad.expand({1 - axis});
        indices = indices.expand({1 - axis});

        // Accumulate the updates into the target
        popops::multiUpdateAdd(graph(),
                               accl,
                               grad,
                               popops::cast(graph(), indices, poplar::UNSIGNED_INT, prog),
                               dpsf,
                               {axis},
                               {1},
                               prog,
                               popops::SlicePlan(),
                               poplar::OptionFlags(),
                               debugPrefix("nonConstSparseSGD1Accl"));

        // reference accl returned
        setOutTensor(SparseSGD1AccumulateOp::getUpdatedVarOutIndex(), accl);
    }
};

static popart::popx::OpxCreator<SparseSGD1AccumulateOpx> SparseSGD1AccumulateOpxCreator(CustomOperators::SparseSGD1Accumulate);