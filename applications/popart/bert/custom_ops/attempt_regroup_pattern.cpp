// Copyright 2019 Graphcore Ltd.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/topocons.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/sgd1accumulate.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <queue>

#include "attempt_regroup.cpp"
#include "detach.cpp"
#include "sparse_sgd1_accumulate.cpp"
#include "utils.cpp"

// This pattern inserts attemptRegroupOp before: 
// 1) SGD1AccumulateOp on the gradient input if:
//   - that gradient comes from a matmul
//   - The accumulator is also consumed by SparseSGD1AccumulateOp
//
// 2) SGD1UpdateOp on the accumulator input if:
//   - that accumulator comes from a matmul
//   - the weight is not also consumed by GatherOp
//
// In both cases there is a scaledAddTo where the tile mapping 
// of the operands does not match. Adding attemptRegroup should remove some
// slow prearranging.
// In the case of (1) the accumulator is laid out as a copy of the weight
// so accumulator + gradient will cause prearrange.
// In the case of (2) the accumulator is laid out as a copy of the gradient
// so weight + accumulator will cause prearrange.

static bool look_for_matmul(popart::Op *op) {
    // Already handled this case
    if (op->isConvertibleTo<AttemptRegroupOp>()) {
        return false;
    }
    // Gradient comes from a matmul
    if (op->isConvertibleTo<popart::MatMulBaseOp>()) {
        return true;
    }
    popart::Tensor *input;
    if (op->isConvertibleTo<popart::SGD1AccumulateOp>()) {
        input = op->input->tensor(popart::SGD1AccumulateOp::getUpdaterInIndex());
    } else if (op->isConvertibleTo<popart::SGD1VarUpdateOp>()) {
        input = op->input->tensor(popart::SGD1VarUpdateOp::getUpdaterInIndex());
    } else if (op->input->n() != 1) {
        popart::logging::pattern::debug("Not looking for matmuls beyond {} as it has >1 inputs", op->debugName());
        return false;
    } else {
        // TODO: Have whitelist of traversable ops.
        input = op->input->tensors().front();
    }
    // Unlikely to happen as inputs without a producer are Streams or Variables.
    if (!input->hasProducer()) {
        return false;
    }
    return look_for_matmul(input->getProducer());
}

class AttemptRegroupPattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {
        if (op->isConvertibleTo<popart::SGD1AccumulateOp>()) {
            return weight_consumed_by<SparseSGD1AccumulateOp>(op->input->tensor(popart::SGD1AccumulateOp::getVarToUpdateInIndex())) && 
                   look_for_matmul(op);
        }
        if (op->isConvertibleTo<popart::SGD1VarUpdateOp>()) {
            return !weight_consumed_by<popart::GatherOp>(op->input->tensor(popart::SGD1VarUpdateOp::getVarToUpdateInIndex())) &&
                   look_for_matmul(op);
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto &ir = op->getIr();
        auto &graph = op->getGraph();

        popart::InIndex updater_index;
        popart::InIndex ref_index;
        if (op->isConvertibleTo<popart::SGD1AccumulateOp>()) {
            updater_index = popart::SGD1AccumulateOp::getUpdaterInIndex();
            ref_index     = popart::SGD1AccumulateOp::getVarToUpdateInIndex();
        } else {
            updater_index = popart::SGD1VarUpdateOp::getUpdaterInIndex();
            ref_index     = popart::SGD1VarUpdateOp::getVarToUpdateInIndex();
        }

        auto attempt_regroup_up = std::make_unique<AttemptRegroupOp>(
            popart::Op::Settings(graph, op->name() + "/attemptRegroup"));
        auto attempt_regroup = attempt_regroup_up.get();
        transferBaseProperties(op, attempt_regroup);
        graph.moveIntoGraph(std::move(attempt_regroup_up));

        auto grad = op->input->tensor(updater_index);
        op->disconnectInTensor(grad);
        attempt_regroup->connectInTensor(AttemptRegroupOp::getInTensorIndex(), grad->id);
        auto ref = op->input->tensor(ref_index);
        attempt_regroup->connectInTensor(AttemptRegroupOp::getRefTensorIndex(), ref->id);

        auto grad_regrouped = grad->id + "_regrouped";
        attempt_regroup->createAndConnectOutTensor(AttemptRegroupOp::getRegroupedOutIndex(), grad_regrouped);
        op->connectInTensor(updater_index, grad_regrouped);

        attempt_regroup->setup();

        return true;
    }
};

static popart::PatternCreator<AttemptRegroupPattern> attemptRegroupPatternCreator("AttemptRegroupPattern", true);
