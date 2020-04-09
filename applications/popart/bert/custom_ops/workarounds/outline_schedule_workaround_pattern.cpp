// Copyright 2019 Graphcore Ltd.
#include <limits>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/topocons.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/varupdate.hpp>

// This pattern is a workaround for the scheduler preference for optimising liveness over outlining.
// By moving all varupdate & ipucopy ops towards the end of the schedule they will not break up the sequence of the
// bwd pass, allowing for consecutive layers to be optimally outlined.
// For example:
//   A_Grad, VarUpdate, B_Grad, VarUpdate, A_Grad, VarUpdate, B_Grad, VarUpdate => [Outline(A_Grad), Outline(B_Grad)]
//   A_Grad, B_Grad, A_Grad, B_Grad, VarUpdate, VarUpdate, VarUpdate, VarUpdate => [Outline(A_Grad, B_Grad)]
//
// TODO Use topocons/binning to enforce optimal schedule:
//  A_Grad, B_Grad, VarUpdate, VarUpdate, A_Grad, B_Grad, VarUpdate, VarUpdate

class OutlineScheduleWorkaroundPattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {
        // This only helps when outlining
        if (!op->getIr().getSessionOptions().enableOutlining) {
            return false;
        }
        if ((op->isConvertibleTo<popart::VarUpdateOp>() || op->isIpuCopyOp()) && 
            op->getSettings().schedulePriority != std::numeric_limits<double>::lowest()) {
            return true;
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        op->getSettings().schedulePriority = std::numeric_limits<double>::lowest();
        return true;
    }
};

static popart::PatternCreator<OutlineScheduleWorkaroundPattern> OutlineScheduleWorkaroundPatternCreator("OutlineScheduleWorkaroundPattern", true);