// Copyright 2020 Graphcore Ltd.
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
#include <popart/op/sgd0varupdate.hpp>
#include <queue>

#include <popart/opidentifier.hpp>
#include <popart/op/clip.hpp>

#include "compile_time_version.h"

const popart::OperatorIdentifier GradClip = popart::Onnx::Operators::Clip_6;

bool check_if_updater_grad_clip(popart::Op *op, popart::InIndex updater_index) {
    // this helper function checks if the producer of the updater is a GradClip op
    auto grad = op->input->tensor(updater_index);
    auto producer_opid = grad->getProducer()->opid;
    return producer_opid == GradClip;
}

// This pattern performs gradient clipping

class GradientClippingPattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {

        popart::InIndex updater_index;
        if (op->isConvertibleTo<popart::SGD0VarUpdateOp>()) {
            updater_index = popart::SGD0VarUpdateOp::getUpdaterInIndex();
            // if producer of updater is GradClip no need to apply pattern
            if (check_if_updater_grad_clip(op, updater_index)) {
                return false;
            }
            return true;
        }
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto &ir = op->getIr();
        auto &graph = op->getGraph();

        popart::InIndex updater_index;
        popart::InIndex ref_index;
       
        if (op->isConvertibleTo<popart::SGD0VarUpdateOp>()) {
            updater_index = popart::SGD0VarUpdateOp::getUpdaterInIndex();
        }
        // specify the clip magnitudes below
        auto gradient_clip_op = std::make_unique<popart::ClipOp>(
            GradClip, -5.0, 5.0, popart::Op::Settings(graph, op->name() + "/gradient_clip"));
        auto gradient_clip = gradient_clip_op.get();
        transferBaseProperties(op, gradient_clip);
        graph.moveIntoGraph(std::move(gradient_clip_op));

        auto grad = op->input->tensor(updater_index);
        op->disconnectInTensor(grad);
        gradient_clip->connectInTensor(gradient_clip->getInIndex(), grad->id);
		
        auto grad_clipped = grad->id + "_clipped";
        gradient_clip->createAndConnectOutTensor(gradient_clip->getOutIndex(), grad_clipped);
        op->connectInTensor(updater_index, grad_clipped);
		
        gradient_clip->setup();
				
        return true;
    }
};

static popart::PatternCreator<GradientClippingPattern> gradientClippingPatternCreator("GradientClippingPattern", true);
