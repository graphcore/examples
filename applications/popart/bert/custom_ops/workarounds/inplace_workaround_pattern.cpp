// Copyright 2019 Graphcore Ltd.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/topocons.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/reshape.hpp>

// This pattern is a workaround for the strict inplacing logic in popart that
// prevents inplacing operations that are:
//   * Recomputed
//   * Consume a tensor with no producer (Stream or Variable)
//
// This will perform the inplacing itself on only a specific operations ie Reshape.
// This is not safe to do in the general case and requires additional analysis to prevent
// inplace modifying Ops being created without tracking aliases of tensors.

class InplaceWorkaroundPattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {
        // No recompute in inference
        if (!op->getIr().canTrain()) {
            return false;
        }
        // InplaceOps do not have a grad definitions so wait until after the bwd pass has been
        // constructed to change to inplace
        if (!op->getIr().hasConstructedBackwards()) {
            return false;
        }
        // Note: Not checking for Recompute as that annotation is added in the
        // pipelining/recompute transformation which happens after the last PreAliasPattern run.
        if (op->isConvertibleTo<popart::ReshapeOp>()) {
            for (auto &index_tensor : op->input->tensorMap()) {
              auto inTensor = index_tensor.second;
              if (!inTensor->hasProducer()) {
                return true;
              }
            }
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto &graph = op->getGraph();
        auto reshape_up = op->getInplaceVariant(popart::Onnx::CustomOperators::ReshapeInplace);
        auto reshape = reshape_up.get();
        transferBaseProperties(op, reshape);
        reshape->setName(getReplacementOpName(op, ""));
        graph.moveIntoGraph(std::move(reshape_up));

        auto inputs = op->input->tensorIdMap();
        auto outputs = op->output->tensorIdMap();

        op->disconnectAllInputs();
        op->disconnectAllOutputs();
        op->getGraph().eraseOp(op->id);

        for (auto input : inputs) {
            reshape->connectInTensor(input.first, input.second);
        }
        for (auto output : outputs) {
            reshape->connectOutTensor(output.first, output.second);
        }

        reshape->setup();
        graph.getTensors().updateAliases(reshape);

        return true;
    }
};

static popart::PatternCreator<InplaceWorkaroundPattern> InplaceWorkaroundPatternCreator("InplaceWorkaroundPattern", true);