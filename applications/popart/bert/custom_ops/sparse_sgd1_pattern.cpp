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
#include <popart/op/accumulate.hpp>
#include <iostream>

#include "sparse_sgd1_accumulate.cpp"

// This pattern replaces:
//  GatherGradOp -> SGD1Accumulate 
//          with
//  SparseSGD1Accumulate

class SparseSGD1AccumulatePattern : public popart::PreAliasPattern {
public:
  bool matches(popart::Op *op) const override {
    if (op->isConvertibleTo<popart::GatherGradOp>()) {
        popart::Tensor *gradient = op->outTensor(popart::GatherGradOp::gradOutIndex());
        bool has_sgd1_consumer = false;
        for (popart::Op *consumer : gradient->consumers.getOps()) {
            if (consumer->isConvertibleTo<popart::AccumulateOp>()) {
                return true;
            }
        }
    }
    return false;
  }

  std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

  bool apply(popart::Op *op) const override {
    auto &ir = op->getIr();
    auto &graph = op->getGraph();

    popart::GatherGradOp *gather_grad = dynamic_cast<popart::GatherGradOp *>(op);
    popart::AccumulateOp *dense_accl;

    popart::Tensor *gradient = op->outTensor(popart::GatherGradOp::gradOutIndex());
    for (popart::Op *consumer : gradient->consumers.getOps()) {
        if (consumer->isConvertibleTo<popart::AccumulateOp>() &&
            !consumer->isConvertibleTo<SparseSGD1AccumulateOp>()) {
            dense_accl = dynamic_cast<popart::AccumulateOp *>(consumer);
            break;
        }
    }

    popart::TensorId accl_id = dense_accl->inId(popart::AccumulateOp::getVarToUpdateInIndex());

    auto sparse_accl_up = std::make_unique<SparseSGD1AccumulateOp>(
        accl_id,
        dense_accl->getFactor(),
        gather_grad->getAxis(),
        popart::Op::Settings(graph, dense_accl->name() + "_accumulate"));

    auto sparse_accl = sparse_accl_up.get();
    transferBaseProperties(dense_accl, sparse_accl);
    graph.moveIntoGraph(std::move(sparse_accl_up));

    // Inputs
    // Accumulator
    sparse_accl->connectInTensor(SparseSGD1AccumulateOp::getVarToUpdateInIndex(),
                                 accl_id);
    // Gradients
    sparse_accl->connectInTensor(SparseSGD1AccumulateOp::getUpdaterInIndex(),
                                 gather_grad->inId(popart::GatherGradOp::gradInIndex()));
    // Scale
    if (!dense_accl->getFactor().isConst()) {
        sparse_accl->connectInTensor(
            // the index at which the dampening scale factor is received,
            SparseSGD1AccumulateOp::getDpsf1InIndex(),
            // the name of the dampening scale factor
            dense_accl->inId(popart::AccumulateOp::getFactorInIndex()));
    }
    // Indices
    sparse_accl->connectInTensor(SparseSGD1AccumulateOp::getIndicesInIndex(),
                                 gather_grad->inId(popart::GatherGradOp::indicesInIndex()));

    auto outId = dense_accl->outId(popart::AccumulateOp::getUpdatedVarOutIndex());
    auto gradId = gather_grad->outId(popart::GatherGradOp::gradOutIndex());

    // Transfer TopoCons
    graph.topoCons->transfer(gather_grad, sparse_accl);
    graph.topoCons->transfer(dense_accl, sparse_accl);

    // Delete the replaced ops
    dense_accl->disconnectAllInputs();
    dense_accl->disconnectAllOutputs();
    graph.eraseOp(dense_accl->id);
    
    gather_grad->disconnectAllInputs();
    gather_grad->disconnectAllOutputs();
    graph.eraseOp(gather_grad->id);

    // Outputs
    // Connect the updated accl
    sparse_accl->connectOutTensor(SparseSGD1AccumulateOp::getUpdatedVarOutIndex(),
                                  outId);
    // remove the gatherGrad output
    graph.getTensors().remove(gradId);

    // Finalise sparse op
    sparse_accl->setup();

    return true;
  }
};

static popart::PatternCreator<SparseSGD1AccumulatePattern> sparsesgd1PatternCreator("SparseSGD1AccumulatePattern", true);
