// Copyright 2019 Graphcore Ltd.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/topocons.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/sgd1accumulate.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <iostream>
#include <queue>

#include "detach.cpp"
#include "embeddingGather.hpp"
#include "utils.cpp"


// This pattern disables fullyConnectedPass for all matmuls that use a tied weight.
// TODO: Make this pattern handle all other requirements to replace embeddingGatherGrad with sparseSGD1Accumulate.

class TiedGatherPattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {
        // Only run in the fwd pass. The updated option will be propagated to any other related matmuls
        if (op->getIr().hasConstructedBackwards()) {
            return false;
        }
        auto matmul = dynamic_cast<popart::MatMulOp *>(op);
        if (matmul) {
            return matmul->useFullyConnectedPass() &&
                   weight_consumed_by<EmbeddingGatherOp>(matmul->input->tensor(popart::MatMulOp::getRhsInIndex()));
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto matmul = dynamic_cast<popart::MatMulOp *>(op);
        matmul->setUseFullyConnectedPass(false);
        return true;
    }
};

static popart::PatternCreator<TiedGatherPattern> tiedGatherPatternCreator("TiedGatherPattern", true);