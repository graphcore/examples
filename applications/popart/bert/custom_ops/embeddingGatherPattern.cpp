// Copyright 2019 Graphcore Ltd.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/slice.hpp>
#include <popart/ces/slicece.hpp>
#include <iostream>
#include "embeddingGather.hpp"

class EmbeddingGatherPattern : public popart::PreAliasPattern
{
public:
  bool matches(popart::Op *op) const override {
    if (op->isConvertibleTo<EmbeddingGatherGradOp>()) {
      // The pattern will try to run against the op multiple times. Check if the inputs
      // have already been assigned and block it if they have.
      if (op->input->hasIndex(EmbeddingGatherGradOp::acclSliceInputFirstIndex())) {
        return false;
      }

      // Hold off running this until the accumlator tensors have been created by SGD1Decompose
      if (!acclTensorsCreated(op)) {
        return false;
      }

      // Don't want to apply this pattern against the positional embedding, only the word embedding
      auto embGatherGradOp = static_cast<EmbeddingGatherGradOp *>(op);
      if (embGatherGradOp->split.factor <= 1) {
        return false;
      }

      return true;
    }
    return false;
  }

  std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

  bool apply(popart::Op *op) const override {
    auto embGatherGradOp = static_cast<EmbeddingGatherGradOp *>(op);
    
    auto &ir = op->getIr();
    auto &graph = op->getGraph();

    auto inputId = embGatherGradOp->acclSliceInputFirstIndex();
    std::vector<popart::TensorId> accumulators;

    for (auto tensorId : graph.getTensors().getAllTensorIds()) {
        if (tensorId.rfind(EmbeddingGatherGradOp::acclTensorPrefix(), 0) == 0) {
            accumulators.push_back(tensorId);
        }
    }

    std::sort(accumulators.begin(), accumulators.end());

    for (size_t offset = 0; offset < accumulators.size(); offset++) {
      embGatherGradOp->connectInTensor(inputId + offset, accumulators[offset]);
    }

    if (accumulators.size() != embGatherGradOp->split.factor) {
      throw popart::error("The EmbeddingGather split.factor attribute does not match the number of accumulators found. "
                          "split.factor {} vs # accumulators {}", embGatherGradOp->split.factor, accumulators.size());
    }
    return true;
  }

private:

  bool acclTensorsCreated(popart::Op *op) const {
      auto &graph = op->getGraph();
      for (auto tensorId : graph.getTensors().getAllTensorIds()) {
        if (tensorId.rfind(EmbeddingGatherGradOp::acclTensorPrefix(), 0) == 0) {
          return true;
        }
      }
      return false;
  }
};

static popart::PatternCreator<EmbeddingGatherPattern> embPatternCreator("EmbeddingGatherPattern", true);