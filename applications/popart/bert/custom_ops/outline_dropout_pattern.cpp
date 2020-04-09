// Copyright 2019 Graphcore Ltd.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/logging.hpp>
#include <popart/op/dropout.hpp>

#include "detach.cpp"
#include "seed_modify.cpp"
#include "utils.cpp"

// This pattern moves the seed modification outside of the dropout op so it can be outlined.
// Dropout(s, m) -> Dropout(SeedModify(s, m), 0)
//
// PopART's Dropout currently uses the seedModifier value as a key to keep track of reference tensors.
// So each "outlined" dropout must have a unique seed modifier. To achieve this:
//   1) seedModifier is set to a global constant value (0)
//   2) the subgraphEquivId is calculated to take into account any additional outlining attributes, ie vGraph, Inputs, Outputs
//   3) Each subgraphEquivId is given a unique value stored in "seeds"
//
// Dropout(x, s, m) -> Dropout(x, SeedModify(s, m), seeds[equivId])
//   where equivId = subgraphEquivIdDropout(x, s, 0)

static constexpr uint32_t SEED_MODIFIER = 0;

class OutlineDropoutPattern : public popart::PreAliasPattern {
    mutable std::map<std::string, uint32_t> seeds;
    mutable uint32_t seed = SEED_MODIFIER + 1;
public:
    bool matches(popart::Op *op) const override {
        // Don't run in inference
        if (!op->getIr().canTrain()) {
            return false;
        }
        // Only run if outlining the graph
        if (!op->getIr().getSessionOptions().enableOutlining) {
            return false;
        }
        // Seeds are connected after the forwards pass preAliasPattern run, so wait until the next possible moment
        if (!op->getIr().hasConstructedBackwards()) {
            return false;
        }
        if (dynamic_cast<popart::DropoutOp *>(op)) {
            return !search_producers_for<SeedModifyOp>(op->input->tensor(op->getSeedInIndex()), 1);
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto &graph = op->getGraph();

        if (!op->input->hasIndex(op->getSeedInIndex())) {
            throw popart::internal_error("Op {} does not have a connected seed tensor after the backwards pass"
                                         " has been constructed. Has the Ir::prepare been reordered?", op->debugName());
        }

        auto s = op->input->id(op->getSeedInIndex());
        auto dropout = dynamic_cast<popart::DropoutOp *>(op);
        auto m = dropout->getSeedModifier();
        // (1)
        dropout->setSeedModifier(SEED_MODIFIER);
        // (2)
        auto hash = op->getSubgraphEquivId();
        if (seeds.find(hash) == seeds.end()) {
            seeds[hash] = seed;
            seed++;
            popart::logging::debug("Adding seed_modifier ({}) for hash {}", seeds[hash], hash);
        } else {
            popart::logging::debug("Found seed_modifier ({}) for op {}", seeds[hash], op->debugName());
        }
        // (3)
        dropout->setSeedModifier(seeds[hash]);

        auto modify_up = std::make_unique<SeedModifyOp>(
            m,
            popart::Op::Settings(graph, op->name() + "_modifier"));

        auto modify = modify_up.get();
        transferBaseProperties(op, modify);
        graph.moveIntoGraph(std::move(modify_up));

        modify->connectInTensor(SeedModifyOp::getInputIndex(), s);
        auto modified_seed_tensor = s + "_modified_for_" + std::to_string(op->id);
        modify->createAndConnectOutTensor(SeedModifyOp::getOutputIndex(), modified_seed_tensor);

        modify->setup();

        op->disconnectInTensor(op->getSeedInIndex(), graph.getTensors().get(s));
        op->connectInTensor(op->getSeedInIndex(), modified_seed_tensor);

        return true;
    }
};

static popart::PatternCreator<OutlineDropoutPattern> OutlineDropoutPatternCreator("OutlineDropoutPattern", true);