// Copyright 2019 Graphcore Ltd.
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/opidentifier.hpp>
#include <popart/topocons.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/add.hpp>
#include <popart/op/subtract.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/transpose.hpp>
#include <popart/op/sgd1varupdate.hpp>

#include <map>

#include "sparse_sgd1_accumulate.cpp"
#include "detach.cpp"
#include "tied_gather.cpp"
#include "utils.cpp"

using SerialiseSettings = popart::MatMulBaseOp::SerialiseSettings;

// This pattern matches for graphs of the shape.
//
//              Weight
//             /     \     
//        Transpose   MatMul
//            |
// Indices --Gather
//
// And performs the following transformations:
//    1) Disable FullyConnectedPass on MatMul
//    2) Add Detach between the Gather and the Weight so no SGD ops are created (they will be added later by TiedGatherGradPattern)
//    3) Replace Gather with TiedGather
// Resulting in:
//              Weight
//             /     \     
//        Transpose   MatMul
//            |
//          Detach
//            |
// Indices --TiedGather
//
// Conditionally, if MatMul is annotated with serialisation it will:
//    4) Replace Gather with N x TiedGather to match the serialisation on the MatMul
// Resulting in:
//    For serialisation factor: 2
//
//              Weight
//             /     \     
//        Transpose  MatMul
//            |
// Indices  Detach
//  |   |    |  |
//  |   |    | Slice--\ 
//  |   Sub -|------TiedGather
//  |        |              |
//  |       Slice--\        |
//  Sub ---------TiedGather |
//                        \ |
//                        Add
//
static bool produced_by_transpose(popart::Tensor *t) {
    return t->hasProducer() && t->getProducer()->isConvertibleTo<popart::TransposeBaseOp>();
}

class TiedGatherPattern : public popart::PreAliasPattern {
    mutable std::map<popart::Op *, popart::MatMulBaseOp *> tied_op_map;
public:
    bool matches(popart::Op *op) const override {
        // Only run in the fwd pass
        if (op->getIr().hasConstructedBackwards()) {
            return false;
        }
        if (op->isConvertibleTo<popart::GatherOp>() && !op->isConvertibleTo<TiedGatherOp>()) {
            if (produced_by_transpose(op->input->tensor(popart::GatherOp::dataInIndex()))) {
                auto matmul = weight_consumed_by<popart::MatMulBaseOp>(op->input->tensor(popart::GatherOp::dataInIndex()));
                if (matmul) {
                    tied_op_map.insert({op, matmul});
                    return true;
                }
            }
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto &graph = op->getGraph();

        auto gather = dynamic_cast<popart::GatherOp *>(op);
        auto matmul = tied_op_map[gather];

        // (1)
        matmul->setUseFullyConnectedPass(false);

        auto axis = gather->getAxis();
        auto serialisation = matmul->getSerialiseSettings();

        auto data    = gather->input->tensor(popart::GatherOp::dataInIndex());
        auto indices = gather->input->tensor(popart::GatherOp::indicesInIndex());
        auto out     = gather->output->tensor(popart::GatherOp::outIndex());

        // Disconnect "out" so it can be connected to the replacing ops.
        gather->disconnectAllOutputs();

        // (2)
        // Note: the detach can be before or after the transpose
        auto detach_up = std::make_unique<DetachOp>(
            CustomOperators::Detach,
            popart::Op::Settings(graph, "TiedGatherDetach"),
            true
        );
        auto detach = detach_up.get();
        transferBaseProperties(gather, detach);
        graph.moveIntoGraph(std::move(detach_up));
        detach->connectInTensor(0, data->id);
        auto detached_data_id = data->id + "/detached";
        detach->createAndConnectOutTensor(0, detached_data_id);
        detach->setup();
        data = graph.getTensors().get(detached_data_id);

        std::string name = gather->name();
        if (name.empty()) {
            name = std::to_string(gather->id);
        }

        auto replace_with_tied_gather = [&](popart::TensorId dict, popart::TensorId ind, int64_t i, const std::string &debugPrefix) {
            auto tied_gather_up = std::make_unique<TiedGatherOp>(
                axis,
                popart::Op::Settings(graph, debugPrefix));
            auto tied_gather = tied_gather_up.get();
            transferBaseProperties(gather, tied_gather);
            graph.moveIntoGraph(std::move(tied_gather_up));

            tied_gather->connectInTensor(TiedGatherOp::dataInIndex(), dict);
            tied_gather->connectInTensor(TiedGatherOp::indicesInIndex(), ind);

            auto out_id = out->id;
            if (i >= 0) {
                out_id = debugPrefix + ":0";
                tied_gather->createAndConnectOutTensor(TiedGatherOp::outIndex(), out_id);
            } else {
                tied_gather->connectOutTensor(TiedGatherOp::outIndex(), out_id);
            }

            graph.topoCons->transfer(gather, tied_gather);

            tied_gather->setup();

            return out_id;
        };

        if (serialisation.factor <= 1 || serialisation.mode == SerialiseSettings::Mode::None) {
            // (3)
            replace_with_tied_gather(data->id, indices->id, -1, name);
        } else {
            // (4)
            if (serialisation.mode != SerialiseSettings::Mode::OutputChannels) {
                throw popart::error("Tied Gather Pattern only supports Serialisation::Mode::OutputChannels");
            }

            auto slice_op = [&](int64_t starts, int64_t ends, const std::string &debugPrefix) {
                auto slice_up = std::make_unique<popart::SliceOp>(
                    popart::Onnx::AiOnnx::OpSet9::Slice,
                    std::vector<int64_t>({starts}),
                    std::vector<int64_t>({ends}),
                    std::vector<int64_t>({axis}),
                    popart::Op::Settings(graph, debugPrefix + "/slice"));
                auto slice = slice_up.get();
                transferBaseProperties(gather, slice);
                graph.moveIntoGraph(std::move(slice_up));
                slice->connectInTensor(popart::SliceOp::getInIndex(), data->id);
                auto data_slice = debugPrefix + "/slice:0";
                slice->createAndConnectOutTensor(popart::SliceOp::getOutIndex(), data_slice);
                slice->setup();
                return data_slice;
            };

            auto subtract_with_constant = [&](popart::Tensor *a, int64_t c, const std::string &debugPrefix) {
                auto sub_up = std::make_unique<popart::SubtractOp>(
                    popart::Onnx::Operators::Sub_7,
                    popart::Op::Settings(graph, debugPrefix + "/sub"));
                auto sub = sub_up.get();
                transferBaseProperties(gather, sub);
                graph.moveIntoGraph(std::move(sub_up));
                sub->connectInTensor(popart::SubtractOp::getArg0InIndex(), a->id);
                // Create constant to subtract from
                static unsigned i = 0;
                auto sub_const_id = a->id + "_sub_const_" + std::to_string(i++);
                popart::TensorInfo subInfo(a->info.dataType(), {1});
                std::vector<unsigned> d(1, c);
                graph.getTensors().addConstInit(sub_const_id, subInfo, d.data());
                sub->connectInTensor(popart::SubtractOp::getArg1InIndex(), sub_const_id);
                auto indices_sub = debugPrefix + "/sub:0";
                sub->createAndConnectOutTensor(popart::SubtractOp::getOutIndex(), indices_sub);
                sub->setup();
                return indices_sub;
            };

            auto add_op = [&](popart::TensorId a, popart::TensorId b, popart::TensorId out, const std::string &debugPrefix) {
                auto add_up = std::make_unique<popart::AddOp>(
                    popart::Onnx::Operators::Add_6,
                    popart::Op::Settings(graph, debugPrefix + "/add"));
                auto add = add_up.get();
                transferBaseProperties(gather, add);
                graph.moveIntoGraph(std::move(add_up));
                add->connectInTensor(popart::AddOp::getArg0InIndex(), a);
                add->connectInTensor(popart::AddOp::getArg1InIndex(), b);
                if (graph.getTensors().contains(out)) {
                    add->connectOutTensor(popart::AddOp::getOutIndex(), out);
                } else {
                    add->createAndConnectOutTensor(popart::AddOp::getOutIndex(), out);
                }
                add->setup();
                return out;
            };

            popart::TensorId tmp_id;
            for (int64_t i = 0; i < serialisation.factor; i++) {
                int64_t slice_size = data->info.dim(axis) / serialisation.factor;
                auto serial_name = name + "/" + std::to_string(i);
                // Slice the Dictionary
                auto data_slice = slice_op(i * slice_size, (i + 1) * slice_size, serial_name);
                // Subtract the indicies
                auto indices_sub = subtract_with_constant(indices, i * slice_size, serial_name);
                // Add the tied gather to the graph
                auto next_id = replace_with_tied_gather(data_slice, indices_sub, i, serial_name);

                // Add the results
                if (i == 0) {
                    tmp_id = next_id;
                } else {
                    auto out_id = out->id;
                    if (i < serialisation.factor - 1) {
                        out_id += "_tmp" + std::to_string(i);   
                    }
                    tmp_id = add_op(tmp_id, next_id, out_id, serial_name);

                    // Tie the add to happen directly after the gather
                    graph.topoCons->insert(
                        graph.getTensors().get(next_id)->getProducer(),
                        graph.getTensors().get(tmp_id)->getProducer(),
                        true);
                }
            }
        }

        gather->disconnectAllInputs();
        graph.eraseOp(gather->id);

        return true;
    }
};

// This pattern matches for graphs of the shape.
//
//    Weight
//    |              \     
// TiedGatherGrad   MatMul
//                    |
//         Accl  -  SGD1Accumulate
//                    |
//                  SGD1VarUpdate
//
// And will perform the following transformation
//   1) Replace TiedGatherGrad with SparseSGD1Accumulate
//
// Resulting in:
//
//    Weight
//    |              \     
//    |             MatMul
//    |               |
//    |    Accl  -  SGD1Accumulate
//    |     |                   |
// SparseSGD1Accumulate --> SGD1VarUpdate
//
// (--> is a topocon)

class TiedGatherGradPattern : public popart::PreAliasPattern { 
public:
    bool matches(popart::Op *op) const override {
        // Only run after the bwds has been created
        if (!op->getIr().hasConstructedBackwards()) {
            return false;
        }
        auto gather_grad = dynamic_cast<TiedGatherGradOp *>(op);
        if (gather_grad) {
            return weight_consumed_by<popart::MatMulOp>(
                gather_grad->fwd_op->input->tensor(popart::GatherOp::dataInIndex()));
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto gather_grad = dynamic_cast<TiedGatherGradOp *>(op);
        auto gather = gather_grad->fwd_op;
        auto root_weight = get_variable(gather->input->tensor(popart::GatherOp::dataInIndex()));

        // Get all the SGD1Accumulators and GatherOps of dataInIndex.
        auto update_ops = find_all_consumers<popart::SGD1VarUpdateOp>(root_weight);
        if (update_ops.size() < 1) {
            // SGD1DecomposePattern has not run.
            return false;
        }
        std::vector<popart::SGD1AccumulateOp *> accumulate_ops(update_ops.size());
        for (size_t i = 0; i < update_ops.size(); i++) {
            auto var_update = update_ops[i];
            auto accl_op = search_producers_for<popart::SGD1AccumulateOp>(var_update->input->tensor(popart::SGD1VarUpdateOp::getUpdaterInIndex()), 3);
            if (!accl_op) {
                throw popart::error("Could not find SGD1AccumulateOp for SGD1VarUpdateOp {}", var_update->name());
            }
            accumulate_ops[i] = accl_op;
        }
        
        auto gather_ops = find_all_consumers<TiedGatherOp>(root_weight);

        if (accumulate_ops.size() != gather_ops.size()) {
            throw popart::error("The number of gather ops ({}) does not match the number of accumulate ops ({}).", gather_ops.size(), accumulate_ops.size());
        }

        // Match up gather serial index to SGD1Accumulator's matmul index.
        // TODO: Find a more robust way than sorting input ids
        std::sort(accumulate_ops.begin(), accumulate_ops.end(),
                  [](const popart::Op *l, const popart::Op *r) {
                      return l->input->tensor(popart::SGD1AccumulateOp::getVarToUpdateInIndex())->id.compare(
                          r->input->tensor(popart::SGD1AccumulateOp::getVarToUpdateInIndex())->id) < 0;
                  });
        std::sort(gather_ops.begin(), gather_ops.end(), 
            [](const popart::Op *l, const popart::Op *r) {
            return l->name().compare(r->name()) < 0;
        });

        auto itr = std::find(gather_ops.begin(), gather_ops.end(), gather);
        if (itr == gather_ops.end()) {
            throw popart::error("Could not find {} in the consumers of {}.", gather->name(), root_weight->id);
        }

        unsigned serial_index = std::distance(gather_ops.begin(), itr);

        auto dense_accl = accumulate_ops[serial_index];

        auto accl_id = dense_accl->input->tensor(popart::SGD1AccumulateOp::getVarToUpdateInIndex())->id;
        auto weight_id = update_ops[serial_index]->input->tensor(popart::SGD1VarUpdateOp::getVarToUpdateInIndex())->id;
        popart::logging::pattern::info("Using tied accumulator {} for {}", accl_id, gather->name());

        // Transpose must be inplace so the accumulator is actually updated
        accl_id    = transpose_inplace(accl_id, gather_grad);
        weight_id  = transpose_inplace(weight_id, gather_grad);

        auto &graph = op->getGraph();

        // TODO: Find a way to share this code between sparse_sgd1_pattern
        // Add sparseSGD1AccumulateOp.
        auto sparse_accl_up = std::make_unique<SparseSGD1AccumulateOp>(
            accl_id,
            dense_accl->initDpsf1,
            gather_grad->getAxis(),
            popart::Op::Settings(graph, "_tiedAccumulate/" + std::to_string(serial_index)));

        auto sparse_accl = sparse_accl_up.get();
        transferBaseProperties(gather_grad, sparse_accl);
        graph.moveIntoGraph(std::move(sparse_accl_up));

        // Inputs
        // Accumulator
        sparse_accl->connectInTensor(SparseSGD1AccumulateOp::getVarToUpdateInIndex(),
                                     accl_id);
        // Gradients
        sparse_accl->connectInTensor(SparseSGD1AccumulateOp::getUpdaterInIndex(),
                                     gather_grad->inId(popart::GatherGradOp::gradInIndex()));
        // Scale
        if (!dense_accl->initDpsf1.isConst()) {
            sparse_accl->connectInTensor(
                // the index at which the dampening scale factor is received,
                SparseSGD1AccumulateOp::getDpsf1InIndex(),
                // the name of the dampening scale factor
                dense_accl->inId(popart::SGD1AccumulateOp::getDpsf1InIndex()));
        }
        // Indices
        sparse_accl->connectInTensor(SparseSGD1AccumulateOp::getIndicesInIndex(),
                                     gather_grad->inId(popart::GatherGradOp::indicesInIndex()));

        // Original weight to be cloned
        sparse_accl->connectInTensor(SparseSGD1AccumulateOp::getOriginalVarInIndex(),
                                     weight_id);

        // Transfer TopoCons
        graph.topoCons->transfer(gather_grad, sparse_accl);

        // gatherGrad output that will be isolated
        auto grad_Id = gather_grad->outId(TiedGatherGradOp::gradOutIndex());

        // Remove TiedGatherGrad
        gather_grad->disconnectAllInputs();
        gather_grad->disconnectAllOutputs();
        graph.eraseOp(gather_grad->id);

        // Outputs
        sparse_accl->createAndConnectOutTensor(SparseSGD1AccumulateOp::getUpdatedVarOutIndex(), sparse_accl->name() + ":0");
        
        // remove the gatherGrad output
        graph.getTensors().remove(grad_Id);

        // Finalise sparse op
        sparse_accl->setup();

        auto var_update = update_ops[serial_index];
        if (should_insert_topocon(sparse_accl, var_update)) {
            graph.topoCons->insert(sparse_accl, var_update);
        }

        return true;
    }

    bool should_insert_topocon(popart::Op *sparse_accl, popart::Op *var_update) const {
        auto &opts = var_update->getIr().getSessionOptions();
        if (opts.enablePipelining) {
            return var_update->getPipelineStage() >= sparse_accl->getPipelineStage();
        }
        return true;
    }

    popart::TensorId transpose_inplace(popart::TensorId tid, popart::Op *op) const {
        auto &graph = op->getGraph();

        // TransposeInplaceOp's constructor requires a transposeOp
        auto outplace_up = std::make_unique<popart::TransposeOp>(
            popart::Onnx::AiOnnx::OpSet9::Transpose,
            std::vector<int64_t>{1, 0},
            popart::Op::Settings(graph, tid + "_Transpose"));
        auto transpose_up = outplace_up->getInplaceVariant(popart::Onnx::CustomOperators::TransposeInplace);

        auto transpose = transpose_up.get();
        transferBaseProperties(op, transpose);
        graph.moveIntoGraph(std::move(transpose_up));

        transpose->connectInTensor(popart::TransposeOp::getInIndex(), tid);
        popart::TensorId out_id = tid + "/transposed";
        transpose->createAndConnectOutTensor(popart::TransposeOp::getOutIndex(), out_id);

        transpose->setup();
        return out_id;
    }
};

static popart::PatternCreator<TiedGatherPattern> TiedGatherPatternCreator("TiedGatherPattern", true);
static popart::PatternCreator<TiedGatherGradPattern> TiedGatherGradPatternCreator("TiedGatherGradPattern", true);
