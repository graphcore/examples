// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// This is a custom op/pattern for computing exponential moving averages of model weights
//

#include <iostream>
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensorlocation.hpp>
#include <popart/tensornames.hpp>

#include <popops/ElementWise.hpp>
#include <popart/op/accumulate.hpp>


// The first thing to do is to provide an identifier that PopART can use later
// to address the operator.
namespace Onnx {
namespace CustomOperators {
const popart::OperatorIdentifier ExpMovAvg = {"com.acme", "ExpMovAvg", 1};
} // namespace CustomOperators
} // namespace Onnx

class ExpMovAvgOp;

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

// The forward Op
class ExpMovAvgOp : public popart::Op {
public:
  ExpMovAvgOp(const popart::OperatorIdentifier &_opid,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  // The output popart Tensor has the same inputInfo and numerical type
  // (i.e. the same TensorInfo) as the input Tensor. This function is
  // required for inputInfo/type inference
  virtual void setup() { outInfo(0) = inInfo(0); }
  
  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
  }
  
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
  }

  std::unique_ptr<Op> clone() const final { return make_unique<ExpMovAvgOp>(*this); }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool hasSideEffect() const override { return true; }
};

// describe the inputs and outputs that are supported by the operation
static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};

static popart::OpDefinition
    expMovAvgOpDef({popart::OpDefinition::Inputs({{"input", T}}),
               popart::OpDefinition::Outputs({{"output", T}}),
               popart::OpDefinition::Attributes({})});

static popart::OpCreator<ExpMovAvgOp> expMovAvgOpCreator(
    popart::OpDefinitions({{Onnx::CustomOperators::ExpMovAvg, expMovAvgOpDef}}),
	[](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
      auto expMovAvg =
          new ExpMovAvgOp(oci.opid, oci.settings);

      return std::unique_ptr<popart::Op>(expMovAvg);
    },
    true);
	
// This pattern replaces ExpMovAvgOp with appropriate AccumulateOp

class ExpMovAvgPattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {
        return(op->isConvertibleTo<ExpMovAvgOp>()) ;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto &ir = op->getIr();
        auto &graph = op->getGraph();
		
        auto ema_op = dynamic_cast<ExpMovAvgOp *>(op);

        auto accumulate_op = std::make_unique<popart::AccumulateOp>(
        popart::AccumulationType::MovingAverage, popart::OptimizerValue(0, false), popart::Op::Settings(graph, ema_op->name() + "/accumulator"));
        auto accumulate = accumulate_op.get();
        transferBaseProperties(ema_op, accumulate);
        graph.moveIntoGraph(std::move(accumulate_op));
		
        // acquiring input weight tensor		
        auto weight_tensor_to_be_averaged = ema_op->inTensor(0);
        auto accumulator_tensor = ema_op->inTensor(1);
        auto ema_factor_tensor = ema_op->inTensor(2);
        // acquiring exp-mov-avg weight tensor
        auto exp_mov_avg_weight_tensor = ema_op->outTensor(0);
		
        ema_op->disconnectAllInputs();
        ema_op->disconnectAllOutputs();
        
        auto accumulator_tensor_id = accumulator_tensor->id;
		
        accumulate->connectInTensor(accumulate->getUpdaterInIndex(), weight_tensor_to_be_averaged->id);
        accumulate->connectInTensor(accumulate->getVarToUpdateInIndex(), accumulator_tensor_id);
        accumulate->connectInTensor(accumulate->getFactorInIndex(), ema_factor_tensor->id);
        // accumulator_tensor_id and exp_mov_avg_tensor->id are different aliases to the same tensor
        accumulate->connectOutTensor(accumulate->getUpdatedVarOutIndex(), exp_mov_avg_weight_tensor->id);
		
        // this execution context ensures accumulate op is scheduled after gradient-accumulation loop
        accumulate->settings.executionContext = popart::ExecutionContext::AccumulateOuterFragment;
        // OffChip location has to be set using session options
        // accumulate->settings.tensorLocation.storage = popart::TensorStorage::OffChip;
        accumulate->setup();
		
        // delete original ExpMovAvgOp
        graph.eraseOp(ema_op->id);
				
        return true;
    }
};

static popart::PatternCreator<ExpMovAvgPattern> expMovAvgPatternCreator("ExpMovAvgPattern", true);


