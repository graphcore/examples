// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "remap_tensor_ce.hpp"
#include <popart/shapeinference.hpp>
#include <popart/popx/opxmanager.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poplar/ArrayRef.hpp>
#include <functional>

RemapCEOp::RemapCEOp(OperatorIdentifier const& opid, 
                     int64_t                   grain_size,
                     Op::Settings const&       settings, 
                     std::string const&        debug_str):Op(opid, settings) {
  grain_size_ = grain_size;
  debug_str_  = debug_str;
}

std::vector<std::unique_ptr<Op>> RemapCEOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<RemapCEGradOp>(*this));

  return upops;
}

std::unique_ptr<Op> RemapCEOp::clone() const {
  return std::make_unique<RemapCEOp>(*this);
}

void RemapCEOp::setup() {
  Shape data_shape = inInfo(0).shape();
  //std::cout << "inInfo(0).dataType: " << inInfo(0).dataType() << ", shape: " << data_shape << std::endl;
  outInfo(0) = {inInfo(0).dataType(), data_shape};
}

//register op
static OpDefinition::DataTypes RemapCEOpDataTensorType = { DataType::FLOAT16,
                                                           DataType::FLOAT };

static OpDefinition
    remapCEOpDef({
      OpDefinition::Inputs
      (
        {
          {"data",  RemapCEOpDataTensorType},
        }
      ),
      OpDefinition::Outputs
      (
        {
          {"out",  RemapCEOpDataTensorType}
        }
      ),
      OpDefinition::Attributes
      (
        {
          {"grain_size", {"*"}}
        }
      )
    });

static OpCreator<RemapCEOp> RemapOpCECreator(
    OpDefinitions({{CustomOperators::remapCEId, remapCEOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
       OperatorIdentifier const& opid        = info.opid;
       Op::Settings const&       settings    = info.settings;
       Attributes const&         attr        = info.attributes;
       int64_t                   grain_size  = attr.getAttribute<Attributes::Int>("grain_size");
       std::string               debug_str   = attr.getAttribute<Attributes::String>("debug_str", "remap");
      return std::unique_ptr<Op>(new RemapCEOp(opid, grain_size, settings, debug_str));
    },
    true);


RemapCEOpx::RemapCEOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {

  verifyOp<RemapCEOp>(op, CustomOperators::remapCEId);
}

void RemapCEOpx::grow(poplar::program::Sequence& prog) const {
  auto               input_tensor = getInTensor(0);
  RemapCEOp&         remap_op     = getOp<RemapCEOp>();
  int                grain_size   = (int)(remap_op.getGrainSize());
  std::string const& debug_str    = remap_op.getDebugStr();
  auto               out_tensor   = add_graph_prog(graph(), prog, input_tensor, grain_size, debug_str);
  //std::cout << "input_tensor: " << input_tensor.elementType() << std::endl; 
  //std::cout << "out_tensor:   " << out_tensor.elementType() << std::endl; 
  setOutTensor(0, out_tensor);
}

poplar::Tensor RemapCEOpx::add_graph_prog(poplar::Graph&                   graph, 
                                          poplar::program::Sequence&       prog,
                                          poplar::Tensor const&            input,
                                          int                              grain_size,
                                          std::string const&               debug_str)
{
  poplar::Tensor out =  graph.addVariable(input.elementType(), input.shape(), debug_str + std::string("/remapCE_out"));
  poputil::mapTensorLinearly(graph, out, 1, grain_size);
  prog.add(poplar::program::Copy(input, out));

  return out;
}

RemapCEGradOp::RemapCEGradOp(const RemapCEOp &fwdOp)
    : popart::Op(CustomGradOperators::remapCEGradId, fwdOp.getSettings()) {
  grain_size_ = fwdOp.getGrainSize();
  debug_str_  = fwdOp.getDebugStr();
}

std::unique_ptr<Op> RemapCEGradOp::clone() const {
  return std::make_unique<RemapCEGradOp>(*this);
}

void RemapCEGradOpx::grow(poplar::program::Sequence &prog) const {

  poplar::Tensor      grad_out_tensor = getInTensor(0);
  RemapCEGradOp&      grad_op         = getOp<RemapCEGradOp>();
  int                 grain_size      = (int)(grad_op.getGrainSize());
  std::string const&  debug_str       = grad_op.getDebugStr();
  auto                out_tensor      = add_graph_prog(graph(), prog, grad_out_tensor, grain_size, debug_str);
  setOutTensor(0, out_tensor);
}

poplar::Tensor RemapCEGradOpx::add_graph_prog(poplar::Graph&              graph, 
                                              poplar::program::Sequence&  prog,
                                              poplar::Tensor const&       input,
                                              int                         grain_size,
                                              std::string const&          debug_str)
{
  poplar::Tensor out =  graph.addVariable(input.elementType(), input.shape(), debug_str + std::string("/remapCE_grad_out"));
  poputil::mapTensorLinearly(graph, out, 1, grain_size);
  prog.add(poplar::program::Copy(input, out));
  return out;
}

static popart::popx::OpxCreator<RemapCEOpx> RemapCEOpxCreator(CustomOperators::remapCEId);

static popart::popx::OpxCreator<RemapCEGradOpx>
RemapCEGradOpxCreator(CustomGradOperators::remapCEGradId);

static popart::RegisterShapeInferenceFunction
    remapCEOpShapeInference(CustomOperators::remapCEId,
                         [](auto &ctx) { ctx.outInfo(0) = ctx.inInfo(0); });