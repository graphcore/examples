// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
#include "TileMappingCommon.hpp"
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <popart/popx/opxmanager.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/Rearrange.hpp>
#include <poplar/ArrayRef.hpp>
#include <popart/shapeinference.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/region.hpp>
#include <functional>

static snap::Tensor add_graph_prog(snap::Graph&              graph, 
                                   snap::program::Sequence&  prog,
                                   snap::Tensor const&       input,
                                   int64_t                   grain_size,
                                   bool                      clone_layout,
                                   bool                      after_matmul,
                                   bool                      isBwd,
                                   std::string const&        debug_str)
{
  std::string  str_mark = true == isBwd ? "Grad" : "";
  std::string  strOp = std::string("/remapCE") + str_mark;
  snap::Tensor src   = input;
  snap::Tensor out;
  if(false == clone_layout)
  {
    out = graph.addVariable(src.elementType(), src.shape(), debug_str + strOp + std::string("_out"));
    size_t                channelCnt  = out.numElements() / grain_size;
    poplar::Target const& target      = graph.getTarget();
    unsigned int          numTiles    = target.getNumTiles();
    unsigned int          tilesPerIPU = target.getTilesPerIPU();
    if(true == after_matmul){
      bool    regroup_res = false;
      size_t  input_rank  = src.rank();
      if(input_rank < 2){
        throw poplar::poplar_error("input_tensor.rank should be >= 2, if fwd_after_matmul is true");
      }
      size_t  x_dim_size   = src.dim(input_rank - 1);
      size_t  y_dim_size   = src.numElements() / x_dim_size;
      auto    src_reshape  = src.reshape({ 
                                         y_dim_size, 
                                         x_dim_size
                                        });
      int regroup_size = 1;
      if(0 == (x_dim_size & 15)){
        regroup_size = 16;
      }else if(0 == (x_dim_size & 7)){
        regroup_size = 8;
      }else if(0 == (x_dim_size & 3)){
        regroup_size = 4;
      }

      if(regroup_size > 1){
        poplar::Tensor&  src_reshape_poplar = src_reshape.getPoplarTensor();
        const auto       inGrouping = poputil::detectDimGroupings(graph.getPoplarGraph(), src_reshape_poplar);  
        if(!inGrouping.empty()){
          if((inGrouping[0].first == 1) &&
             ((0 == (inGrouping[0].second % regroup_size)) ||
              (inGrouping[0].second > regroup_size) ||
              (8 == inGrouping[0].second))){
            regroup_res = true;
          }else{
            auto             input_tilemapping = graph.getPoplarGraph().getTileMapping(src_reshape.getPoplarTensor());
            poplar::Tensor   regroup    = popops::rearrange::regroupIfBeneficial(graph.getPoplarGraph(), 
                                                                                src_reshape.getPoplarTensor(), 
                                                                                regroup_size, 
                                                                                prog.getPoplarSequence(), 
                                                                                { debug_str + std::string("/regroupCE_out") });
            auto              regroup_tilemapping = graph.getPoplarGraph().getTileMapping(regroup);
            if(input_tilemapping != regroup_tilemapping){
              regroup = regroup.reshape(src.shape());
              src     = snap::Tensor(regroup, graph);
              regroup_res = true;
            }
          }
        }
      }

      if(false == regroup_res){
        size_t  grain_size_after_matmul = grain_size;
        if(0 == (x_dim_size & 15)){
          grain_size_after_matmul = 16;
        }else if(0 == (x_dim_size & 7)){
          grain_size_after_matmul = 8;
        }else if(0 == (x_dim_size & 3)){
          grain_size_after_matmul = 4;
        }else{
          throw poplar::poplar_error("current tensor's last dim size is not 4x/8x/16x");
        }
        src_reshape = src.reshape({ 
                                    y_dim_size, 
                                    x_dim_size / grain_size_after_matmul,
                                    grain_size_after_matmul 
                                  });
        src_reshape = src_reshape.dimShuffle( { 1, 0, 2 } );
        auto  matmul_out_remap = graph.addVariable(src.elementType(), 
                                                    { 
                                                      x_dim_size / grain_size_after_matmul, 
                                                      y_dim_size, 
                                                      grain_size_after_matmul 
                                                    }, 
                                                    debug_str + std::string("/matmul_out_remap"));
        SplitChannelInfo splitInfo    = splitChannelByGroup((x_dim_size / grain_size_after_matmul) * y_dim_size,  
                                                            1, numTiles, tilesPerIPU);
        auto  matmulOutRemapReshape = matmul_out_remap.getPoplarTensor().reshape({ (x_dim_size / grain_size_after_matmul) * y_dim_size, 
                                                                                  (size_t)grain_size_after_matmul });
        std::vector<size_t> const& tileStart  = std::get<0>(splitInfo);
        std::vector<size_t> const& tileCount  = std::get<1>(splitInfo);
        for (unsigned i = 0; i < numTiles; ++i)
        {
          if(0 == tileCount[i])
            continue;
          
          poplar::Tensor curOut = matmulOutRemapReshape.slice(tileStart[i], tileStart[i] + tileCount[i], 0).flatten();
          graph.getPoplarGraph().setTileMapping(curOut, i);
        }
        prog.add(snap::program::Copy(src_reshape, matmul_out_remap));
        matmul_out_remap = matmul_out_remap.dimShuffle( { 1, 0, 2 } );
        matmul_out_remap = matmul_out_remap.reshape( {y_dim_size, x_dim_size} );
        matmul_out_remap = matmul_out_remap.reshape(src.shape());
        src              = matmul_out_remap;
      }
    }
    if(channelCnt*grain_size == out.numElements()){
      SplitChannelInfo splitInfo    = splitChannelByGroup(channelCnt,  1, numTiles, tilesPerIPU);
      auto  outReshape = out.getPoplarTensor().reshape({ channelCnt, (size_t)grain_size });
      std::vector<size_t> const& tileStart  = std::get<0>(splitInfo);
      std::vector<size_t> const& tileCount  = std::get<1>(splitInfo);
      for (unsigned i = 0; i < numTiles; ++i)
      {
        if(0 == tileCount[i])
          continue;
        
        poplar::Tensor curOut = outReshape.slice(tileStart[i], tileStart[i] + tileCount[i], 0).flatten();
        graph.getPoplarGraph().setTileMapping(curOut, i);
      }
    }
    else{
      poputil::mapTensorLinearly(graph.getPoplarGraph(), out.getPoplarTensor(), 1, grain_size);
    }
  }
  else{
    out = graph.clone(src, debug_str + strOp + std::string("/_clone_out"));
  }
    
  prog.add(snap::program::Copy(src, out));
  return out;
}

RemapCEOp::RemapCEOp(OperatorIdentifier const& opid, 
                     Op::Settings const&       settings_, 
                     int64_t                   fwd_grain_size,
                     int64_t                   bwd_grain_size,
                     bool                      fwd_clone_layout,
                     bool                      bwd_clone_layout,
                     bool                      fwd_after_matmul,
                     bool                      bwd_after_matmul,
                     std::string const&        debug_str):Op(opid, settings_) {
  fwd_grain_size_   = fwd_grain_size;
  bwd_grain_size_   = (0 == bwd_grain_size ? fwd_grain_size : bwd_grain_size);
  fwd_clone_layout_ = fwd_clone_layout;
  bwd_clone_layout_ = bwd_clone_layout;
  fwd_after_matmul_ = fwd_after_matmul;
  bwd_after_matmul_ = bwd_after_matmul;
  debug_str_        = debug_str;
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
  outInfo(0) = {inInfo(0).dataType(), data_shape};
}

poprithms::memory::inplace::Proposal
RemapCEOp::mapInplaceProposal(const AliasModel &aliasModel,
                                 OperatorIdentifier id) const {
  if(true == fwd_clone_layout_){
    return mapInplaceProposalGate0(aliasModel, id);
  }
  return Op::mapInplaceProposal(aliasModel, id);
}

ReplicatedTensorShardingIndices
RemapCEOp::getReplicatedTensorShardingIndices() const {
  return {{{0}, {0}}};
}

void RemapCEOp::growAliasModel(AliasModel &m) const {
  if(true == fwd_clone_layout_){
    m.insertUnaryModifier0(*this);
  } else {
    Op::growAliasModel(m);
  }
}

std::vector<std::tuple<OperatorIdentifier, float>>
RemapCEOp::inplacePriorityDefault() const {
  if(true == fwd_clone_layout_){
    return {{CustomOperators::remapCEInplaceId, 10.0f}};
  }
  return {};
}

std::unique_ptr<Op>
RemapCEOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == CustomOperators::remapCEInplaceId) {
    return std::make_unique<RemapCEInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

std::unique_ptr<Op> RemapCEInplaceOp::clone() const {
  return std::make_unique<RemapCEInplaceOp>(*this);
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
          {"fwd_grain_size", {"*"}},
          {"bwd_grain_size", {"*"}},
          {"fwd_clone_layout", {"*"}},
          {"bwd_clone_layout", {"*"}},
          {"fwd_after_matmul", {"*"}},
          {"bwd_after_matmul", {"*"}},
        }
      )
    });

static OpCreator<RemapCEOp> RemapOpCECreator(
    OpDefinitions({{CustomOperators::remapCEId, remapCEOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
       OperatorIdentifier const& opid             = info.opid;
       Op::Settings const&       settings_        = info.settings;
       Attributes const&         attr             = info.attributes;
       int64_t                   fwd_grain_size   = attr.getAttribute<Attributes::Int>("fwd_grain_size");
       int64_t                   bwd_grain_size   = attr.getAttribute<Attributes::Int>("bwd_grain_size");
       int64_t                   fwd_clone_layout = attr.getAttribute<Attributes::Int>("fwd_clone_layout");
       int64_t                   bwd_clone_layout = attr.getAttribute<Attributes::Int>("bwd_clone_layout");
       int64_t                   fwd_after_matmul = attr.getAttribute<Attributes::Int>("fwd_after_matmul");
       int64_t                   bwd_after_matmul = attr.getAttribute<Attributes::Int>("bwd_after_matmul");
       std::string               debug_str        = attr.getAttribute<Attributes::String>("debug_str", "remap");
      return std::unique_ptr<Op>(new RemapCEOp(opid, 
                                               settings_, 
                                               fwd_grain_size, 
                                               bwd_grain_size, 
                                               fwd_clone_layout, 
                                               bwd_clone_layout, 
                                               fwd_after_matmul,
                                               bwd_after_matmul,
                                               debug_str));
    },
    true);


RemapCEBaseOpx::RemapCEBaseOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {}

InputCreatorType RemapCEBaseOpx::getInputCreatorType(InIndex) const {
  RemapCEOp& remapOp = getOp<RemapCEOp>();
  if(true == remapOp.isFwdCloneLayout()){
    return InputCreatorType::CanUnwind;
  }
  return InputCreatorType::Deadend;
}

snap::Tensor RemapCEBaseOpx::unwindTensorLayout(snap::Tensor tensor,
                                                InIndex,
                                                OutIndex) const {
  return tensor;
}

view::RegMap RemapCEBaseOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

RemapCEOutplaceOpx::RemapCEOutplaceOpx(Op *op, Devicex *devicex) : RemapCEBaseOpx(op, devicex) {

  verifyOp<RemapCEOp>(op, CustomOperators::remapCEId);
}

void RemapCEOutplaceOpx::grow(snap::program::Sequence& prog) const {
  auto               input_tensor = getInTensor(0);
  RemapCEOp&         remap_op     = getOp<RemapCEOp>();
  int64_t            grain_size   = remap_op.getFwdGrainSize();
  bool               clone_layout = remap_op.isFwdCloneLayout();
  bool               after_matmul = (0 != remap_op.isFwdAfterMatmul() ? true : false);
  std::string const& debug_str    = remap_op.getDebugStr();
  auto               out_tensor   = add_graph_prog(graph(), 
                                                   prog, 
                                                   input_tensor, 
                                                   grain_size, 
                                                   clone_layout, 
                                                   after_matmul,
                                                   false,
                                                   debug_str);
  setOutTensor(0, out_tensor);
}

RemapCEInplaceOpx::RemapCEInplaceOpx(Op *op, Devicex *devx)
    : RemapCEBaseOpx(op, devx) {
  verifyOp<RemapCEInplaceOp>(op, CustomOperators::remapCEInplaceId);
}

void RemapCEInplaceOpx::grow(snap::program::Sequence &prog) const {
  auto input_tensor = getInTensor(0);
  RemapCEOp& remapOp = getOp<RemapCEOp>();
  setOutTensor(0, input_tensor);
}

RemapCEGradOp::RemapCEGradOp(const RemapCEOp &fwdOp)
    : popart::Op(CustomGradOperators::remapCEGradId, fwdOp.getSettings()) {
  grain_size_    = fwdOp.getBwdGrainSize();
  clone_layout_  = fwdOp.isBwdCloneLayout();
  after_matmul_  = fwdOp.isBwdAfterMatmul();
  debug_str_     = fwdOp.getDebugStr();
}

std::unique_ptr<Op> RemapCEGradOp::clone() const {
  return std::make_unique<RemapCEGradOp>(*this);
}

void RemapCEGradOpx::grow(snap::program::Sequence &prog) const {

  auto                grad_out_tensor = getInTensor(0);
  RemapCEGradOp&      grad_op         = getOp<RemapCEGradOp>();
  int64_t             grain_size      = grad_op.getGrainSize();
  bool                clone_layout    = grad_op.isCloneLayout();
  bool                after_matmul    = grad_op.isAfterMatmul();
  std::string const&  debug_str       = grad_op.getDebugStr();
  auto                out_tensor      = add_graph_prog(graph(), 
                                                       prog, 
                                                       grad_out_tensor, 
                                                       grain_size, 
                                                       clone_layout, 
                                                       after_matmul,
                                                       true,
                                                       debug_str);
  setOutTensor(0, out_tensor);
}

static popart::popx::OpxCreator<RemapCEOutplaceOpx> RemapCEOpxCreator(CustomOperators::remapCEId);
static popart::popx::OpxCreator<RemapCEInplaceOpx> RemapCEOpxInplaceCreator(CustomOperators::remapCEInplaceId);

static popart::popx::OpxCreator<RemapCEGradOpx> RemapCEGradOpxCreator(CustomGradOperators::remapCEGradId);

static popart::RegisterShapeInferenceFunction
    remapCEOpShapeInference(CustomOperators::remapCEId,
                         [](auto &ctx) { ctx.outInfo(0) = ctx.inInfo(0); });
static popart::RegisterShapeInferenceFunction
    remapCEInplaceOpShapeInference(CustomOperators::remapCEInplaceId,
                         [](auto &ctx) { ctx.outInfo(0) = ctx.inInfo(0); });