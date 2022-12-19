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
#ifndef   __REMAP_TENSOR_HPP__
#define   __REMAP_TENSOR_HPP__

#include <iostream>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/popopx.hpp>

using namespace popart;
using namespace popart::popx;

namespace popart {

  namespace CustomOperators {
    const OperatorIdentifier remapCEId = {"ai.graphcore", "RemapCE", 1};
    const OperatorIdentifier remapCEInplaceId = {"ai.graphcore", "RemapCEInplace", 1};
  } // namespace CustomOperators

  namespace CustomGradOperators {
    const OperatorIdentifier remapCEGradId = {"ai.graphcore", "RemapCEGrad", 1};
  }	// namespace CustomGradOperators

} // namespace popart

class RemapCEOp : public Op {
public:
  RemapCEOp(OperatorIdentifier const&   opid, 
            Op::Settings const&         settings_, 
            int64_t                     fwd_grain_size, 
            int64_t                     bwd_grain_size, 
            bool                        fwd_clone_layout,
            bool                        bwd_clone_layout,
            bool                        fwd_after_matmul,
            bool                        bwd_after_matmul,
            std::string const&          debug_str);

  RemapCEOp(const RemapCEOp &)            = default;
  RemapCEOp &operator=(const RemapCEOp &) = delete;
  ~RemapCEOp() override                   = default;

  std::vector<std::unique_ptr<Op>>  getGradOps() final;
  std::unique_ptr<Op>               clone() const override;
  virtual void                      setup() final;

  bool canShard() const override { return false; };
  virtual bool isIdentity() const { return canBeReplacedByIdentity(); };

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  virtual void growAliasModel(AliasModel &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  float                             getSubgraphValue() const final { return getLowSubgraphValue(); }
  int64_t                           getFwdGrainSize() const  { return fwd_grain_size_; }; 
  int64_t                           getBwdGrainSize() const  { return bwd_grain_size_; }; 
  int64_t                           isFwdCloneLayout() const { return fwd_clone_layout_; };
  int64_t                           isBwdCloneLayout() const { return bwd_clone_layout_; };
  int64_t                           isFwdAfterMatmul() const { return fwd_after_matmul_; };
  int64_t                           isBwdAfterMatmul() const { return bwd_after_matmul_; };
  std::string const&                getDebugStr() const      { return debug_str_; };

private:
  int64_t                    fwd_grain_size_;
  int64_t                    bwd_grain_size_;
  int64_t                    fwd_clone_layout_;
  int64_t                    bwd_clone_layout_;
  int64_t                    fwd_after_matmul_;
  int64_t                    bwd_after_matmul_;
  std::string                debug_str_;
};

class RemapCEInplaceOp : public RemapCEOp {
public:
  RemapCEInplaceOp(OperatorIdentifier const &opid,
                   Op::Settings const       &settings_,
                   int64_t                   fwd_grain_size, 
                   int64_t                   bwd_grain_size, 
                   bool                      fwd_clone_layout,
                   bool                      bwd_clone_layout,
                   bool                      fwd_after_matmul,
                   bool                      bwd_after_matmul,
                   std::string const&        debug_str)
      : RemapCEOp(opid, 
                  settings_, 
                  fwd_grain_size, 
                  bwd_grain_size, 
                  fwd_clone_layout, 
                  bwd_clone_layout,
                  fwd_after_matmul, 
                  bwd_after_matmul,
                  debug_str) {};
  RemapCEInplaceOp(const RemapCEOp &op)
      : RemapCEOp(CustomOperators::remapCEInplaceId,
                  op.getSettings(),
                  op.getFwdGrainSize(),
                  op.getBwdGrainSize(),
                  op.isFwdCloneLayout(),
                  op.isBwdCloneLayout(),
                  op.isFwdAfterMatmul(),
                  op.isBwdAfterMatmul(),
                  op.getDebugStr()){};

  std::unique_ptr<Op> clone() const final;

  view::Regions modifies(InIndex index) const final { return uses(index); }
  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
};

class RemapCEBaseOpx : public PopOpx {

public:
  RemapCEBaseOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  snap::Tensor
      unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

class RemapCEOutplaceOpx : public RemapCEBaseOpx {

public:
  RemapCEOutplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &prog) const final;
};

class RemapCEInplaceOpx : public RemapCEBaseOpx {

public:
  RemapCEInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &prog) const final;
};

class RemapCEGradOp : public Op {
public:
  RemapCEGradOp(const RemapCEOp &fwdOp);

  std::unique_ptr<Op> clone() const final;
  virtual void        setup() {

    outInfo(0) = {inInfo(0).dataType(), inInfo(0).shape()};
  }

  /* Describes the relationship of the inputs of the grad op to the
     inputs/outputs of the non-grad op */
  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {

    static const std::vector<popart::GradInOutMapper> in_info = {
      // The input of grad op at index 0 is the gradient of the input at
      // index 0 of the non-grad op
      {0, 0, popart::GradOpInType::GradOut}, // gradient of output
      //{1, 0, popart::GradOpInType::Out}, // output
    };

    return in_info;
  }

  /* Describes the relationship of the outputs of the grad op to the
     inputs/outputs of the non-grad op */
  virtual const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> out_info = {
      // The output at index 0 is dLhs, i.e the gradient of the input at index 0
      // of non grad op
      {0, 0},
    };
    return out_info;
  }

  float getSubgraphValue() const final   { return getLowSubgraphValue();} ;
  const std::string& getDebugStr() const { return debug_str_; } ;
  int64_t getGrainSize() const           { return grain_size_; };
  int64_t isCloneLayout() const          { return clone_layout_; };
  int64_t isAfterMatmul() const          { return after_matmul_; };

private:
  int64_t                    grain_size_;
  int64_t                    clone_layout_;
  int64_t                    after_matmul_;
  std::string                debug_str_;
};


class RemapCEGradOpx : public PopOpx {
public:
  RemapCEGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::PopOpx(op, devicex) {
    verifyOp<RemapCEGradOp>(op, CustomGradOperators::remapCEGradId);
  }

  void grow(snap::program::Sequence &prog) const final;
};

#endif