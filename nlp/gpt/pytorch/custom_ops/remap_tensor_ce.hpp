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

#ifndef   __REMAP_TENSOR_HPP__
#define   __REMAP_TENSOR_HPP__

#include <iostream>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/devicex.hpp>

using namespace popart;
using namespace popart::popx;

namespace popart {

  namespace CustomOperators {
    const OperatorIdentifier remapCEId = {"ai.graphcore", "RemapCE", 1};
  } // namespace CustomOperators

  namespace CustomGradOperators {
    const OperatorIdentifier remapCEGradId = {"ai.graphcore", "RemapCEGrad", 1};
  }	// namespace CustomGradOperators

} // namespace popart

class RemapCEOp : public Op {
public:
  RemapCEOp(OperatorIdentifier const&   opid, 
          int64_t                       grain_size, 
          Op::Settings const&           settings, 
          std::string const&            debug_str);

  RemapCEOp(const RemapCEOp &)            = default;
  RemapCEOp &operator=(const RemapCEOp &) = delete;
  ~RemapCEOp() override                   = default;

  std::vector<std::unique_ptr<Op>>  getGradOps() final;
  std::unique_ptr<Op>               clone() const final;
  virtual void                      setup() final;
  float                             getSubgraphValue() const final { return getHighSubgraphValue(); }
  int64_t                           getGrainSize() const   { return grain_size_; }; 
  std::string const&                getDebugStr() const    { return debug_str_; };

private:
  int64_t                    grain_size_;
  std::string                debug_str_;
};

class RemapCEOpx : public Opx {

public:
  RemapCEOpx(Op *, Devicex *);
  ~RemapCEOpx() override = default;

  void grow(poplar::program::Sequence &) const final;

private:
  static poplar::Tensor add_graph_prog(poplar::Graph&              graph, 
                                       poplar::program::Sequence&  prog,
                                       poplar::Tensor const&       input,
                                       int                         grain_size,
                                       std::string const&          debug_str);
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

private:
  int64_t                    grain_size_;
  std::string                debug_str_;
};


class RemapCEGradOpx : public Opx {
public:
  RemapCEGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<RemapCEGradOp>(op, CustomGradOperators::remapCEGradId);
  }

  void grow(poplar::program::Sequence &prog) const final;

private:
  static poplar::Tensor add_graph_prog(poplar::Graph&              graph, 
                                       poplar::program::Sequence&  prog,
                                       poplar::Tensor const&       input,
                                       int                         grain_size,
                                       std::string const&          debug_str);
};

#endif