// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

#include <popart/op.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier Detach = {"ai.graphcore", "Detach", 1};
} // namespace CustomOperators

// An InplaceIdentityOp that doesn't return any grad ops. This allows you to disconnect the flow of gradients when creating the backwards pass
class DetachOp : public popart::Op {
public:
  bool pass_through_creation = false;

  DetachOp(const popart::OperatorIdentifier &_opid, const Op::Settings &settings_, bool pass_through_creation)
      : Op(_opid, settings_), pass_through_creation(pass_through_creation) {}

  void setup() final { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<DetachOp>(*this);
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static popart::OpDefinition detachOpDef({});

static popart::OpCreator<DetachOp> detachOpCreator(
    popart::OpDefinitions({{CustomOperators::Detach, detachOpDef}}),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
      int64_t pass_through_creation =
          oci.attributes.getAttribute<popart::Attributes::Int>(
              "pass_through_creation", 0);
      return std::unique_ptr<DetachOp>(
          new DetachOp(oci.opid, oci.settings, pass_through_creation));
    },
    true);

class DetachOpx : public popart::popx::Opx
{
  bool pass_through_creation = false;
public:
  DetachOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex) {
    verifyOp<DetachOp>(op, CustomOperators::Detach);
    auto detach_op = dynamic_cast<DetachOp *>(op_p);
    pass_through_creation = detach_op->pass_through_creation;
  }

  popart::popx::InputCreatorType getInputCreatorType(popart::InIndex) const {
    if (pass_through_creation)
      return popart::popx::InputCreatorType::CanUnwind;
    return popart::popx::InputCreatorType::Deadend;
  }

  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor, popart::InIndex, popart::OutIndex) const {
    return tensor;
  }

  popart::view::RegMap unwindRegion(popart::InIndex, popart::OutIndex) const {
    return [this](const popart::view::Region &r) {
      return popart::view::Regions(1, r);
    };
  }

  void grow(poplar::program::Sequence &prog) const final {
    insert(outId(0), getInTensor(0));
  }
};

static popart::popx::OpxCreator<DetachOpx>
    detachOpxCreator(CustomOperators::Detach);
