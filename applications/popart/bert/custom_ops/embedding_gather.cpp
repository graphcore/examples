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

#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/op.hpp>
#include <popart/op/gather.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/op/gatherx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/util.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Cast.hpp>

namespace CustomOperators
{
  const popart::OperatorIdentifier EmbeddingGather = {"ai.graphcore", "EmbeddingGather", 1};
} // namespace CustomOperators


class EmbeddingGatherOp : public popart::GatherOp
{
public:
  EmbeddingGatherOp(int64_t axis_, const popart::Op::Settings &settings_)
      : popart::GatherOp(CustomOperators::EmbeddingGather, axis_, settings_) {}

  std::unique_ptr<Op> clone() const override
  {
    return std::make_unique<EmbeddingGatherOp>(*this);
  }
  bool check_indices = true;
};

class EmbeddingGatherOpx : public popart::popx::Opx
{
public:
  EmbeddingGatherOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex)
  {
    verifyOp<EmbeddingGatherOp>(op, CustomOperators::EmbeddingGather);
    // We always want this to layout its inputs
    inputCreatorPriority = std::numeric_limits<double>::max();
  }

  bool createsEquiv(int, const popart::popx::Opx *, int) const final { return false; }

  std::set<popart::TensorId> mustExistBeforeCreate(int) const final { return {}; }

  popart::popx::InputCreatorType getInputCreatorType(int index0) const final {
    return index0 == EmbeddingGatherOp::dataInIndex() ? popart::popx::InputCreatorType::CanCreate
                                                 : popart::popx::Opx::getInputCreatorType(index0);
  }

  poplar::Tensor createInput(int index,
                             const poplar::DebugNameAndId &dnai) const final {
    popart::logging::debug("EmbeddingGather asked to create index {}: name {}", index, dnai);
    if (index != EmbeddingGatherOp::dataInIndex())
    {
      throw popart::error("CustomOps Error: GatherOpx::createInput Cannot create input {}", index);
    }

    auto inputInfo = inInfo(EmbeddingGatherOp::indicesInIndex());
    auto weightInfo = inInfo(EmbeddingGatherOp::dataInIndex());

    unsigned inputSize = inputInfo.nelms();
    unsigned inChannels = weightInfo.dim(0);
    unsigned outChannels = weightInfo.nelms() / inChannels;

    std::vector<std::size_t> lhsShape = {inputSize, inChannels};
    std::vector<std::size_t> rhsShape = {inChannels, outChannels};

    return poplin::createMatMulInputRHS(graph(),
                                        popart::popx::popType(weightInfo),
                                        lhsShape,
                                        rhsShape,
                                        dnai,
                                        {},
                                        &dv_p->matmulCache);
  }

  // Identical to popart::opx::GatherOpx::grow however:
  //    1) uses popops::gather instead of popops::multislice
  //    2) range checks the indices and masks those out of range
  void grow(poplar::program::Sequence &prog) const final {
    const auto indicesShape = inShape(EmbeddingGatherOp::indicesInIndex());
    const auto outputShape =
        popart::vXtoY<int64_t, std::size_t>(outShape(EmbeddingGatherOp::outIndex()));

    auto op = getOp<EmbeddingGatherOp>();
    unsigned axis = op.getAxis();
    auto indices = getInTensor(EmbeddingGatherOp::indicesInIndex());
    auto data = getInTensor(EmbeddingGatherOp::dataInIndex());

    // If there are no indices, return an empty tensor of the appropriate
    // shape
    if (indices.numElements() == 0)
    {
      auto result = graph().addVariable(
          data.elementType(), outputShape, debugContext("result"));

      setOutTensor(EmbeddingGatherOp::outIndex(), result);
    }
    else
    {
      // Flatten the scalar indices.
      auto offsets = indices.flatten();
      // reinterpret the indices as unsigned int. This assumes negative indices.
      // are impossible.
      offsets = offsets.reinterpret(poplar::UNSIGNED_INT);

      // Place the gather axis at the front.
      data = data.dimShufflePartial({0}, {axis});
      // Store the shape for later.
      auto tmp_shape = data.shape();
      // Flatten the other dimensions.
      data = data.flatten(1, data.rank());

      // Change (2)
      poplar::Tensor mask;
      if (op.check_indices)
      {
        auto gather_size = data.shape()[0];
        mask = popops::lt(graph(), offsets, static_cast<unsigned>(gather_size), prog, debugContext("mask<size"));
        auto indices_mask = popops::cast(graph(), mask, offsets.elementType(), prog, debugContext("mask_castInt"));
        offsets = popops::mul(graph(), offsets, indices_mask, prog, debugContext("masked_indices"));
      }

      // Change (1)
      auto result = popops::gather(graph(),
                                   data,
                                   offsets,
                                   0,
                                   prog,
                                   popops::GatherParams(),
                                   debugContext());

      // // Change (2)
      if (op.check_indices)
      {
        auto out_mask = popops::cast(graph(), mask, data.elementType(), prog, debugContext("mask_cast"));
        popops::mulInPlace(graph(), result, out_mask.expand({1}), prog, debugContext("masked_result"));
      }

      // Reshape the result to "unflatten" the other dimensions.
      tmp_shape.front() = result.dim(0);
      result = result.reshape(tmp_shape);
      // Put the gather axis dimension back in the right place.
      result = result.dimShufflePartial({axis}, {0});

      // Reshape into the expected ONNX shape.
      result = result.reshape(outputShape);

      setOutTensor(EmbeddingGatherOp::outIndex(), result);
    }
  }
};

static popart::popx::OpxCreator<EmbeddingGatherOpx>
    EmbeddingGatherOpxCreator(CustomOperators::EmbeddingGather);

static popart::OpDefinition EmbeddingGatherOpDef({});

static popart::OpCreator<EmbeddingGatherOp> EmbeddingGatherOpCreator(
    popart::OpDefinitions({{CustomOperators::EmbeddingGather,
                            EmbeddingGatherOpDef}}),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
      int64_t axis =
          oci.attributes.getAttribute<popart::Attributes::Int>("axis", 0);
      return std::unique_ptr<EmbeddingGatherOp>(
          new EmbeddingGatherOp(axis, oci.settings));
    },
    true);
