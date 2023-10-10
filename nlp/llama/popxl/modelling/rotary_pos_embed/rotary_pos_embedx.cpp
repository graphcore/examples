// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>
#include <poplar/Tensor.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/exceptions.hpp>

#include <gcl/Collectives.hpp>
#include <poplar/Program.hpp>
#include <poplar/TensorCloneMethod.hpp>
#include <poplar/Type.hpp>
#include <poplin/MatMul.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>
#include <popops/Fill.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/SelectScalarFromRows.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>

#include "common.hpp"
#include "rotary_pos_embed.hpp"
#include "rotary_pos_embedx.hpp"

#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace popart {
namespace popx {

namespace {

namespace pe = popops::expr;

poplar::Tensor rotateTensor(poplar::Graph &graph, poplar::Tensor &x,
                            poplar::Tensor &sin, poplar::Tensor &cos,
                            poplar::program::Sequence &prog) {
  // Get contiguous real/img parts
  auto h_2 = x.shape()[3] / 2;
  auto x1 = x.slice({0, h_2}, 3);
  auto x2 = x.slice({h_2, h_2 * 2}, 3);
  // Include broadcast dims
  sin = sin.expand({2});
  cos = cos.expand({2});
  popart::logging::info("x1/x2 {} sin/cos {}", x1.shape(), sin.shape());
  // y1 = x1 * cos - x2 * sin;
  auto y1 = popops::map(graph, (pe::_1 * pe::_4) - (pe::_2 * pe::_3),
                        {x1, x2, sin, cos}, prog);
  // y2 = x2 * cos + x1 * sin;
  auto y2 = popops::map(graph, (pe::_2 * pe::_4) + (pe::_1 * pe::_3),
                        {x1, x2, sin, cos}, prog);

  return poplar::concat({y1, y2}, 3);
}

poplar::Tensor rotateGradTensor(poplar::Graph &graph, poplar::Tensor &x,
                                poplar::Tensor &sin, poplar::Tensor &cos,
                                poplar::program::Sequence &prog) {
  // Get interleave real/img parts
  std::vector<size_t> new_shape(x.shape());
  new_shape[3] = x.shape()[3] / 2;
  new_shape.push_back(2);
  auto x_reshaped = x.reshape(new_shape);
  auto x_real = x_reshaped.slice({0, 1}, 4);
  auto x_img = x_reshaped.slice({1, 2}, 4);
  // Include broadcast dims
  sin = sin.expand({2, 3});
  cos = cos.expand({2, 3});
  // y_real = x_real * cos_reshape + x_img * sin_reshape;
  auto y_real = popops::map(graph, (pe::_1 * pe::_4) + (pe::_2 * pe::_3),
                            {x_real, x_img, sin, cos}, prog);
  // y_img = x_img * cos_reshape - x_real * sin_reshape;
  auto y_img = popops::map(graph, (pe::_2 * pe::_4) - (pe::_1 * pe::_3),
                           {x_real, x_img, sin, cos}, prog);

  return poplar::concat({y_real, y_img}, 4).reshape(x.shape());
}

} // namespace

/////////////////////////////////////////////////////////////
/// Forwards opx

RotaryPosEmbedOpx::RotaryPosEmbedOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<RotaryPosEmbedOp>(op, {RotaryPosEmbed});
}

void RotaryPosEmbedOpx::grow(poplar::program::Sequence &prog) const {
  auto rotary_dim = getOp<RotaryPosEmbedOp>().rotary_dim;

  auto x = getInTensor(0);
  auto sin = getInTensor(1);
  auto cos = getInTensor(2);

  auto x_rotated = x;
  if (rotary_dim < x.dim(3)) {
    x_rotated = x_rotated.slice(0, rotary_dim, 3);
  }

  x_rotated = rotateTensor(graph(), x_rotated, sin, cos, prog);

  if (rotary_dim < x.dim(3)) {
    x_rotated =
        poplar::concat({x_rotated, x.slice(rotary_dim, x.dim(3), 3)}, 3);
  }

  // Copy back to original layout
  auto result = graph().clone(x);
  prog.add(poplar::program::Copy(x_rotated, result));
  setOutTensor(0, result);
}

/////////////////////////////////////////////////////////////
///// Grad opx

RotaryPosEmbedGradOpx::RotaryPosEmbedGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<RotaryPosEmbedGradOp>(op, {RotaryPosEmbedGrad});
}

void RotaryPosEmbedGradOpx::grow(poplar::program::Sequence &prog) const {
  auto rotary_dim = getOp<RotaryPosEmbedGradOp>().rotary_dim;

  auto dx_rotated = getInTensor(0);
  auto sin = getInTensor(1);
  auto cos = getInTensor(2);

  auto dx = dx_rotated;
  if (rotary_dim < dx_rotated.dim(3)) {
    dx = dx.slice(0, rotary_dim, 3);
  }

  dx = rotateGradTensor(graph(), dx, sin, cos, prog);

  if (rotary_dim < dx_rotated.dim(3)) {
    dx = poplar::concat(
        {dx, dx_rotated.slice(rotary_dim, dx_rotated.dim(3), 3)}, 3);
  }

  // Copy back to original layout
  auto result = graph().clone(dx_rotated);
  prog.add(poplar::program::Copy(dx, result));
  setOutTensor(0, result);
}

/////////////////////////////////////////////////////////////

namespace {
popx::OpxCreator<RotaryPosEmbedOpx> RotaryPosEmbedOpxCreator(RotaryPosEmbed);
popx::OpxCreator<RotaryPosEmbedGradOpx>
    RotaryPosEmbedGradOpxCreator(RotaryPosEmbedGrad);
} // namespace

} // namespace popx
} // namespace popart
