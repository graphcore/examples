// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/error.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include "bssoftmax.hpp"
#include "bssoftmaxx.hpp"

#include "popops/ElementWise.hpp"
#include "popops/ElementWiseUtil.hpp"

namespace popart {
namespace popx {

// Forward pass
BsSoftmaxOpx::BsSoftmaxOpx(Op *op, Devicex *devicex)  : popart::popx::Opx(op, devicex){
    verifyOp<BsSoftmaxOp>(op, CustomOperators::BsSoftmax);
} 

void BsSoftmaxOpx::grow(poplar::program::Sequence &prog) const {
  popart::logging::debug("Growing BsSoftmax");

  BsSoftmaxOp* op = dynamic_cast<BsSoftmaxOp *> (op_p);

  poplar::Tensor logits = getInTensor(0);
  assert(logits.rank() == 2);
  
  popart::logging::debug("BsSoftmaxOpx: Block size: {}, {}", op->getBlockSize()[0], op->getBlockSize()[1]);
  popart::logging::debug("Matrix dims: {}", op->getMatrixDims());
  popart::logging::debug("Group sizes: {}", op->getGroupSizes());

  std::size_t denseRank = op->getMatrixDims().size();
  assert(denseRank >= 2);
  const auto &denseShape = op->getMatrixDims();
  std::array<int, 2> dims{static_cast<int>(denseShape[denseRank - 2]), static_cast<int>(denseShape[denseRank - 1])};

  std::string debugStr = op->getDebugStr();
  debugStr = debugStr + "_" + std::to_string(dims[0]) + "x" + std::to_string(dims[1]);

  const auto &blockSize = op->getBlockSize();
  assert(blockSize[0] * blockSize[1] == logits.shape()[1]);

  if (denseRank == 2) {
    // If tensor is 2D there is no need for slicing, just wrap the popsparse API
    SubBlockMask subblockMask = op->getSubBlockMaskTypePerGroup()[0];
    poplar::Tensor probs =
          bsSoftmax(graph(), logits,
          dims,
          blockSize,
          op->getSparsity(),
          subblockMask,
          1,
          prog,
          debugStr);
    setOutTensor(0, probs);
  } else {
    std::size_t numGroupDims = denseRank - 2;
    uint32_t numGroupElems = 1;
    for (std::size_t i = 0; i < numGroupDims; i++) {
      numGroupElems *= denseShape[i];
    }

    uint32_t innerGroupSize = op->getInnerGroupSize();
    if (innerGroupSize == 0) {
      // By default all op is executed in 1 step
      innerGroupSize = numGroupElems;
    }
    dims[0] *= innerGroupSize;

    if (innerGroupSize > 1) {
      debugStr = debugStr + "(" + std::to_string(innerGroupSize) + ")";
    }

    if (numGroupElems % innerGroupSize != 0) {
      throw error("number of group elements: " + std::to_string(numGroupElems) + 
      "is not divisible by the size of inner group" + std::to_string(innerGroupSize));
    }
    uint32_t numInnerGroups = numGroupElems / innerGroupSize;

    std::array<int, 2> blocks{dims[0] / blockSize[0], dims[1] / blockSize[1]};

    uint32_t numBlocks = static_cast<uint32_t>(blocks[0] * blocks[1]);
    uint32_t bytesToCopy = numBlocks;

    std::vector<poplar::Tensor> out2dv;
    for (std::size_t idxIg = 0, sliceStart = 0; idxIg < numInnerGroups; ++idxIg) {
      std::vector<unsigned char> sparsityMask;
      assert(op->getSparsity().size() >= (idxIg + 1) * bytesToCopy);
      std::copy_n(op->getSparsity().begin() + idxIg * bytesToCopy,
                  bytesToCopy,
                  std::back_inserter(sparsityMask));
      uint32_t nzBlocksPerInnerGroup = std::accumulate(sparsityMask.begin(), sparsityMask.end(), 0);
    
      std::size_t sliceEnd = sliceStart + nzBlocksPerInnerGroup;
      poplar::Tensor logitsSlice = logits.slice(sliceStart, sliceEnd);
      sliceStart = sliceEnd;

      // If using inner groups, sub-block mask must be the same for all elements in each inner group
      SubBlockMask subblockMask = op->getSubBlockMaskTypePerGroup()[idxIg * innerGroupSize];

      poplar::Tensor out =
                bsSoftmax(graph(), logitsSlice,
                dims,
                blockSize,
                sparsityMask,
                subblockMask,
                1,
                prog,
                debugStr + "[" + std::to_string(idxIg) + "]");
      assert(out.shape() == logitsSlice.shape());
      out2dv.push_back(out);
    }
    poplar::Tensor probs = poplar::concat(out2dv);
    setOutTensor(0, probs);
  }
}

// Backward pass
BsSoftmaxGradOpx::BsSoftmaxGradOpx(Op *op, Devicex *devicex)  : popart::popx::Opx(op, devicex) {
  verifyOp<BsSoftmaxGradOp>(op, CustomGradOperators::BsSoftmaxGrad);
}
void BsSoftmaxGradOpx::grow(poplar::program::Sequence &prog) const {
  popart::logging::debug("Growing BsSoftmaxGrad");
  BsSoftmaxGradOp* op = dynamic_cast<BsSoftmaxGradOp *> (op_p);

  // // Computes the gradient of the loss w.r.t. the probabilities (g in above description)
  poplar::Tensor upstreamGrad = getInTensor(0);
  poplar::Tensor probs = getInTensor(1);
 
  assert(upstreamGrad.shape() == probs.shape());
  assert(probs.rank() == 2);

  popart::logging::debug("BsSoftmaxGradOpx: Block size: {}, {}", op->getBlockSize()[0], op->getBlockSize()[1]);
  popart::logging::debug("Matrix size: {}", op->getMatrixDims());

  std::size_t denseRank = op->getMatrixDims().size();
  assert(denseRank >= 2);
  const auto &denseShape = op->getMatrixDims();
  std::array<int, 2> dims{static_cast<int>(denseShape[denseRank - 2]), static_cast<int>(denseShape[denseRank - 1])};

  std::string debugStr = op->getDebugStr();
  debugStr = debugStr + "_" + std::to_string(dims[0]) + "x" + std::to_string(dims[1]);

  const auto &blockSize = op->getBlockSize();
  assert(blockSize[0] * blockSize[1] == upstreamGrad.shape()[1]);

  if (op->getMatrixDims().size() == 2){
      poplar::Tensor dlogits =
              bsSoftmaxGrad(graph(),
              probs, upstreamGrad,
              dims,
              blockSize,
              op->getSparsity(),
              prog,
              debugStr);

    setOutTensor(0, dlogits);
  } else {
    std::size_t numGroupDims = denseRank - 2;
    uint32_t numGroupElems = 1;
    for (std::size_t i = 0; i < numGroupDims; i++) {
      numGroupElems *= denseShape[i];
    }

    uint32_t innerGroupSize = op->getInnerGroupSize();
    if (innerGroupSize == 0) {
      // By default all op is executed in 1 step
      innerGroupSize = numGroupElems;
    }
    dims[0] *= innerGroupSize;

    if (innerGroupSize > 1) {
      debugStr = debugStr + "(" + std::to_string(innerGroupSize) + ")";
    }

    if (numGroupElems % innerGroupSize != 0) {
      throw error("number of group elements: " + std::to_string(numGroupElems) + 
      "is not divisible by the size of inner group" + std::to_string(innerGroupSize));
    }
    uint32_t numInnerGroups = numGroupElems / innerGroupSize;

    std::array<int, 2> blocks{dims[0] / blockSize[0], dims[1] / blockSize[1]};

    uint32_t numBlocks = static_cast<uint32_t>(blocks[0] * blocks[1]);
    uint32_t bytesToCopy = numBlocks;

    std::vector<poplar::Tensor> out2dv;
    for (std::size_t idxIg = 0, sliceStart = 0; idxIg < numInnerGroups; ++idxIg) {
      std::vector<unsigned char> sparsityMask;
      assert(op->getSparsity().size() >= (idxIg + 1) * bytesToCopy);
      std::copy_n(op->getSparsity().begin() + idxIg * bytesToCopy,
                  bytesToCopy,
                  std::back_inserter(sparsityMask));
      uint32_t nzBlocksPerInnerGroup = std::accumulate(sparsityMask.begin(), sparsityMask.end(), 0);
    
      std::size_t sliceEnd = sliceStart + nzBlocksPerInnerGroup;
      poplar::Tensor probsSlice = probs.slice(sliceStart, sliceEnd);
      poplar::Tensor upstreamGradSlice = upstreamGrad.slice(sliceStart, sliceEnd);
      sliceStart = sliceEnd;

      poplar::Tensor out =
                  bsSoftmaxGrad(graph(),
                  probsSlice, upstreamGradSlice,
                  dims,
                  blockSize,
                  sparsityMask,
                  prog,
                  debugStr + "[" + std::to_string(idxIg) + "]");
      assert(out.shape() == probsSlice.shape());
      out2dv.push_back(out);
    }
    poplar::Tensor dlogits = poplar::concat(out2dv);
    setOutTensor(0, dlogits);
  }
}

namespace {
OpxCreator<BsSoftmaxOpx> BsSoftmaxOpxCreator(CustomOperators::BsSoftmax);
OpxCreator<BsSoftmaxGradOpx> BsSoftmaxGradOpxCreator(CustomGradOperators::BsSoftmaxGrad);
} // namespace

} // namespace popx
} // namespace popart
