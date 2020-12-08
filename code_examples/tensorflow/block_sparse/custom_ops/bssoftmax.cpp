// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popsparse/experimental/BlockSparse.hpp>
#include <poplar/Type.hpp>
#include <poplar/Graph.hpp>
#include <poputil/exceptions.hpp>
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include <array>

#include "bsutils.hpp"

extern "C" {

static auto logger = createLogger();

static poplar::Tensor BsSoftmax(
  poplar::Graph& graph, poplar::Tensor logits, poplar::program::Sequence& prog,
  const std::vector<int>& dimDense,
  const std::array<int, 2>& blockSize,
  const std::vector<unsigned char>& sparsityMask,
  std::vector<popsparse::experimental::SubBlockMask> subBlockMaskType,
  uint32_t innerGroupSize,
  bool inPlace,
  const std::string& debugPrefix) {

  logger->trace((debugPrefix + " BsSoftmax() {} entry.").c_str(), inPlace ? "in place" : "");

  if (logits.rank() != 2) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax: input tensor must have rank 2.");
  }
  std::size_t denseRank = dimDense.size();
  assert(denseRank >= 2);
  std::size_t numGroupDims = denseRank - 2;

  std::array<int, 2> dim{dimDense[denseRank - 2], dimDense[denseRank - 1]};

  std::array<uint32_t, 2> numBlocks;
  for (int i = 0; i < 2; ++i) {
    if (dim[i] % blockSize[i] != 0) {
      throw poputil::poplibs_error(debugPrefix + " bs softmax: for index " + std::to_string(i) +
                                  " dimension: " + std::to_string(dim[i]) +
                                  " is not divisible by block size: " + std::to_string(blockSize[i]));
    }
    numBlocks[i] = dim[i] / blockSize[i];
  }
  uint32_t numBlocks2d = numBlocks[0] * numBlocks[1];

  uint32_t numGroupElems = 1;
  for (std::size_t i = 0; i < numGroupDims; i++) {
    numGroupElems *= dimDense[i];
  }
  if (sparsityMask.size() != numBlocks2d * numGroupElems) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax: sparsity mask size: " + std::to_string(sparsityMask.size()) +
    " is different from the total number of blocks: " + std::to_string(numBlocks2d * numGroupElems));
  }
  innerGroupSize = innerGroupSize == 0 ? numGroupElems : innerGroupSize;
  if (numGroupElems % innerGroupSize != 0) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax: number of group elements: " + std::to_string(numGroupElems) +
    " is not divisible by the size of inner group: " + std::to_string(innerGroupSize));
  }
  uint32_t numInnerGroups = numGroupElems / innerGroupSize;
  if (subBlockMaskType.size() == 1 && numGroupElems > 1) {
    subBlockMaskType = std::vector<popsparse::experimental::SubBlockMask>(numGroupElems, subBlockMaskType[0]);
  }
  if (subBlockMaskType.size() != numGroupElems) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax: subblock mask size: " + std::to_string(subBlockMaskType.size()) +
    " is different from the number of groups: " + std::to_string(numGroupElems));
  }

  std::string debugStr = debugPrefix + "_" + std::to_string(dim[0]) + "x" + std::to_string(dim[1]);
  if (innerGroupSize > 1) {
    debugStr = debugStr + "ig[" + std::to_string(innerGroupSize) + "]";
  }

  poplar::Tensor out;
  if (denseRank == 2) {
    // If tensor is 2D there is no need for slicing, just wrap the popsparse API
    if (!inPlace) {
      out =
        popsparse::experimental::bsSoftmax(
          graph, logits,
          dim,
          blockSize,
          sparsityMask,
          subBlockMaskType[0],
          innerGroupSize,
          prog,
          debugStr);
    } else {
      popsparse::experimental::bsSoftmaxInPlace(
        graph, logits,
        dim,
        blockSize,
        sparsityMask,
        subBlockMaskType[0],
        innerGroupSize,
        prog,
        debugStr);
      out = logits;
    }
  } else {   
    uint32_t bytesToCopy = numBlocks2d * innerGroupSize;
    std::vector<poplar::Tensor> out2dv;
    auto sparsityMaskIter = sparsityMask.begin();
    for (std::size_t idxIg = 0, idxG = 0, sliceStart = 0; idxIg < numInnerGroups; ++idxIg, idxG += innerGroupSize) {
      std::vector<unsigned char> sparsityMaskSlice;
      std::copy_n(sparsityMaskIter,
                  bytesToCopy,
                  std::back_inserter(sparsityMaskSlice));
      sparsityMaskIter += bytesToCopy;

      uint32_t nzBlocksPerInnerGroup = std::accumulate(sparsityMaskSlice.begin(), sparsityMaskSlice.end(), 0);

      std::size_t sliceEnd = sliceStart + nzBlocksPerInnerGroup;
      poplar::Tensor logitsSlice = logits.slice(sliceStart, sliceEnd);
      sliceStart = sliceEnd;

      poplar::Tensor outSlice;
      if (!inPlace) {
        outSlice =
          popsparse::experimental::bsSoftmax(
            graph, logitsSlice,
            dim,
            blockSize,
            sparsityMaskSlice,
            subBlockMaskType[idxG],
            innerGroupSize,
            prog,
            debugStr + "[" + std::to_string(idxIg) + "]");
      } else {
        popsparse::experimental::bsSoftmaxInPlace(
          graph, logitsSlice,
          dim,
          blockSize,
          sparsityMaskSlice,
          subBlockMaskType[idxG],
          innerGroupSize,
          prog,
          debugStr + "[" + std::to_string(idxIg) + "]");
        outSlice = logitsSlice;
      }
      assert(outSlice.shape() == logitsSlice.shape());
      out2dv.push_back(outSlice);
    }
    out = poplar::concat(out2dv);
  }

  logger->trace((debugPrefix + " BsSoftmax() exit.").c_str());
  return out;
}

static poplar::program::Program InternalBuild(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  bool inPlace,
  const std::string& debugPrefix) {

  if (inputs.size() != 1) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax: 1 input required.");
  }

  BsSoftmaxArgs args = parseBsSoftmaxJsonArgs(attributes);

  const poplar::Tensor& logits = inputs[0];

  poplar::program::Sequence prog;
  auto out = BsSoftmax(graph, logits, prog,
                      args.dimDense,
                      args.blockSize,
                      args.sparsityMask,
                      args.subBlockMaskType,
                      args.innerGroupSize,
                      inPlace,
                      debugPrefix);

  outputs.push_back(out);
  return prog;
}

void BuildDSoftmax_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::uint32_t& num_inplace,
  bool& is_elementwise,
  bool& is_stateless,
  std::uint32_t num_inputs) {

  logger->trace("BuildDSoftmax_metadata()");
  allocating_indices.clear();
  num_inplace = 0;
  is_elementwise = true;
  is_stateless = true;
}

poplar::program::Program BuildSoftmax(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  logger->trace((debugPrefix + " BuildSoftmax() entry").c_str());

  auto out = InternalBuild(graph, inputs, outputs, attributes, false, debugPrefix);

  logger->trace((debugPrefix + " BuildSoftmax() exit").c_str());
  return out;
}

void BuildDSoftmaxInPlace_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::uint32_t& num_inplace,
  bool& is_elementwise,
  bool& is_stateless,
  std::uint32_t num_inputs) {

  logger->trace("BuildDSoftmaxInPlace_metadata()");
  allocating_indices.clear();
  num_inplace = 1;
  is_elementwise = true;
  is_stateless = true;
}

poplar::program::Program BuildSoftmaxInPlace(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  logger->trace((debugPrefix + " BuildSoftmaxInPlace() entry").c_str());

  auto out = InternalBuild(graph, inputs, outputs, attributes, true, debugPrefix);

  logger->trace((debugPrefix + " BuildSoftmaxInPlace() exit").c_str());
  return out;
}

static poplar::Tensor BsSoftmaxGrad(
  poplar::Graph& graph, poplar::Tensor dProbs, poplar::Tensor probs,
  poplar::program::Sequence& prog,
  const std::vector<int>& dimDense,
  const std::array<int, 2>& blockSize,
  const std::vector<unsigned char>& sparsityMask,
  uint32_t innerGroupSize,
  const std::string& debugPrefix) {

  logger->trace((debugPrefix + " BsSoftmaxGrad() entry.").c_str());

  if (dProbs.rank() != 2) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax gradients: input tensors must have rank 2 or more.");
  }
  if (dProbs.shape() != probs.shape()) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax gradients: input tensors must have the same shape.");
  }
  std::size_t denseRank = dimDense.size();
  assert(denseRank >= 2);
  std::size_t numGroupDims = denseRank - 2;

  std::array<int, 2> dim{dimDense[denseRank - 2], dimDense[denseRank - 1]};

  std::array<uint32_t, 2> numBlocks;
  for (int i = 0; i < 2; ++i) {
    if (dim[i] % blockSize[i] != 0) {
      throw poputil::poplibs_error(debugPrefix + " bs softmax gradients: for index " + std::to_string(i) +
                                  " dimension: " + std::to_string(dim[i]) +
                                  " is not divisible by block size: " + std::to_string(blockSize[i]));
    }
    numBlocks[i] = dim[i] / blockSize[i];
  }
  uint32_t numBlocks2d = numBlocks[0] * numBlocks[1];

  uint32_t numGroupElems = 1;
  for (std::size_t i = 0; i < numGroupDims; i++) {
    numGroupElems *= dimDense[i];
  }
  if (sparsityMask.size() != numBlocks2d * numGroupElems) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax gradients: sparsity mask size: " + std::to_string(sparsityMask.size()) +
    " is different from the total number of blocks: " + std::to_string(numBlocks2d * numGroupElems));
  }
  innerGroupSize = innerGroupSize == 0 ? numGroupElems : innerGroupSize;
  if (numGroupElems % innerGroupSize != 0) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax gradients: number of group elements: " + std::to_string(numGroupElems) +
    " is not divisible by the size of inner group: " + std::to_string(innerGroupSize));
  }
  uint32_t numInnerGroups = numGroupElems / innerGroupSize;

  std::string debugStr = debugPrefix + "_" + std::to_string(dim[0]) + "x" + std::to_string(dim[1]);
  if (innerGroupSize > 1) {
    debugStr = debugStr + "ig[" + std::to_string(innerGroupSize) + "]";
  }

  dim[0] *= innerGroupSize;

  poplar::Tensor out;
  if (denseRank == 2) {
    // If tensor is 2D there is no need for slicing, just wrap the popsparse API
    out = popsparse::experimental::bsSoftmaxGrad(
      graph, probs, dProbs,
      dim,
      blockSize,
      sparsityMask,
      prog,
      debugStr);
  } else {   
    uint32_t bytesToCopy = numBlocks2d * innerGroupSize;
    std::vector<poplar::Tensor> out2dv;
    auto sparsityMaskIter = sparsityMask.begin();
    for (std::size_t idxIg = 0, sliceStart = 0; idxIg < numInnerGroups; ++idxIg) {
      std::vector<unsigned char> sparsityMaskSlice;
      std::copy_n(sparsityMaskIter,
                  bytesToCopy,
                  std::back_inserter(sparsityMaskSlice));
      sparsityMaskIter += bytesToCopy;

      uint32_t nzBlocksPerInnerGroup = std::accumulate(sparsityMaskSlice.begin(), sparsityMaskSlice.end(), 0);

      std::size_t sliceEnd = sliceStart + nzBlocksPerInnerGroup;
      poplar::Tensor probsSlice = probs.slice(sliceStart, sliceEnd);
      poplar::Tensor dProbsSlice = dProbs.slice(sliceStart, sliceEnd);
      sliceStart = sliceEnd;

      poplar::Tensor outSlice =
        popsparse::experimental::bsSoftmaxGrad(
          graph, probsSlice, dProbsSlice,
          dim,
          blockSize,
          sparsityMaskSlice,
          prog,
          debugStr + "[" + std::to_string(idxIg) + "]");
      assert(outSlice.shape() == probsSlice.shape());
      out2dv.push_back(outSlice);
    }
    out = poplar::concat(out2dv);
  }

  logger->trace((debugPrefix + " BsSoftmaxGrad() exit.").c_str());
  return out;
}

static poplar::program::Program InternalBuild_grad(
  poplar::Graph& graph, int inputGradIndex,
  const std::vector<poplar::Tensor>& gradients,
  const std::vector<poplar::Tensor>& fwdInputs,
  const std::vector<poplar::Tensor>& fwdOutputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  if (gradients.size() != 1) {
    throw poputil::poplibs_error(debugPrefix + " bs softmax grad: 1 gradient required.");
  }
  if (fwdInputs.size() != 1) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul grad: 1 forward inputs required.");
  }
  if (fwdOutputs.size() != 1) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul grad: 1 forward output required.");
  }

  BsSoftmaxArgs args = parseBsSoftmaxJsonArgs(attributes);

  poplar::Tensor dProbs = gradients[0];
  poplar::Tensor probs = fwdOutputs[0];

  poplar::program::Sequence prog;
  auto out = BsSoftmaxGrad(graph, dProbs, probs, prog,
                           args.dimDense,
                           args.blockSize,
                           args.sparsityMask,
                           args.innerGroupSize,
                           debugPrefix);

  outputs.push_back(out);

  return prog;
}

void BuildDSoftmaxGrad_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::uint32_t& num_inplace,
  bool& is_elementwise,
  bool& is_stateless,
  std::uint32_t num_inputs) {

  logger->trace("BuildDSoftmaxGrad_metadata()");
  allocating_indices.clear();
  num_inplace = 0;
  is_elementwise = true;
  is_stateless = true;
}

poplar::program::Program BuildSoftmax_grad(
  poplar::Graph& graph, int inputGradIndex,
  const std::vector<poplar::Tensor>& gradients,
  const std::vector<poplar::Tensor>& fwdInputs,
  const std::vector<poplar::Tensor>& fwdOutputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  logger->trace((debugPrefix + " BuildSoftmax_grad() entry").c_str());

  auto out = InternalBuild_grad(graph, inputGradIndex, gradients, fwdInputs, fwdOutputs, outputs, attributes, debugPrefix);

  logger->trace((debugPrefix + " BuildSoftmax_grad() exit").c_str());
  return out;
}

poplar::program::Program BuildSoftmaxInPlace_grad(
  poplar::Graph& graph, int inputGradIndex,
  const std::vector<poplar::Tensor>& gradients,
  const std::vector<poplar::Tensor>& fwdInputs,
  const std::vector<poplar::Tensor>& fwdOutputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  logger->trace((debugPrefix + " BuildSoftmaxInPlace_grad() entry").c_str());

  auto out = InternalBuild_grad(graph, inputGradIndex, gradients, fwdInputs, fwdOutputs, outputs, attributes, debugPrefix);

  logger->trace((debugPrefix + " BuildSoftmaxInPlace_grad() exit").c_str());
  return out;
}

}
