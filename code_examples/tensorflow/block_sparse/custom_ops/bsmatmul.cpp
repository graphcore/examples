// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popsparse/experimental/BlockSparseMatMul.hpp>
#include <poplar/Type.hpp>
#include <poplar/Graph.hpp>
#include <poputil/exceptions.hpp>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <functional>
#include <string>
#include <array>
#include <memory>
#include <mutex>
#include <stdlib.h>

#include "bsutils.hpp"

extern "C" {

/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
int32_t custom_op_api_level = 5;

static auto logger = createLogger();

// BSMatMulParams object must be live between allocator call and bsMatMul call
static std::unordered_map<std::string, std::shared_ptr<popsparse::experimental::BSMatMulParams>> bSParamsCache;
static std::mutex mtx;

static std::shared_ptr<popsparse::experimental::BSMatMulParams>
getOrCreateBsMatmul(const std::string &tag,
                          std::function<std::shared_ptr<popsparse::experimental::BSMatMulParams>()> creator,
                          bool createOnly,
                          const std::string& debugPrefix) {
  std::shared_ptr<popsparse::experimental::BSMatMulParams> spBSMatMulParams;
  bool paramFound = true;
  {
    std::lock_guard<std::mutex> lk(mtx);
    auto iter = bSParamsCache.find(tag);
    if (iter == bSParamsCache.end()) {
      auto spObject = creator();
      iter = bSParamsCache.insert(std::pair<std::string, std::shared_ptr<popsparse::experimental::BSMatMulParams>>(tag, spObject)).first;
      paramFound = false;
    } else if (createOnly) {
      auto spObject = creator();
      iter->second = std::shared_ptr<popsparse::experimental::BSMatMulParams>(spObject);
    }
    spBSMatMulParams = iter->second;
  }

  if (paramFound) {
    if (!createOnly) {
      logger->trace((debugPrefix + " BSMatMulParams() for the tag {} was found in a cache.").c_str(), tag.c_str());
    } else {
      logger->warn((debugPrefix + " BSMatMulParams() for the tag {} should not be in the cache, but it was found there. A new object is created.").c_str(), tag.c_str());
    }
  } else {
    if (createOnly) {
      logger->trace((debugPrefix + " BSMatMulParams() for the tag {} was created").c_str(), tag.c_str());
    } else {
      logger->warn((debugPrefix + " BSMatMulParams() for the tag {} was not found in cache as expected. A new object is created.").c_str(), tag.c_str());
    }
  }
  return spBSMatMulParams;
}

static poplar::Tensor BsMatMul(
  poplar::Graph& graph, poplar::Tensor lhs, poplar::Tensor rhs, poplar::program::Sequence& prog,
  const std::array<int, 3>& dim,
  const std::array<int, 3>& blockSize,
  const std::vector<unsigned char>& sparsityMask,
  bool transposedRhs,
  const poplar::Type& dataType,
  const poplar::Type& partialDataType,
  uint32_t innerGroupSize,
  std::string& partitionMethod,
  float memoryCycleRatio,
  bool sparseOut,
  bool usePreallocatedBsMatMul,
  const std::string& debugPrefix) {

  auto kind = sparseOut ? "dds" : "dsd";

  logger->trace((debugPrefix + " BsMatMul() {} entry.").c_str(), kind);

  std::array<uint32_t, 3> numBlocks;
  for (int i = 0; i < 3; ++i) {
    if (dim[i] % blockSize[i] != 0) {
      throw poputil::poplibs_error(debugPrefix + " bs matmul: for index " + std::to_string(i) +
                                  " dimension: " + std::to_string(dim[i]) +
                                  " is not divisible by block size: " + std::to_string(blockSize[i]));
    }
    numBlocks[i] = dim[i] / blockSize[i];
  }

  if (lhs.rank() < 2) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul: left hand size tensor must have rank 2 or larger.");
  }
  if (!sparseOut) {
    if (rhs.rank() != 2) {
      throw poputil::poplibs_error(debugPrefix + " bs matmul: right hand size tensor must have rank 2.");
    }
  } else {
    if (rhs.rank() != lhs.rank()) {
      throw poputil::poplibs_error(debugPrefix + " bs matmul: left hand size tensor rank " + std::to_string(lhs.rank()) +
        " is not equal to right hand size tensor rank " + std::to_string(rhs.rank()));
    }
  }
  auto lhsShape = lhs.shape();
  auto rhsShape = rhs.shape();
  poplar::Tensor out;

  uint32_t numGroupElems = 1;
  bool performGroupedMatmul = lhs.rank() > 2;
  std::vector<size_t> inputGroupDims;
  uint32_t numInnerGroups = 1;
  uint32_t endGroupIdx = lhsShape.size() - 2;
  if (performGroupedMatmul) {
    // figure out number of groups dimensions
    for (uint32_t i = 0; i < endGroupIdx; i++) {
      numGroupElems *= lhsShape[i];
      inputGroupDims.push_back(lhsShape[i]);
    }
    innerGroupSize = innerGroupSize == 0 ? numGroupElems : innerGroupSize;
    if (numGroupElems % innerGroupSize != 0) {
      throw poputil::poplibs_error(debugPrefix + " bs matmul: number of group elements: " + std::to_string(numGroupElems) +
      " is not divisible by the size of inner group: " + std::to_string(innerGroupSize));
    }
    numInnerGroups = numGroupElems / innerGroupSize;
    // Flatten all groups
    lhs = lhs.reshapePartial(0, endGroupIdx, {numGroupElems});
  } else {
    // Insert virtual empty group
    lhs = lhs.expand({0});
    innerGroupSize = 1;
  }
  assert(lhs.rank() == 3);
  std::size_t lhsRows = lhs.dim(1);
  std::size_t lhsCols = lhs.dim(2);
  std::size_t lhsRowsInnerGroup = lhsRows * innerGroupSize;

  uint32_t numBlocks2d = sparseOut ? (numBlocks[0] * numBlocks[2]) : (numBlocks[1] * numBlocks[2]);
  if (sparsityMask.size() != numBlocks2d * numGroupElems) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul: sparsity mask size: " + std::to_string(sparsityMask.size()) +
    " is different from the total number of blocks: " + std::to_string(numBlocks2d * numGroupElems));
  }
  uint32_t bytesToCopy = numBlocks2d * innerGroupSize;

  std::string debugStr = debugPrefix + '_' + kind + '_' + partitionMethod;
  if (innerGroupSize > 1) {
    debugStr = debugStr + "ig[" + std::to_string(innerGroupSize) + "]";
  }
  debugStr = debugStr + "_" + std::to_string(dim[0]) + "x" + std::to_string(dim[1]) + "x" + std::to_string(dim[2]);
  std::string debugInShapes = "(" + std::to_string(lhsShape[0]);
  for (size_t i = 1; i < lhsShape.size(); ++i) {
    debugInShapes += "," + std::to_string(lhsShape[i]);
  }
  debugInShapes += ")x(" + std::to_string(rhsShape[0]);
  for (size_t i = 1; i < rhsShape.size(); ++i) {
    debugInShapes += "," + std::to_string(rhsShape[i]);
  }
  debugInShapes += ")";
  logger->debug((debugStr + debugInShapes + " numGroupElems = {}").c_str(), numGroupElems);

  poplar::OptionFlags options = {
    {"memoryCycleRatio", std::to_string(memoryCycleRatio)},
    {"partitionMethod", partitionMethod}
  };

  if (!sparseOut) {
    std::string tag0 = std::to_string(reinterpret_cast<std::size_t>(&graph)) + "_" + debugPrefix;
    std::vector<poplar::Tensor> out3dVec;
    std::size_t idxElem = 0;
    std::size_t rhsSliceStart = 0;
    auto sparsityMaskIter = sparsityMask.begin();
    for (std::size_t idxIg = 0; idxIg < numInnerGroups; ++idxIg) {
      std::size_t idxNextElem = idxElem + innerGroupSize;

      std::vector<unsigned char> sparsityMaskSlice;
      std::copy_n(sparsityMaskIter,
                  bytesToCopy,
                  std::back_inserter(sparsityMaskSlice));
      sparsityMaskIter += bytesToCopy;

      std::function<std::shared_ptr<popsparse::experimental::BSMatMulParams>()> creator =
        [&]() {
          return std::make_shared<popsparse::experimental::BSMatMulParams>(
            dim,
            blockSize,
            sparsityMaskSlice,
            transposedRhs,
            dataType,
            dataType,
            partialDataType,
            innerGroupSize
          );
        };

      std::shared_ptr<popsparse::experimental::BSMatMulParams> spBsMatMulObj;
      if (usePreallocatedBsMatMul) {
        std::string tag = tag0;
        if (numInnerGroups > 1) {
          tag = tag + "[" + std::to_string(idxIg) + "]";
        }
        spBsMatMulObj = getOrCreateBsMatmul(tag, creator, false, debugPrefix);
      } else {
        spBsMatMulObj = creator();
      }

      auto& bsMatMulObj = *spBsMatMulObj;

      poplar::Tensor lhsSlice = lhs.slice(idxElem, idxNextElem);
      lhsSlice = lhsSlice.reshape({lhsRowsInnerGroup, lhsCols});

      std::size_t rhsSparseGroupLen = std::accumulate(sparsityMaskSlice.begin(), sparsityMaskSlice.end(), 0);
      std::size_t rhsSliceEnd = rhsSliceStart + rhsSparseGroupLen;
      poplar::Tensor rhsSlice = rhs.slice(rhsSliceStart, rhsSliceEnd);

      poplar::Tensor outSlice;
      try {
        outSlice = popsparse::experimental::bsMatMul(graph,
                                                     bsMatMulObj,
                                                     prog,
                                                     lhsSlice,
                                                     rhsSlice,
                                                     options,
                                                     debugStr);
      } catch (const poputil::poplibs_error &e) {
        logger->error((debugPrefix + " bsMatMul() failed with the message: '{}'").c_str(), e.what());
        throw;
      }

      logger->trace((debugStr + " out[{}] shape = {} x {}").c_str(), idxIg, outSlice.dim(0), outSlice.dim(1));

      std::size_t outRowsInnerGroup = outSlice.dim(0);
      assert(outRowsInnerGroup % innerGroupSize == 0);
      std::size_t outRows = outRowsInnerGroup / innerGroupSize;
      for (uint32_t r = 0; r < outRowsInnerGroup; r += outRows) {
        out3dVec.push_back(outSlice.slice(r, r + outRows).expand({0}));
      }

      idxElem = idxNextElem;
      rhsSliceStart = rhsSliceEnd;
    }

    out = poplar::concat(out3dVec, 0);
    std::vector<std::size_t> expectedOutShape = {numGroupElems,
                                                static_cast<std::size_t>(dim[0]),
                                                static_cast<std::size_t>(dim[2])};
    if (out.shape() != expectedOutShape) {
      throw poputil::poplibs_error(debugPrefix + " bs matmul: out shape and expectedOutShape do not match");
    }
    // need to reshape the output
    if (performGroupedMatmul) {
      std::vector<size_t> newOutShape = inputGroupDims;
      newOutShape.push_back(dim[0]);
      newOutShape.push_back(dim[2]);

      out = out.reshape(newOutShape);
    } else {
      // remove singleton dimension
      out = out.squeeze({0});
    }
  } else {
    if (performGroupedMatmul) {
      for (uint32_t i = 0; i < endGroupIdx; i++) {
        if (lhsShape[i] != rhsShape[i]) {
          throw poputil::poplibs_error(debugPrefix + " bs matmul: left hand size tensor and right hand size tensor must have the same dimensions except the last two.");
        }
      }
      rhs = rhs.reshapePartial(0, endGroupIdx, {numGroupElems});
    } else {
      rhs = rhs.expand({0});
    }
    assert(rhs.rank() == 3);
    std::size_t rhsRows = rhs.dim(1);
    std::size_t rhsCols = rhs.dim(2);
    std::size_t rhsRowsInnerGroup = rhsRows * innerGroupSize;

    std::vector<poplar::Tensor> out2dVec;
    std::size_t idxElem = 0;
    auto sparsityMaskIter = sparsityMask.begin();
    for (std::size_t idxIg = 0; idxIg < numInnerGroups; ++idxIg) {
      std::size_t idxNextElem = idxElem + innerGroupSize;

      std::vector<unsigned char> sparsityMaskSlice;
      std::copy_n(sparsityMaskIter,
                  bytesToCopy,
                  std::back_inserter(sparsityMaskSlice));

      popsparse::experimental::BSMatMulParams bsMatMulObj(
            dim,
            blockSize,
            sparsityMaskSlice,
            //transposedRhs,
            dataType,
            dataType,
            partialDataType,
            popsparse::experimental::SubBlockMask::None,
            innerGroupSize
          );

      poplar::Tensor lhsSlice = lhs.slice(idxElem, idxNextElem);
      lhsSlice = lhsSlice.reshape({lhsRowsInnerGroup, lhsCols});

      poplar::Tensor rhsSlice = rhs.slice(idxElem, idxNextElem);
      rhsSlice = rhsSlice.reshape({rhsRowsInnerGroup, rhsCols});

      poplar::Tensor outSlice;
      try {
        outSlice = popsparse::experimental::bsMatMul(graph,
                                                     bsMatMulObj,
                                                     prog,
                                                     lhsSlice,
                                                     rhsSlice,
                                                     options,
                                                     debugStr);
      } catch (const poputil::poplibs_error &e) {
        logger->error((debugPrefix + " bsMatMul() failed with the message: '{}'").c_str(), e.what());
        throw;
      }

      logger->trace((debugStr + " out[{}] shape = {} x {}").c_str(), idxIg, outSlice.dim(0), outSlice.dim(1));

      out2dVec.push_back(outSlice);

      idxElem = idxNextElem;
      sparsityMaskIter += bytesToCopy;
    }
    out = poplar::concat(out2dVec, 0);
    size_t totalNumNonZeroBlocks = std::accumulate(
                                    sparsityMask.begin(),
                                    sparsityMask.end(),
                                    0);
    std::vector<std::size_t> expectedOutShape = {totalNumNonZeroBlocks,
                                                static_cast<std::size_t>(blockSize[0] * blockSize[2])};
    if (out.shape() != expectedOutShape) {
      throw poputil::poplibs_error(debugPrefix + " bs matmul: out shape and expectedOutShape do not match");
    }
  }
  std::string debugOutShape = "(" + std::to_string(out.shape()[0]);
  for (size_t i = 1; i < out.shape().size(); ++i) {
    debugOutShape += "," + std::to_string(out.shape()[i]);
  }
  debugOutShape += ")";
  logger->debug((debugStr + " out shape = " + debugOutShape).c_str());
  logger->trace((debugPrefix + " BsMatMul() {} exit.").c_str(), kind);
  return out;
}

static poplar::program::Program InternalBuild(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  bool sparseOut,
  const std::string& debugPrefix) {

  if (inputs.size() != 2) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul: 2 inputs required.");
  }

  BsMatMulArgs args = parseBsMatMulJsonArgs(attributes);

  const poplar::Tensor& lhs = inputs[0];
  const poplar::Tensor& rhs = inputs[1];

  poplar::program::Sequence prog;
  auto out = BsMatMul(graph, lhs, rhs, prog,
                      args.dim,
                      args.blockSize,
                      args.sparsityMask,
                      args.transposedRhs,
                      args.dataType,
                      args.partialDataType,
                      args.innerGroupSize,
                      args.partitionMethod,
                      args.memoryCycleRatio,
                      sparseOut, !sparseOut,
                      debugPrefix);

  outputs.push_back(out);
  return prog;
}

poplar::program::Program BuildDSD(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  logger->trace((debugPrefix + " BuildDSD() entry").c_str());

  auto out = InternalBuild(graph, inputs, outputs, attributes, false, debugPrefix);

  logger->trace((debugPrefix + " BuildDSD() exit").c_str());
  return out;
}

poplar::program::Program BuildDDS(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  logger->trace((debugPrefix + " BuildDDS() entry").c_str());

  auto out = InternalBuild(graph, inputs, outputs, attributes, true, debugPrefix);

  logger->trace((debugPrefix + " BuildDDS() exit").c_str());
  return out;
}

/// Meta data function sets properties of the forward op.
void BuildDSD_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {

  logger->trace("BuildDSD_metadata()");
  // We only create RHS matrix now
  allocating_indices = {1};
  is_elementwise = false;
  is_stateless = true;
}

void BuildDDS_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::vector<std::int64_t>& replica_identical_output_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {

  logger->trace("BuildDDS_metadata()");
  allocating_indices.clear();
  is_elementwise = false;
  is_stateless = true;
}

poplar::Tensor BuildDSD_allocator(
  poplar::Graph& graph, std::uint32_t operand,
  const std::vector<size_t>& shape, poplar::Type type,
  const std::string& attributes,
  const std::string& debugPrefix) {

  BsMatMulArgs args = parseBsMatMulJsonArgs(attributes);

  if (operand != 1) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul allocator: Unexpected operand index: " + std::to_string(operand));
  }
  if (shape.size() != 2) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul allocator: unexpected shape of rank = " +
                                 std::to_string(shape.size()) + " provided. The rank of shape should be 2.");
  }

  logger->trace((debugPrefix + "BuildDSD_allocator() entry").c_str());
  logger->debug((debugPrefix + " dsd allocator called for shape: {} x {} and dims: {} x {} x {}").c_str(),
                shape[0], shape[1], args.dim[0], args.dim[1], args.dim[2]);

  if (shape[1] != args.blockSize[1] * args.blockSize[2]) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul allocator: unexpected shape[1]: " +
                                 std::to_string(shape[1]) + " provided. The 2nd dimension must be a product of"
                                 " block sizes 1 and 2: " + std::to_string(args.blockSize[1]) + " * " + std::to_string(args.blockSize[2]));
  }
  std::array<uint32_t, 3> numBlocks;
  for (int i = 0; i < 3; ++i) {
    if (args.dim[i] % args.blockSize[i] != 0) {
      throw poputil::poplibs_error(debugPrefix + " bs matmul allocator: for index " + std::to_string(i) +
                                  " dimension: " + std::to_string(args.dim[i]) +
                                  " is not divisible by block size: " + std::to_string(args.blockSize[i]));
    }
    numBlocks[i] = args.dim[i] / args.blockSize[i];
  }
  uint32_t numBlocks2d = numBlocks[1] * numBlocks[2];
  if (args.sparsityMask.size() % numBlocks2d) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul allocator: the length of sparsity mask: " +
                                std::to_string(args.sparsityMask.size()) +
                                 " must be divisible by a product of "
                                 " number of block 1 and 2: " + std::to_string(numBlocks[1]) + " * " + std::to_string(numBlocks[2]));
  }

  size_t totalNumNonZeroBlocks = std::accumulate(
                                    args.sparsityMask.begin(),
                                    args.sparsityMask.end(),
                                    0);

  if (shape[0] != totalNumNonZeroBlocks) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul allocator: unexpected shape[0]: " +
                                 std::to_string(shape[0]) + " provided. The 1st dimension must be equal to "
                                 "the total number of non-zero blocks: " + std::to_string(totalNumNonZeroBlocks));
  }

  uint32_t numGroupElems = args.sparsityMask.size() / numBlocks2d;
  if (numGroupElems % args.innerGroupSize != 0) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul allocator: number of group elements: " + std::to_string(numGroupElems) +
    " is not divisible by the size of inner group: " + std::to_string(args.innerGroupSize));
  }
  uint32_t numInnerGroups = numGroupElems / args.innerGroupSize;
  uint32_t bytesToCopy = numBlocks2d * args.innerGroupSize;

  poplar::OptionFlags options = {
    {"memoryCycleRatio", std::to_string(args.memoryCycleRatio)},
    {"partitionMethod", args.partitionMethod}
  };

  // To get the debug prefix for the op, remove operand number suffix
  std::string debugPrefixForOp = debugPrefix;
  auto posOperandSep = debugPrefixForOp.rfind(":");
  if (posOperandSep != std::string::npos) {
    debugPrefixForOp = debugPrefixForOp.substr(0, posOperandSep);
  }
  std::string tag0 = std::to_string(reinterpret_cast<std::size_t>(&graph)) + "_" + debugPrefixForOp;

  std::vector<poplar::Tensor> rhsVec;
  auto sparsityMaskIter = args.sparsityMask.begin();
  for (std::size_t idxIg = 0; idxIg < numInnerGroups; ++idxIg) {
    std::vector<unsigned char> sparsityMaskSlice;
    std::copy_n(sparsityMaskIter,
                bytesToCopy,
                std::back_inserter(sparsityMaskSlice));

    std::function<std::shared_ptr<popsparse::experimental::BSMatMulParams>()> creator =
          [&]() {
            return std::make_shared<popsparse::experimental::BSMatMulParams>(
              args.dim,
              args.blockSize,
              sparsityMaskSlice,
              args.transposedRhs,
              args.dataType,
              args.dataType,
              args.partialDataType,
              args.innerGroupSize
            );
          };

    std::string tag = tag0;
    if (numInnerGroups > 1) {
      tag = tag + "[" + std::to_string(idxIg) + "]";
    }
    std::shared_ptr<popsparse::experimental::BSMatMulParams> spBsMatMulObj =
          getOrCreateBsMatmul(tag, creator, true, debugPrefix);

    auto& bsMatMulObj = *spBsMatMulObj;
    poplar::Tensor rhsSlice = popsparse::experimental::createBSMatMulInputRHS(graph, bsMatMulObj, debugPrefix, options);
    assert(rhsSlice.rank() == 2);
    logger->trace((debugPrefix + " rhs[{}] shape = {} x {}").c_str(), idxIg, rhsSlice.dim(0), rhsSlice.dim(1));
    rhsVec.push_back(rhsSlice);

    sparsityMaskIter += bytesToCopy;
  }
  poplar::Tensor rhs = poplar::concat(rhsVec, 0);
  assert(rhs.rank() == 2);
  if (rhs.shape() != shape) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul allocator: created tensor has unexpected shape: " +
                                 std::to_string(rhs.shape()[0]) + " x " + std::to_string(rhs.shape()[1]) + "."
                                 " The should be " + std::to_string(shape[0]) + " x " + std::to_string(shape[1]) + ".");
  }
  logger->debug((debugPrefix + " created rhs tensor of shape: {} x {}").c_str(), rhs.dim(0), rhs.dim(1));
  logger->trace((debugPrefix + "BuildDSD_allocator() exit").c_str());
  return rhs;
}

/// Meta data function sets properties of the gradient op.
void BuildDSD_grad_metadata(std::vector<std::int64_t>& allocating_indices,
                            std::vector<std::int64_t>& replica_identical_output_indices,
                            std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
                            bool& is_elementwise,
                            bool& is_stateless,
                            bool& is_hashable,
                            std::uint32_t num_inputs) {

  logger->trace("BuildDSD_grad_metadata()");
  allocating_indices.clear();
  is_elementwise = false;
  is_stateless = true;
}

void BuildDDS_grad_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::vector<std::int64_t>& replica_identical_output_indices,
                    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
                    bool& is_elementwise,
                    bool& is_stateless,
                    bool& is_hashable,
                    std::uint32_t num_inputs) {

  logger->trace("BuildDDS_grad_metadata()");
  allocating_indices.clear();
  is_elementwise = false;
  is_stateless = true;
}

static poplar::program::Program InternalBuild_grad(
  poplar::Graph& graph, int inputGradIndex,
  const std::vector<poplar::Tensor>& gradients,
  const std::vector<poplar::Tensor>& fwdInputs,
  const std::vector<poplar::Tensor>& fwdOutputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  bool sparseOutput,
  const std::string& debugPrefix) {

  if (gradients.size() != 1) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul grad: 1 gradient required.");
  }
  if (fwdInputs.size() != 2) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul grad: 2 forward inputs required.");
  }
  if (fwdOutputs.size() != 1) {
    throw poputil::poplibs_error(debugPrefix + " bs matmul grad: 1 forward output required.");
  }

  BsMatMulArgs args = parseBsMatMulJsonArgs(attributes);

  poplar::Tensor dY = gradients[0];
  poplar::Tensor x = fwdInputs[0];
  poplar::Tensor w = fwdInputs[1];

  std::array<int, 3> dim;
  std::array<int, 3> blockSize;

  poplar::program::Sequence prog;

  if (!sparseOutput) {
    // dY x dWt = dX
    dim[0] = args.dim[0];
    dim[1] = args.dim[2];
    dim[2] = args.dim[1];

    blockSize[0] = args.blockSize[0];
    blockSize[1] = args.blockSize[2];
    blockSize[2] = args.blockSize[1];
  } else {
    // dY is sparse, must be on a right side
    // dW x dYt = dXt
    dim[0] = args.dim[1];
    dim[1] = args.dim[2];
    dim[2] = args.dim[0];

    blockSize[0] = args.blockSize[1];
    blockSize[1] = args.blockSize[2];
    blockSize[2] = args.blockSize[0];
  }

  poplar::Tensor dX;

  // Sometimes tensorflow runtime
  // does not call gradient calculations for the input 0
  // That is why we have to use separate_gradients=False
  // and always compute gradients for input and weight

  if (!sparseOutput) {
    dX = BsMatMul(graph, dY, w, prog,
                  dim,
                  blockSize,
                  args.sparsityMask,
                  !args.transposedRhs,
                  args.dataType,
                  args.partialDataType,
                  args.innerGroupSize,
                  args.partitionMethod,
                  args.memoryCycleRatio,
                  false, false,
                  debugPrefix + "_bwd");
  } else {
    auto dXt = BsMatMul(graph, w, dY, prog,
                        dim,
                        blockSize,
                        args.sparsityMask,
                        true,
                        args.dataType,
                        args.partialDataType,
                        args.innerGroupSize,
                        args.partitionMethod,
                        args.memoryCycleRatio,
                        false, false,
                        debugPrefix + "_bwd");
    std::array<unsigned, 2> dXtlast2Dims = {dXt.rank() - 2, dXt.rank() - 1};
    dX = dXt.dimShufflePartial({dXtlast2Dims[0], dXtlast2Dims[1]}, {dXtlast2Dims[1], dXtlast2Dims[0]});
  }
  assert(dX.valid());
  outputs.push_back(dX);

  poplar::Tensor dW;
  if (sparseOutput || !args.transposedRhs) {
    // dW must come in non-transposed form
    // Xt x dY = dW
    std::array<unsigned, 2> xlast2Dims = {x.rank() - 2, x.rank() - 1};
    poplar::Tensor xt = x.dimShufflePartial({xlast2Dims[0], xlast2Dims[1]}, {xlast2Dims[1], xlast2Dims[0]});

    dim[0] = args.dim[1];
    dim[1] = args.dim[0];
    dim[2] = args.dim[2];

    blockSize[0] = args.blockSize[1];
    blockSize[1] = args.blockSize[0];
    blockSize[2] = args.blockSize[2];

    dW = BsMatMul(graph, xt, dY, prog,
                  dim,
                  blockSize,
                  args.sparsityMask,
                  false,
                  args.dataType,
                  args.partialDataType,
                  args.innerGroupSize,
                  args.partitionMethod,
                  args.memoryCycleRatio,
                  !sparseOutput, false,
                  debugPrefix + "_wu");
  } else {
    // dW must come in transposed form
    // dYt x X = dWt
    std::array<unsigned, 2> dYlast2Dims = {dY.rank() - 2, dY.rank() - 1};
    poplar::Tensor dYt = dY.dimShufflePartial({dYlast2Dims[0], dYlast2Dims[1]}, {dYlast2Dims[1], dYlast2Dims[0]});

    dim[0] = args.dim[2];
    dim[1] = args.dim[0];
    dim[2] = args.dim[1];

    blockSize[0] = args.blockSize[2];
    blockSize[1] = args.blockSize[0];
    blockSize[2] = args.blockSize[1];

    dW = BsMatMul(graph, dYt, x, prog,
                  dim,
                  blockSize,
                  args.sparsityMask,
                  false,
                  args.dataType,
                  args.partialDataType,
                  args.innerGroupSize,
                  args.partitionMethod,
                  args.memoryCycleRatio,
                  true, false,
                  debugPrefix + "_wu");
  }
  assert(dW.valid());
  outputs.push_back(dW);
  return prog;
}

poplar::program::Program BuildDSD_grad(
  poplar::Graph& graph, int inputGradIndex,
  const std::vector<poplar::Tensor>& gradients,
  const std::vector<poplar::Tensor>& fwdInputs,
  const std::vector<poplar::Tensor>& fwdOutputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  logger->trace("BuildDSD_grad({})", inputGradIndex);
  return InternalBuild_grad(graph, inputGradIndex, gradients, fwdInputs, fwdOutputs, outputs, attributes, false, debugPrefix);
}

poplar::program::Program BuildDDS_grad(
  poplar::Graph& graph, int inputGradIndex,
  const std::vector<poplar::Tensor>& gradients,
  const std::vector<poplar::Tensor>& fwdInputs,
  const std::vector<poplar::Tensor>& fwdOutputs,
  std::vector<poplar::Tensor>& outputs,
  const std::string& attributes,
  const std::string& debugPrefix) {

  logger->trace("BuildDDS_grad({})", inputGradIndex);
  return InternalBuild_grad(graph, inputGradIndex, gradients, fwdInputs, fwdOutputs, outputs, attributes, true, debugPrefix);
}

}
