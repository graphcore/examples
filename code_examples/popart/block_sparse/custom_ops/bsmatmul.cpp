// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "bsmatmul.hpp"
#include <popart/popx/opxmanager.hpp>
#include <popsparse/experimental/BlockSparseMatMul.hpp>
#include <poputil/exceptions.hpp>
#include <functional>

using namespace std;

BsMatMulOp::BsMatMulOp(const OperatorIdentifier &opid_,
                       const std::vector<int64_t> &bsrLengthsPer2dPlane_,
                       const std::vector<int64_t> &matrixDims_,
                       const std::vector<int64_t> &blockSizes_,
                       const std::vector<int64_t> &sparsityMask_,
                       const BsMatMulType bsMatMulType_,
                       const bool transposeRhs_,
                       const poplar::Type inType_,
                       const poplar::Type outType_,
                       const poplar::Type ppType_,
                       const std::string &partitionMethod_,
                       const float memoryCycleRatio_,
                       unsigned innerGroupSize_,
                       const Op::Settings &settings_,
                       const std::string &debugStr_) :
                      Op(opid_, settings_) {

  bsrLengthsPer2dPlane = bsrLengthsPer2dPlane_;
  std::copy_n(matrixDims_.begin(), 3, dims.begin());
  std::copy_n(blockSizes_.begin(), 3, blockSizes.begin());

  // convert sparsity mask from std::vector<int> to std::vector<unsigned char>
  std::transform(std::begin(sparsityMask_), std::end(sparsityMask_),
              std::back_inserter(sparsityMask), [](const int i){ return (i);});

  bsMatMulType     = bsMatMulType_;
  transposeRhs     = transposeRhs_;
  inType           = inType_;
  outType          = outType_;
  ppType           = ppType_;
  partitionMethod  = partitionMethod_;
  memoryCycleRatio = memoryCycleRatio_;
  innerGroupSize   = innerGroupSize_;
  debugStr         = debugStr_;

  if (bsMatMulType == BsMatMulType::DENSE_LHS_SPARSE_RHS_DENSE_OUT) {
    size_t numRowsOfBlockForRhs = dims.at(1) / blockSizes.at(1);
    size_t numColsOfBlockForRhs = dims.at(2) / blockSizes.at(2);

    sparsityMaskSizePerMatMul = numRowsOfBlockForRhs * numColsOfBlockForRhs;
  } else if (bsMatMulType == BsMatMulType::DENSE_LHS_DENSE_RHS_SPARSE_OUT) {
    size_t numRowsOfBlockForOutput = dims.at(0) / blockSizes.at(0);
    size_t numColsOfBlockForOutput = dims.at(2) / blockSizes.at(2);

    sparsityMaskSizePerMatMul = numRowsOfBlockForOutput * numColsOfBlockForOutput;
  } else {
    throw error("Illegal bsMatMulType");
  }
}

std::vector<std::unique_ptr<Op>> BsMatMulOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<BsMatMulGradOp>(*this));

  return upops;
}

std::unique_ptr<Op> BsMatMulOp::clone() const {
  return std::make_unique<BsMatMulOp>(*this);
}

void BsMatMulOp::setup() {
  Shape lhsShape = inInfo(0).shape();
  Shape rhsShape = inInfo(1).shape();

  Shape outShape;
  if (bsMatMulType == BsMatMulType::DENSE_LHS_SPARSE_RHS_DENSE_OUT) {
    // want rank rhs to be 2,
    assert(rhsShape.size() == 2);

    // in this case, lhs is a n-d tensor and rhs is a 2d tensor of shape
    // [total number of non-zero blocks across all n dims, num of elems per block]
    int64_t numElemsPerBlockInRhs = blockSizes.at(1) * blockSizes.at(2);
    int64_t numNonZeroBlocksAcrossNd = 0;
    for(auto it : sparsityMask) {
      // sparsityMask is a vector<unsigned char>
      numNonZeroBlocksAcrossNd = (it == 1) ?  numNonZeroBlocksAcrossNd + 1:
                                              numNonZeroBlocksAcrossNd;
    }
    Shape expectedRhsShape = {numNonZeroBlocksAcrossNd, numElemsPerBlockInRhs};
    if (rhsShape != expectedRhsShape) {
      throw error("expected rhs shape does not match what was received");
    }

    if (lhsShape.size() > 2) {
      // copy the group dimensions as is
      for(size_t i = 0; i + 2 < lhsShape.size(); i++) {
        outShape.push_back(lhsShape.at(i));
      }
    }

    outShape.push_back(dims.at(0));
    outShape.push_back(dims.at(2));
  }
  else if (bsMatMulType == BsMatMulType::DENSE_LHS_DENSE_RHS_SPARSE_OUT) {
    if (lhsShape.size() != rhsShape.size()) {
      throw error("lhs and rhs must have same rank");
    }

    for (size_t i = 0; i + 2 < lhsShape.size(); i++) {
      if (lhsShape.at(i) != rhsShape.at(i)) {
        throw error("lhs and rhs must have same values in group dimensions ");
      }
    }

    if (lhsShape.at(lhsShape.size() - 1) != rhsShape.at(lhsShape.size() - 2)) {
      throw error("common dimensions must match");
    }

    int64_t numElemsPerBlockInOutput = blockSizes.at(0) * blockSizes.at(2);
    int64_t numNonZeroBlocksAcrossNd = 0;
    for (auto it : sparsityMask) {
      // sparsityMask is a vector<unsigned char>
      numNonZeroBlocksAcrossNd = (it == 1) ?  numNonZeroBlocksAcrossNd + 1:
                                              numNonZeroBlocksAcrossNd;
    }

    outShape = {numNonZeroBlocksAcrossNd, numElemsPerBlockInOutput};
  } else {
    throw error("Illegal baMatMul type");
  }

  outInfo(0) = {inInfo(0).dataType(), outShape};
}

//register op
static OpDefinition::DataTypes BsMatMulOpTensorType = { DataType::FLOAT16,
                                                        DataType::FLOAT};

static OpDefinition
    bsMatMulOpDef({
      OpDefinition::Inputs
      (
        {
          {"lhs", BsMatMulOpTensorType},
          {"rhs", BsMatMulOpTensorType},
        }
      ),
      OpDefinition::Outputs
      (
        {
          {"Y", BsMatMulOpTensorType}
        }
      ),
      OpDefinition::Attributes
      (
        {
          {"bsr_rhs_lengths_per_2d_plane",   {"*"}},
          {"matrix_dims",                    {"*"}},
          {"block_size",                     {"*"}},
          {"sparsity_mask",                  {"*"}},
          {"transpose_rhs",                  {"*"}},
          {"bsmatmul_type",                  {"*"}},
          {"in_type",                        {"*"}},
          {"out_type",                       {"*"}},
          {"pp_type",                        {"*"}},
          {"partition_method",               {"*"}},
          {"memory_cycle_ratio",             {"*"}},
          {"inner_group_size" ,              {"*"}}
        }
      )
    });

static OpCreator<BsMatMulOp> BsMatMulOpCreator(
    OpDefinitions({{CustomOperators::bsMatMul, bsMatMulOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {

      std::vector<int64_t> bsr_rhs_lengths_per_2d_plane   =
        attr.getAttribute<Attributes::Ints>("bsr_rhs_lengths_per_2d_plane");
      std::vector<int64_t> matrix_dims   =
        attr.getAttribute<Attributes::Ints>("matrix_dims");
      std::vector<int64_t> block_size    =
        attr.getAttribute<Attributes::Ints>("block_size");
      std::vector<int64_t> sparsity_mask =
        attr.getAttribute<Attributes::Ints>("sparsity_mask");
      BsMatMulType bsMatMulType          =
        BsMatMulType(attr.getAttribute<Attributes::Int>("bsmatmul_type"));
      bool transposeRhs                  =
        attr.getAttribute<Attributes::Int>("transpose_rhs") == 1;

      std::string type = attr.getAttribute<Attributes::String>("in_type");
      poplar::Type inType  = (type == "float32") ? poplar::FLOAT : poplar::HALF;

      type = attr.getAttribute<Attributes::String>("out_type");
      poplar::Type outType = (type == "float32") ? poplar::FLOAT : poplar::HALF;

      type = attr.getAttribute<Attributes::String>("pp_type");
      poplar::Type ppType  = (type == "float32") ? poplar::FLOAT : poplar::HALF;

      std::string partitionMethod = attr.getAttribute<Attributes::String>("partition_method", "block-naive");

      float memoryCycleRatio             =
      attr.getAttribute<Attributes::Float>("memory_cycle_ratio", 1.0);

      unsigned innerGroupSize            =
      std::max(1U, static_cast<unsigned>(attr.getAttribute<Attributes::Int>("inner_group_size", 1U)));

      std::string debugStr = attr.getAttribute<Attributes::String>("debug_str", "bs-matmul");

      // Input validation
      if(bsMatMulType < BsMatMulType::DENSE_LHS_SPARSE_RHS_DENSE_OUT ||
         bsMatMulType > BsMatMulType::DENSE_LHS_DENSE_RHS_SPARSE_OUT)
      {
        throw error("Illegal bsMatMulType:{}",
                    static_cast<int64_t>(bsMatMulType));
      }

      return std::unique_ptr<Op>(new BsMatMulOp(_opid,
                                                bsr_rhs_lengths_per_2d_plane,
                                                matrix_dims,
                                                block_size,
                                                sparsity_mask,
                                                bsMatMulType,
                                                transposeRhs,
                                                inType,
                                                outType,
                                                ppType,
                                                partitionMethod,
                                                memoryCycleRatio,
                                                innerGroupSize,
                                                settings,
                                                debugStr));
    },
    true);


BsMatMulOpx::BsMatMulOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {

  verifyOp<BsMatMulOp>(op, CustomOperators::bsMatMul);
}

poplar::Tensor BsMatMulOpx::denseLhsSparseRhsDenseOutMatMul(
  poplar::Tensor& inputLhs,
  poplar::Tensor& inputRhs,
  bool transposeRhs,
  const std::array<int, 3>& matrixDims,
  const std::array<int, 3>& blockSizes,
  poplar::program::Sequence& prog) const {

  BsMatMulOp &matmul = getOp<BsMatMulOp>();
  std::vector<size_t> inputGroupDims;
  poplar::Tensor outTensor;

  const std::string &debugStr0 = matmul.getDebugStr();

  try {
    uint32_t numGroupElems = 1;
    bool performGroupedMatmul = inputLhs.rank() > 2;

    uint32_t innerGroupSize = matmul.getInnerGroupSize();
    uint32_t numInnerGroups = 1;
    if (performGroupedMatmul) {
      // reshape the n-d lhs tensor into a 3-d tensor of dims
      // (inputGroupDims, rows, cols)

      // figure out number of groups dimensions
      std::vector<size_t> lhsShape = inputLhs.shape();
      assert(lhsShape.size() >= 2);
      inputGroupDims.reserve(lhsShape.size() - 2);
      for (size_t i = 0; i + 2 < lhsShape.size(); i++) {
        numGroupElems *= lhsShape.at(i);
        inputGroupDims.push_back(lhsShape.at(i));
      }

      if (numGroupElems % innerGroupSize != 0) {
        throw error("number of group elements: " + std::to_string(numGroupElems) + 
        "is not divisible by the size of inner group" + std::to_string(innerGroupSize));
      }
      numInnerGroups = numGroupElems / innerGroupSize;

      // perform reshapePartial
      // beginIndex must lie in the half closed interval [0, rank())
      // endIndex must lie in the half close interval (0, rank()]
      uint32_t beginIdx = 0;
      uint32_t endIdx = lhsShape.size() - 2;

      inputLhs = inputLhs.reshapePartial(beginIdx, endIdx, {numGroupElems});
    } else {
      // in this case, expand the dimension of the lhs by 1 to convert into
      // 3d tensor
      inputLhs = inputLhs.expand({0});
    }
    popart::logging::trace("BsMatMulOpx::denseLhsSparseRhsDenseOutMatMul() numGroupElems = {}, innerGroupSize = {}", numGroupElems, innerGroupSize);

    poplar::OptionFlags options = {
      {"memoryCycleRatio", std::to_string(matmul.getMemoryCycleRatio())},
      {"partitionMethod", matmul.getPartitionMethod()}
    };

    std::string debugStr = debugStr0 + "_dsd_" + matmul.getPartitionMethod();
    if (innerGroupSize > 1) {
      debugStr = debugStr + "(" + std::to_string(innerGroupSize) + ")";
    }
    debugStr = debugStr + "_" + std::to_string(matrixDims[0]) + "x" + std::to_string(matrixDims[1]) + "x" + std::to_string(matrixDims[2]);

    std::vector<poplar::Tensor> outTensors3d;
    uint32_t bytesToCopy = matmul.getsparsityMaskSizePerMatMul() * innerGroupSize;
    for (std::size_t idxIg = 0, idxElem = 0, rhsSliceStart = 0; idxIg < numInnerGroups; ++idxIg, idxElem += innerGroupSize) {
      std::size_t idxNextElem = idxElem + innerGroupSize;

      std::vector<unsigned char> sparsityMask;
      std::copy_n(matmul.getSparsityMask().begin() + idxIg * bytesToCopy,
                  bytesToCopy,
                  std::back_inserter(sparsityMask));

      popart::logging::trace("matmul shape = {} x {} x {}", matrixDims[0], matrixDims[1], matrixDims[2]);

      popsparse::experimental::BSMatMulParams bsMatMulObj =
        popsparse::experimental::BSMatMulParams(  matrixDims,
                                                  blockSizes,
                                                  sparsityMask,
                                                  transposeRhs,
                                                  matmul.getInType(),
                                                  matmul.getOutType(),
                                                  matmul.getPpType(),
                                                  innerGroupSize);

      poplar::Tensor lhs = inputLhs.slice(idxElem, idxNextElem);
      assert(lhs.rank() >= 2);
      std::size_t lhsCols = lhs.dim(lhs.rank() - 1);
      std::size_t lhsRows = lhs.dim(lhs.rank() - 2);
      lhs = lhs.reshape({innerGroupSize * lhsRows, lhsCols});

      std::size_t rhsSparseGroupLen = std::accumulate(sparsityMask.begin(), sparsityMask.end(), 0);
      std::size_t rhsSliceEnd = rhsSliceStart + rhsSparseGroupLen;
      poplar::Tensor rhs = inputRhs.slice(rhsSliceStart, rhsSliceEnd);
      rhsSliceStart = rhsSliceEnd;

      // perform 2d multiplication
      poplar::Tensor out = popsparse::experimental::bsMatMul(graph(),
                                                            bsMatMulObj,
                                                            prog,
                                                            lhs,
                                                            rhs,
                                                            options,
                                                            debugStr);

      popart::logging::trace("out shape = {} x {}", out.dim(0), out.dim(1));
      
      std::size_t outRowsTotal = out.dim(0);
      assert(outRowsTotal % innerGroupSize == 0);
      std::size_t outRows = outRowsTotal / innerGroupSize;
      for (uint32_t r = 0; r < outRowsTotal; r += outRows) {
        outTensors3d.push_back(out.slice(r, r + outRows).expand({0}));
      }
    }

    // concatenate the 3d tensors into single 3D tensor
    outTensor = poplar::concat(outTensors3d, 0);

    std::vector<size_t> outTensorShape = outTensor.shape();
    std::vector<size_t> expectedOutShape = {numGroupElems,
                                            static_cast<size_t>(matrixDims.at(0)),
                                            static_cast<size_t>(matrixDims.at(2))};
    if (outTensorShape != expectedOutShape) {
      throw error("outTensorShape and expectedOutShape are not matching");
    }

    // need to reshape the outTensor
    if (performGroupedMatmul) {
      std::vector<size_t> newOutShape = inputGroupDims;
      newOutShape.push_back(matrixDims.at(0));
      newOutShape.push_back(matrixDims.at(2));

      outTensor = outTensor.reshape(newOutShape);
    } else {
      // remove singleton dimension
      outTensor = outTensor.squeeze({0});
    }

  } catch (const poputil::poplibs_error &e) {
    std::cout << "bsMatMul() failed with message " << e.what() << std::endl;
  }
  return outTensor;
}

poplar::Tensor BsMatMulOpx::denseLhsDenseRhsSparseOutMatMul(
  poplar::Tensor& inputLhs,
  poplar::Tensor& inputRhs,
  bool transposeRhs,
  const std::array<int, 3>& matrixDims,
  const std::array<int, 3>& blockSizes,
  poplar::program::Sequence& prog) const {

  // sparse matmul
  BsMatMulOp &matmul = getOp<BsMatMulOp>();
  std::vector<size_t> inputGroupDims;
  poplar::Tensor outTensor;

  const std::string &debugStr0 = matmul.getDebugStr();

  try {
    uint32_t numGroupElems = 1;
    bool performGroupedMatmul = inputLhs.rank() > 2;

    uint32_t innerGroupSize = matmul.getInnerGroupSize();
    uint32_t numInnerGroups = 1;
    if (performGroupedMatmul) {
      // figure out number of groups dimensions
      std::vector<size_t> lhsShape = inputLhs.shape();
      inputGroupDims.reserve(lhsShape.size() - 2);
      for (size_t i = 0; i + 2 < lhsShape.size(); i++) {
        numGroupElems *= lhsShape.at(i);
        inputGroupDims.push_back(lhsShape.at(i));
      }

      if (numGroupElems % innerGroupSize != 0) {
        throw error("number of group elements: " + std::to_string(numGroupElems) + 
        "is not divisible by the size of inner group" + std::to_string(innerGroupSize));
      }
      numInnerGroups = numGroupElems / innerGroupSize;

      // perform reshapePartial
      // beginIndex must lie in the half closed interval [0, rank())
      // endIndex must lie in the half close interval (0, rank()]
      uint32_t beginIdx = 0;
      uint32_t endIdx = lhsShape.size() - 2;

      inputLhs = inputLhs.reshapePartial(beginIdx, endIdx, {numGroupElems});
      inputRhs = inputRhs.reshapePartial(beginIdx, endIdx, {numGroupElems});
    }
    else {
      // in this case, expand the dimension of the lhs and rhs by 1 to convert into
      // 3d
      inputLhs = inputLhs.expand({0});
      inputRhs = inputRhs.expand({0});
    }
    popart::logging::trace("BsMatMulOpx::denseLhsDenseRhsSparseOutMatMul() numGroupElems = {}, innerGroupSize = {}", numGroupElems, innerGroupSize);

    poplar::OptionFlags options = {
      {"memoryCycleRatio", std::to_string(matmul.getMemoryCycleRatio())},
      {"partitionMethod", matmul.getPartitionMethod()}
    };

    std::string debugStr = debugStr0 + "_dds_" + matmul.getPartitionMethod();
    if (innerGroupSize > 1) {
      debugStr = debugStr + "(" + std::to_string(innerGroupSize) + ")";
    }
    debugStr = debugStr + "_" + std::to_string(matrixDims[0]) + "x" + std::to_string(matrixDims[1]) + "x" + std::to_string(matrixDims[2]);

    std::vector<poplar::Tensor> outTensors2d;
    uint32_t bytesToCopy = matmul.getsparsityMaskSizePerMatMul() * innerGroupSize;
    for (std::size_t idxIg = 0, idxElem = 0; idxIg < numInnerGroups; ++idxIg, idxElem += innerGroupSize) {
      std::size_t idxNextElem = idxElem + innerGroupSize;

      std::vector<unsigned char> sparsityMask;
      std::copy_n(matmul.getSparsityMask().begin() + idxIg * bytesToCopy,
                  bytesToCopy,
                  std::back_inserter(sparsityMask));

      popart::logging::trace("matmul shape = {} x {} x {}", matrixDims[0], matrixDims[1], matrixDims[2]);

      popsparse::experimental::BSMatMulParams bsMatMulObj =
        popsparse::experimental::BSMatMulParams(  matrixDims,
                                                  blockSizes,
                                                  sparsityMask,
                                                  //transposeRhs,
                                                  matmul.getInType(),
                                                  matmul.getOutType(),
                                                  matmul.getPpType(),
                                                  popsparse::experimental::SubBlockMask::None,
                                                  innerGroupSize);
      
      poplar::Tensor lhs = inputLhs.slice(idxElem, idxNextElem);
      assert(lhs.rank() >= 2);
      std::size_t lhsCols = lhs.dim(lhs.rank() - 1);
      std::size_t lhsRows = lhs.dim(lhs.rank() - 2);
      lhs = lhs.reshape({innerGroupSize * lhsRows, lhsCols});

      poplar::Tensor rhs = inputRhs.slice(idxElem, idxNextElem);
      assert(rhs.rank() >= 2);
      std::size_t rhsCols = rhs.dim(rhs.rank() - 1);
      std::size_t rhsRows = rhs.dim(rhs.rank() - 2);
      rhs = rhs.reshape({innerGroupSize * rhsRows, rhsCols});

      // perform 2d multiplication
      poplar::Tensor out = popsparse::experimental::bsMatMul(graph(),
                                                            bsMatMulObj,
                                                            prog,
                                                            lhs,
                                                            rhs,
                                                            options,
                                                            debugStr);

      popart::logging::trace("out shape = {} x {}", out.dim(0), out.dim(1));
      
      std::size_t resSliceStart = 0;
      outTensors2d.push_back(out);
    }

    // concatenate the 2d tensors into a single 2d tensor
    outTensor = poplar::concat(outTensors2d, 0);

    std::vector<size_t> outTensorShape = outTensor.shape();
    size_t totalNumNonZeroBlocks = std::accumulate(
                                    matmul.getSparsityMask().begin(),
                                    matmul.getSparsityMask().end(),
                                    0);

    std::vector<size_t> expectedOutShape = {totalNumNonZeroBlocks,
      static_cast<size_t>(blockSizes.at(0)) *
      static_cast<size_t>(blockSizes.at(2))};

    if(outTensorShape != expectedOutShape) {
      throw error("outTensorShape and expectedOutShape are not matching");
    }

  } catch (const poputil::poplibs_error &e) {
    std::cout << "bsMatMul() failed with message " << e.what() << std::endl;
  }

  return outTensor;
}

void BsMatMulOpx::grow(poplar::program::Sequence& prog) const {

  auto lhsTensor = getInTensor(BsMatMulOp::getLhsInIndex());
  auto rhsTensor = getInTensor(BsMatMulOp::getRhsInIndex());

  BsMatMulOp &matmul = getOp<BsMatMulOp>();
  poplar::Tensor outTensor;
  if(matmul.getBsMatMulType() == BsMatMulType::DENSE_LHS_SPARSE_RHS_DENSE_OUT) {
    outTensor = denseLhsSparseRhsDenseOutMatMul(lhsTensor,
                                                rhsTensor,
                                                matmul.getTransposeRhs(),
                                                matmul.getMatrixDims(),
                                                matmul.getBlockSizes(),
                                                prog);
  } else if(matmul.getBsMatMulType() == BsMatMulType::DENSE_LHS_DENSE_RHS_SPARSE_OUT) {
    outTensor = denseLhsDenseRhsSparseOutMatMul(lhsTensor,
                                                rhsTensor,
                                                matmul.getTransposeRhs(),
                                                matmul.getMatrixDims(),
                                                matmul.getBlockSizes(),
                                                prog);
  }

  setOutTensor(BsMatMulOp::getOutIndex(), outTensor);
}

BsMatMulGradOp::BsMatMulGradOp(const BsMatMulOp &fwdOp)
    : popart::Op(CustomGradOperators::bsMatMulGrad, fwdOp.getSettings()) {
  dims                      = fwdOp.getMatrixDims();
  blockSizes                = fwdOp.getBlockSizes();
  bsrLengthsPer2dPlane      = fwdOp.getBsrLengthsPer2dPlane();
  sparsityMask              = fwdOp.getSparsityMask();
  bsMatMulType              = fwdOp.getBsMatMulType();
  transposeRhs              = fwdOp.getTransposeRhs();
  inType                    = fwdOp.getInType();
  outType                   = fwdOp.getOutType();
  ppType                    = fwdOp.getPpType();
  partitionMethod           = fwdOp.getPartitionMethod();
  memoryCycleRatio          = fwdOp.getMemoryCycleRatio();
  innerGroupSize            = fwdOp.getInnerGroupSize();
  sparsityMaskSizePerMatMul = fwdOp.getsparsityMaskSizePerMatMul();

  fwdLhsTensorInfoInfo      = fwdOp.inInfo(BsMatMulOp::getLhsInIndex());
  fwdRhsTensorInfoInfo      = fwdOp.inInfo(BsMatMulOp::getRhsInIndex());
  debugStr                  = fwdOp.getDebugStr();
}

std::unique_ptr<Op> BsMatMulGradOp::clone() const {
  return std::make_unique<BsMatMulGradOp>(*this);
}

poplar::Tensor BsMatMulGradOpx::denseLhsSparseRhsDenseOutMatMul(
  poplar::Tensor inputLhs,
  poplar::Tensor inputRhs,
  bool transposeRhs,
  const std::array<int, 3>& matrixDims,
  const std::array<int, 3>& blockSizes,
  poplar::program::Sequence& prog) const {

  BsMatMulGradOp &matmul = getOp<BsMatMulGradOp>();
  std::vector<size_t> inputGroupDims;
  poplar::Tensor outTensor;

  const std::string &debugStr0 = matmul.getDebugStr();

  try {
    uint32_t numGroupElems = 1;
    bool performGroupedMatmul = inputLhs.rank() > 2;

    uint32_t innerGroupSize = matmul.getInnerGroupSize();
    uint32_t numInnerGroups = 1;
    if (performGroupedMatmul) {
      // figure out number of groups dimensions
      std::vector<size_t> lhsShape = inputLhs.shape();
      inputGroupDims.reserve(lhsShape.size() - 2);
      for (size_t i = 0; i + 2 < lhsShape.size(); i++) {
        numGroupElems *= lhsShape.at(i);
        inputGroupDims.push_back(lhsShape.at(i));
      }

      if (numGroupElems % innerGroupSize != 0) {
        throw error("number of group elements: " + std::to_string(numGroupElems) + 
        "is not divisible by the size of inner group" + std::to_string(innerGroupSize));
      }
      numInnerGroups = numGroupElems / innerGroupSize;

      // perform reshapePartial
      // beginIndex must lie in the half closed interval [0, rank())
      // endIndex must lie in the half close interval (0, rank()]
      uint32_t beginIdx = 0;
      uint32_t endIdx = lhsShape.size() - 2;

      inputLhs = inputLhs.reshapePartial(beginIdx, endIdx, {numGroupElems});
    } else {
      // in this case, expand the dimension of the lhs by 1 to convert into
      // 3d tensor
      inputLhs = inputLhs.expand({0});

      auto lhsShape = inputLhs.shape();
    }
    popart::logging::trace("BsMatMulGradOpx::denseLhsSparseRhsDenseOutMatMul() {} numGroupElems", numGroupElems);

    std::vector<int64_t> rhsLengths0 = matmul.getBsrLengthsPer2dPlane();
    if (rhsLengths0.size() != numGroupElems) {
      throw error("number of group dimensions and rhs lengths don't match");
    }

    poplar::OptionFlags options = {
      {"memoryCycleRatio", std::to_string(matmul.getMemoryCycleRatio())},
      {"partitionMethod", matmul.getPartitionMethod()}
    };

    std::string debugStr = debugStr0 + "_grad_dsd_" + matmul.getPartitionMethod();
    if (innerGroupSize > 1) {
      debugStr = debugStr + "(" + std::to_string(innerGroupSize) + ")";
    }
    debugStr = debugStr + "_" + std::to_string(matrixDims[0]) + "x" + std::to_string(matrixDims[1]) + "x" + std::to_string(matrixDims[2]);

    std::vector<poplar::Tensor> outTensors3d;
    uint32_t bytesToCopy = matmul.getsparsityMaskSizePerMatMul() * innerGroupSize;
    for (std::size_t idxIg = 0, idxElem = 0, rhsSliceStart = 0; idxIg < numInnerGroups; ++idxIg, idxElem += innerGroupSize) {
      std::size_t idxNextElem = idxElem + innerGroupSize;

      std::vector<unsigned char> sparsityMask;
      std::copy_n(matmul.getSparsityMask().begin() + idxIg * bytesToCopy,
                  bytesToCopy,
                  std::back_inserter(sparsityMask));

      popsparse::experimental::BSMatMulParams bsMatMulObj =
        popsparse::experimental::BSMatMulParams(  matrixDims,
                                                  blockSizes,
                                                  sparsityMask,
                                                  transposeRhs,
                                                  matmul.getInType(),
                                                  matmul.getOutType(),
                                                  matmul.getPpType(),
                                                  innerGroupSize);

      poplar::Tensor lhs = inputLhs.slice(idxElem, idxNextElem);
      assert(lhs.rank() >= 2);
      std::size_t lhsCols = lhs.dim(lhs.rank() - 1);
      std::size_t lhsRows = lhs.dim(lhs.rank() - 2);
      lhs = lhs.reshape({innerGroupSize * lhsRows, lhsCols});

      std::size_t rhsSparseGroupLen = std::accumulate(sparsityMask.begin(), sparsityMask.end(), 0);
      std::size_t rhsSliceEnd = rhsSliceStart + rhsSparseGroupLen;
      poplar::Tensor rhs = inputRhs.slice(rhsSliceStart, rhsSliceEnd);
      rhsSliceStart = rhsSliceEnd;

      // perform 2d multiplication
      poplar::Tensor out = popsparse::experimental::bsMatMul(graph(),
                                                            bsMatMulObj,
                                                            prog,
                                                            lhs,
                                                            rhs,
                                                            options,
                                                            debugStr);

      popart::logging::trace("out shape = {} x {}", out.dim(0), out.dim(1));
      
      std::size_t outRowsTotal = out.dim(0);
      assert(outRowsTotal % innerGroupSize == 0);
      std::size_t outRows = outRowsTotal / innerGroupSize;
      for (uint32_t r = 0; r < outRowsTotal; r += outRows) {
        outTensors3d.push_back(out.slice(r, r + outRows).expand({0}));
      }
    }

    outTensor = poplar::concat(outTensors3d, 0);

    std::vector<size_t> outTensorShape = outTensor.shape();
    std::vector<size_t> expectedOutShape = {numGroupElems,
                                            static_cast<size_t>(matrixDims.at(0)),
                                            static_cast<size_t>(matrixDims.at(2))};
    if (outTensorShape != expectedOutShape) {
      throw error("outTensorShape and expectedOutShape are not matching");
    }

    // need to reshape the outTensor
    if (performGroupedMatmul) {
      std::vector<size_t> newOutShape = inputGroupDims;
      newOutShape.push_back(matrixDims.at(0));
      newOutShape.push_back(matrixDims.at(2));

      outTensor = outTensor.reshape(newOutShape);
    } else {
      // remove singleton dimension
      outTensor = outTensor.squeeze({0});
    }

  } catch (const poputil::poplibs_error &e) {
    std::cout << "bsMatMul() failed with message " << e.what() << std::endl;
  }

  return outTensor;
}

poplar::Tensor BsMatMulGradOpx::denseLhsDenseRhsSparseOutMatMul(
  poplar::Tensor inputLhs,
  poplar::Tensor inputRhs,
  bool transposeRhs,
  const std::array<int, 3>& matrixDims,
  const std::array<int, 3>& blockSizes,
  poplar::program::Sequence& prog) const {

  // sparse matmul
  BsMatMulGradOp &matmul = getOp<BsMatMulGradOp>();
  std::vector<size_t> inputGroupDims;
  poplar::Tensor outTensor;

  const std::string &debugStr0 = matmul.getDebugStr();

  try {
    uint32_t numGroupElems = 1;
    bool performGroupedMatmul = inputLhs.rank() > 2;

    uint32_t innerGroupSize = matmul.getInnerGroupSize();
    uint32_t numInnerGroups = 1;
    if (performGroupedMatmul) {

      // figure out number of groups dimensions
      std::vector<size_t> lhsShape = inputLhs.shape();
      inputGroupDims.reserve(lhsShape.size() - 2);
      for (size_t i = 0; i + 2 < lhsShape.size(); i++) {
        numGroupElems *= lhsShape.at(i);
        inputGroupDims.push_back(lhsShape.at(i));
      }

      if (numGroupElems % innerGroupSize != 0) {
        throw error("number of group elements: " + std::to_string(numGroupElems) + 
        "is not divisible by the size of inner group" + std::to_string(innerGroupSize));
      }
      numInnerGroups = numGroupElems / innerGroupSize;

      // perform reshapePartial
      // beginIndex must lie in the half closed interval [0, rank())
      // endIndex must lie in the half close interval (0, rank()]
      uint32_t beginIdx = 0;
      uint32_t endIdx = lhsShape.size() - 2;

      inputLhs = inputLhs.reshapePartial(beginIdx, endIdx, {numGroupElems});
      inputRhs = inputRhs.reshapePartial(beginIdx, endIdx, {numGroupElems});

      lhsShape = inputLhs.shape();
      std::vector<size_t> rhsShape = inputRhs.shape();
    } else {
      // in this case, expand the dimension of the lhs and rhs by 1 to convert into
      // 3d
      inputLhs = inputLhs.expand({0});
      inputRhs = inputRhs.expand({0});
    }
    popart::logging::trace("BsMatMulGradOpx::denseLhsDenseRhsSparseOutMatMul() {} numGroupElems ", numGroupElems);

    std::vector<int64_t> resLengths0 = matmul.getBsrLengthsPer2dPlane();
    if (resLengths0.size() != numGroupElems) {
      throw error("number of group dimensions and res lengths don't match");
    }

    poplar::OptionFlags options = {
      {"memoryCycleRatio", std::to_string(matmul.getMemoryCycleRatio())},
      {"partitionMethod", matmul.getPartitionMethod()}
    };

    std::string debugStr = debugStr0 + "_grad_dds_" + matmul.getPartitionMethod();
    if (innerGroupSize > 1) {
      debugStr = debugStr + "(" + std::to_string(innerGroupSize) + ")";
    }
    debugStr = debugStr + "_" + std::to_string(matrixDims[0]) + "x" + std::to_string(matrixDims[1]) + "x" + std::to_string(matrixDims[2]);

    std::vector<poplar::Tensor> outTensors2d;
    uint32_t bytesToCopy = matmul.getsparsityMaskSizePerMatMul() * innerGroupSize;
    for (std::size_t idxIg = 0, idxElem = 0; idxIg < numInnerGroups; ++idxIg, idxElem += innerGroupSize) {
      std::size_t idxNextElem = idxElem + innerGroupSize;

      std::vector<unsigned char> sparsityMask;
      std::copy_n(matmul.getSparsityMask().begin() + idxIg * bytesToCopy,
                  bytesToCopy,
                  std::back_inserter(sparsityMask));

      popsparse::experimental::BSMatMulParams bsMatMulObj =
        popsparse::experimental::BSMatMulParams(  matrixDims,
                                                  blockSizes,
                                                  sparsityMask,
                                                  //transposeRhs,
                                                  matmul.getInType(),
                                                  matmul.getOutType(),
                                                  matmul.getPpType(),
                                                  popsparse::experimental::SubBlockMask::None,
                                                  innerGroupSize);

      poplar::Tensor lhs = inputLhs.slice(idxElem, idxNextElem);
      assert(lhs.rank() >= 2);
      std::size_t lhsCols = lhs.dim(lhs.rank() - 1);
      std::size_t lhsRows = lhs.dim(lhs.rank() - 2);
      lhs = lhs.reshape({innerGroupSize * lhsRows, lhsCols});

      poplar::Tensor rhs = inputRhs.slice(idxElem, idxNextElem);
      assert(rhs.rank() >= 2);
      std::size_t rhsCols = rhs.dim(rhs.rank() - 1);
      std::size_t rhsRows = rhs.dim(rhs.rank() - 2);
      rhs = rhs.reshape({innerGroupSize * rhsRows, rhsCols});

      // perform 2d multiplication
      poplar::Tensor out = popsparse::experimental::bsMatMul(graph(),
                                                            bsMatMulObj,
                                                            prog,
                                                            lhs,
                                                            rhs,
                                                            options,
                                                            debugStr);

      popart::logging::trace("out shape = {} x {}", out.dim(0), out.dim(1));
      
      outTensors2d.push_back(out);
    }

    // concatenate the 2d tensors into a single 2d tensor
    outTensor = poplar::concat(outTensors2d, 0);

    std::vector<size_t> outTensorShape = outTensor.shape();
    size_t totalNumNonZeroBlocks = std::accumulate(
                                    matmul.getSparsityMask().begin(),
                                    matmul.getSparsityMask().end(),
                                    0);

    std::vector<size_t> expectedOutShape = {totalNumNonZeroBlocks,
      static_cast<size_t>(blockSizes.at(0)) *
      static_cast<size_t>(blockSizes.at(2))};

    if (outTensorShape != expectedOutShape) {
      for (int i = 0; i < expectedOutShape.size(); i++) {
        std::cout << expectedOutShape.at(i) << ", ";
      }
      std::cout << "\n";

      throw error("outTensorShape and expectedOutShape are not matching");
    }

  } catch (const poputil::poplibs_error &e) {
    std::cout << "bsMatMul() failed with message " << e.what() << std::endl;
  }

  return outTensor;
}

void BsMatMulGradOpx::grow(poplar::program::Sequence &prog) const {

  poplar::Tensor dFwdOutputTensor  = getInTensor(0);
  poplar::Tensor fwdLhsTensor      = getInTensor(1);
  poplar::Tensor fwdRhsTensor      = getInTensor(2);

  poplar::Tensor dFwdLhsTensor, dFwdRhsTensor;

  auto op = getOp<BsMatMulGradOp>();
  if (op.getBsMatMulType() == BsMatMulType::DENSE_LHS_SPARSE_RHS_DENSE_OUT) {
    std::array<int, 3> matrixDimsLhs = {
      op.getMatrixDims()[0], // rows of dFwdOutputTensor will be same as rows of fwdLhsTensor
      op.getMatrixDims()[2], // cols of dFwdOutputTensor will be same as cols of fwdRhsTensor
      op.getMatrixDims()[1]  // cols of fwdRhsTensor' will be same as rows of fwdRhsTensor
    };

    std::array<int, 3> blockSizesLhs = {
      op.getBlockSizes()[0], // rows of dFwdOutputTensor will be same as rows of fwdLhsTensor
      op.getBlockSizes()[2], // cols of dFwdOutputTensor will be same as cols of fwdRhsTensor
      op.getBlockSizes()[1]  // cols of fwdRhsTensor' will be same as rows of fwdRhsTensor
    };

    // dFwdLhsTensor = dFwdOutputTensor * fwdRhsTensor'
    dFwdLhsTensor = denseLhsSparseRhsDenseOutMatMul(dFwdOutputTensor,
                                                    fwdRhsTensor,
                                                    true, //transpose RHS
                                                    matrixDimsLhs,
                                                    blockSizesLhs,
                                                    prog);

    std::array<int, 3> matrixDimsRhs = {
      op.getMatrixDims()[1], // rows of fwdLhsTensor' will be same as cols of fwdLhsTensor
      op.getMatrixDims()[0], // cols of fwdLhsTensor' will be same as rows of fwdLhsTensor
      op.getMatrixDims()[2]  // cols of dFwdOutputTensor will be same as rows of fwdRhsTensor
    };

    std::array<int, 3> blockSizesRhs = {
      op.getBlockSizes()[1], // rows of fwdLhsTensor' will be same as cols of fwdLhsTensor
      op.getBlockSizes()[0], // cols of fwdLhsTensor' will be same as rows of fwdLhsTensor
      op.getBlockSizes()[2]  // cols of dFwdOutputTensor will be same as rows of fwdRhsTensor
    };

    // dFwdRhsTensor = fwdLhsTensor' * dFwdOutputTensor
    std::vector<size_t> fwdLhsTensorShape = fwdLhsTensor.shape();
    uint32_t lastDimIndex = fwdLhsTensorShape.size() - 1;
    uint32_t prevLastDimIndex = fwdLhsTensorShape.size() - 2;

    poplar::Tensor fwdLhsTTensor = fwdLhsTensor.dimShufflePartial({prevLastDimIndex, lastDimIndex}, {lastDimIndex, prevLastDimIndex});
    dFwdRhsTensor = denseLhsDenseRhsSparseOutMatMul(fwdLhsTTensor,
                                                    dFwdOutputTensor,
                                                    false,
                                                    matrixDimsRhs,
                                                    blockSizesRhs,
                                                    prog);
  } else if (op.getBsMatMulType() == BsMatMulType::DENSE_LHS_DENSE_RHS_SPARSE_OUT) {
    std::array<int, 3> matrixDimsLhs = {
      op.getMatrixDims()[1], // rows of fwdRhsTensor
      op.getMatrixDims()[2], // cols of fwdRhsTensor
      op.getMatrixDims()[0]  // cols of dFwdLhsTensor' will be same as rows of fwdLhsTensor
    };

    std::array<int, 3> blockSizesLhs = {
      op.getBlockSizes()[1], // rows of fwdRhsTensor
      op.getBlockSizes()[2], // cols of fwdRhsTensor
      op.getBlockSizes()[0]  // cols of dFwdLhsTensor' will be same as rows of fwdLhsTensor
    };

    // dFwdLhsTensor' = fwdRhsTensor * dFwdOutputTensor'
    dFwdLhsTensor = denseLhsSparseRhsDenseOutMatMul(fwdRhsTensor,
                                                    dFwdOutputTensor,
                                                    true, //transpose RHS
                                                    matrixDimsLhs,
                                                    blockSizesLhs,
                                                    prog);

    std::vector<size_t> dFwdLhsTensorShape = dFwdLhsTensor.shape();
    uint32_t lastDimIndex = dFwdLhsTensorShape.size() - 1;
    uint32_t prevLastDimIndex = dFwdLhsTensorShape.size() - 2;

    dFwdLhsTensor = dFwdLhsTensor.dimShufflePartial({prevLastDimIndex, lastDimIndex}, {lastDimIndex, prevLastDimIndex});

    std::array<int, 3> matrixDimsRhs = {
      op.getMatrixDims()[1], // rows of fwdLhsTensor' will be same as cols of fwdLhsTensor
      op.getMatrixDims()[0], // cols of fwdLhsTensor' will be same as rows of fwdLhsTensor
      op.getMatrixDims()[2]  // cols of dFwdOutputTensor will be same as cols of fwdRhsTensor
    };

    std::array<int, 3> blockSizesRhs = {
      op.getBlockSizes()[1], // rows of fwdLhsTensor' will be same as cols of fwdLhsTensor
      op.getBlockSizes()[0], // cols of fwdLhsTensor' will be same as rows of fwdLhsTensor
      op.getBlockSizes()[2]  // cols of dFwdOutputTensor will be same as cols of fwdRhsTensor
    };

    // dFwdRhsTensor = fwdLhsTensor' * dFwdOutputTensor
    std::vector<size_t> fwdLhsTensorShape = fwdLhsTensor.shape();
    lastDimIndex = fwdLhsTensorShape.size() - 1;
    prevLastDimIndex = fwdLhsTensorShape.size() - 2;

    poplar::Tensor fwdLhsTTensor = fwdLhsTensor.dimShufflePartial({prevLastDimIndex, lastDimIndex}, {lastDimIndex, prevLastDimIndex});
    dFwdRhsTensor = denseLhsSparseRhsDenseOutMatMul(fwdLhsTTensor,
                                                    dFwdOutputTensor,
                                                    false,
                                                    matrixDimsRhs,
                                                    blockSizesRhs,
                                                    prog);
  }
  setOutTensor(0, dFwdLhsTensor);
  setOutTensor(1, dFwdRhsTensor);
}


static popart::popx::OpxCreator<BsMatMulOpx> BsMatMulOpxCreator(CustomOperators::bsMatMul);

static popart::popx::OpxCreator<BsMatMulGradOpx>
BsMatMulGradOpxCreator(CustomGradOperators::bsMatMulGrad);

