// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/names.hpp>
#include <popart/tensornames.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/devicex.hpp>

using namespace popart;
using namespace popart::popx;

enum class BsMatMulType {
  DENSE_LHS_SPARSE_RHS_DENSE_OUT = 0,
  DENSE_LHS_DENSE_RHS_SPARSE_OUT = 1,
  SPARSE_LHS_SPARSE_RHS_SPARSE_OUT = 2
};

class BsMatMulOp : public Op {
  std::array<int, 3> dims, blockSizes;
  std::vector<int64_t> bsrLengthsPer2dPlane;
  std::vector<unsigned char> sparsityMask;
  BsMatMulType bsMatMulType;
  bool transposeRhs;
  poplar::Type inType, outType, ppType;
  std::string partitionMethod;
  float memoryCycleRatio;
  uint32_t innerGroupSize;
  size_t sparsityMaskSizePerMatMul = -1;
  std::string debugStr;

  public:
  BsMatMulOp(const OperatorIdentifier &opid,
             const std::vector<int64_t> &bsrLengthsPer2dPlane,
             const std::vector<int64_t> &matrixDims,
             const std::vector<int64_t> &blockSizes,
             const std::vector<int64_t> &sparsityMask,
             const BsMatMulType bsMatMulType,
             const bool transposeRhs,
             const poplar::Type inType,
             const poplar::Type outType,
             const poplar::Type ppType,
             const std::string &partitionMethod,
             const float memoryCycleRatio,
             uint32_t innerGroupSize,
             const Op::Settings &settings,
             const std::string &debugStr);

  BsMatMulOp(const BsMatMulOp &) = default;
  BsMatMulOp &operator=(const BsMatMulOp &) = delete;
  ~BsMatMulOp() override                  = default;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::unique_ptr<Op> clone() const final;
  virtual void setup() final;

  static InIndex getLhsInIndex() { return 0; }
  static InIndex getRhsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  const Tensor *lhsIn() const;
  const Tensor *rhsIn() const;
  const Tensor *out() const;

  const std::array<int, 3>& getMatrixDims() const { return dims;};
  const std::array<int, 3>& getBlockSizes() const { return blockSizes;};
  const std::vector<unsigned char>&
  getSparsityMask() const { return sparsityMask;};
  const BsMatMulType getBsMatMulType() const { return bsMatMulType;};
  const bool getTransposeRhs() const { return transposeRhs;};
  const poplar::Type getInType() const { return inType;};
  const poplar::Type getOutType() const { return outType;};
  const poplar::Type getPpType() const { return ppType;};
  const std::string getPartitionMethod() const { return partitionMethod; }
  const float getMemoryCycleRatio() const { return memoryCycleRatio;};
  uint32_t getInnerGroupSize() const { return innerGroupSize; }

  const std::vector<int64_t>
  getBsrLengthsPer2dPlane() const { return bsrLengthsPer2dPlane;};
  const int64_t getsparsityMaskSizePerMatMul() const { return sparsityMaskSizePerMatMul;};
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  const std::string& getDebugStr() const { return debugStr; }
};

class BsMatMulOpx : public Opx {

public:
  BsMatMulOpx(Op *, Devicex *);
  ~BsMatMulOpx() override = default;

  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor denseLhsSparseRhsDenseOutMatMul(poplar::Tensor& lhs,
                              poplar::Tensor& rhs,
                              bool transposeRhs,
                              const std::array<int, 3>& matrixDims,
                              const std::array<int, 3>& blockSizes,
                              poplar::program::Sequence& prog) const;

  poplar::Tensor denseLhsDenseRhsSparseOutMatMul(poplar::Tensor& lhs,
                              poplar::Tensor& rhs,
                              bool transposeRhs,
                              const std::array<int, 3>& matrixDims,
                              const std::array<int, 3>& blockSizes,
                              poplar::program::Sequence& prog) const;

private:

};

class BsMatMulGradOp : public Op {
  // some settings that need to be stored from forward op
  std::array<int, 3> dims, blockSizes;
  std::vector<int64_t> bsrLengthsPer2dPlane;
  std::vector<unsigned char> sparsityMask;
  BsMatMulType bsMatMulType;
  bool transposeRhs;
  poplar::Type inType, outType, ppType;
  std::string partitionMethod;
  float memoryCycleRatio;
  uint32_t innerGroupSize;
  size_t sparsityMaskSizePerMatMul = -1;

  TensorInfo fwdLhsTensorInfoInfo, fwdRhsTensorInfoInfo;
  std::string debugStr;
public:
  BsMatMulGradOp(const BsMatMulOp &fwdOp);

  std::unique_ptr<Op> clone() const final;
  virtual void setup() {
    // dLhs = dOutput * rhs'
    Shape dLhsShape = fwdLhsTensorInfoInfo.shape();

    // dRhs = lhs' * dOutput
    Shape dRhsShape = fwdRhsTensorInfoInfo.shape();

    outInfo(getDLhsOutIndex()) = {inInfo(getLhsInIndex()).dataType(), dLhsShape};
    outInfo(getDRhsOutIndex()) = {inInfo(getRhsInIndex()).dataType(), dRhsShape};
  }

  static InIndex getDOutputInIndex() { return 0; }
  static InIndex getLhsInIndex()     { return 1; }
  static InIndex getRhsInIndex()     { return 2; }
  static OutIndex getDLhsOutIndex()  { return 0; }
  static OutIndex getDRhsOutIndex()  { return 1; }

  /* Describes the relationship of the inputs of the grad op to the
     inputs/outputs of the non-grad op */
  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    /*
       The grad op takes dOutput, forward lhs and forward rhs as inputs and
       produces dLhs and dRhs as outputs as described by the two equations below.
       dLhs = dOutput * rhs'
       dRhs = lhs' * dOutput
    */
    static const std::vector<popart::GradInOutMapper> inInfo = {
      // The input of grad op at index 0 is the gradient of the input at
      // index 0 of the non-grad op
      {0, 0, popart::GradOpInType::GradOut}, // gradient of output

      // The input of grad op at index 1 is the input at index 0
      // of the non-grad op
      {1, 0, popart::GradOpInType::In}, // forward Lhs

      // The input of grad op at index 2 is the input at index 1
      // of the non-grad op
      {2, 1, popart::GradOpInType::In} // forward Rhs
    };
    return inInfo;
  }

  /* Describes the relationship of the outputs of the grad op to the
     inputs/outputs of the non-grad op */
  virtual const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {
      // The output at index 0 is dLhs, i.e the gradient of the input at index 0
      // of non grad op
      {0, 0},

      // The output at index 1 is dRhs, i.e the gradient of the input at index 1
      // of non grad op
      {1, 1}
    };
    return outInfo;
  }

  const std::array<int, 3>& getMatrixDims() const { return dims;}
  const std::array<int, 3>& getBlockSizes() const { return blockSizes;}
  const std::vector<unsigned char>&
  getSparsityMask() const { return sparsityMask;}
  const BsMatMulType getBsMatMulType() const { return bsMatMulType;}
  const bool getTransposeRhs() const { return transposeRhs;}
  const poplar::Type getInType() const { return inType;}
  const poplar::Type getOutType() const { return outType;}
  const poplar::Type getPpType() const { return ppType;}
  const std::string& getPartitionMethod() const { return partitionMethod; };
  const float getMemoryCycleRatio() const { return memoryCycleRatio;}
  unsigned getInnerGroupSize() const { return innerGroupSize; }

  const std::vector<int64_t>&
  getBsrLengthsPer2dPlane() const { return bsrLengthsPer2dPlane;}
  const int64_t getsparsityMaskSizePerMatMul() const { return sparsityMaskSizePerMatMul;}
  float getSubgraphValue() const final { return getLowSubgraphValue();}
  const std::string& getDebugStr() const { return debugStr; }
};


class BsMatMulGradOpx : public Opx {
public:
  BsMatMulGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<BsMatMulGradOp>(op, CustomGradOperators::bsMatMulGrad);
  }

  void grow(poplar::program::Sequence &prog) const final;

  poplar::Tensor denseLhsSparseRhsDenseOutMatMul(
                              poplar::Tensor lhs,
                              poplar::Tensor rhs,
                              bool transposeRhs,
                              const std::array<int, 3>& matrixDims,
                              const std::array<int, 3>& blockSizes,
                              poplar::program::Sequence& prog) const;

  poplar::Tensor denseLhsDenseRhsSparseOutMatMul(
                              poplar::Tensor lhs,
                              poplar::Tensor rhs,
                              bool transposeRhs,
                              const std::array<int, 3>& matrixDims,
                              const std::array<int, 3>& blockSizes,
                              poplar::program::Sequence& prog) const;
};

