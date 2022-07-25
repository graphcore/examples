// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef GUARD_NEURALNET_BSSOFTMAX_HPP
#define GUARD_NEURALNET_BSSOFTMAX_HPP

#include <popsparse/experimental/BlockSparse.hpp>
using  popsparse::experimental::SubBlockMask;
using  popsparse::experimental::bsSoftmax;
using  popsparse::experimental::bsSoftmaxGrad;

namespace popart {

class BsSoftmaxOp : public Op {
public:
  BsSoftmaxOp(const OperatorIdentifier &opid,
                         const std::vector<int64_t> &matrixDims,
                         const std::array<int, 2> &blockSize,
                         const std::vector<unsigned char> &sparsity,
                         const std::vector<int64_t> &groupSizes,
                         const std::vector<SubBlockMask> &subBlockMaskTypePerGroup,
                         uint32_t innerGroupSize,
                         const Op::Settings &settings,
                         const std::string &debugStr);
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void appendOutlineAttributes(OpSerialiserBase &os) const override;
  float getSubgraphValue() const final {return getHighSubgraphValue();}

  const std::vector<int64_t>& getMatrixDims() const {return matrixDims;}
  const std::array<int, 2>& getBlockSize() const {return blockSize;}
  const std::vector<unsigned char>& getSparsity() const {return sparsity;}
  const std::vector<int64_t>& getGroupSizes() const {return groupSizes;}
  const std::vector<SubBlockMask>& getSubBlockMaskTypePerGroup() const {return subBlockMaskTypePerGroup;}
  uint32_t getInnerGroupSize() const { return innerGroupSize; }
  const std::string& getDebugStr() const { return debugStr; }

private:
  std::vector<int64_t> matrixDims;
  std::array<int, 2> blockSize;
  std::vector<unsigned char> sparsity;
  std::vector<int64_t> groupSizes;
  std::vector<SubBlockMask> subBlockMaskTypePerGroup;
  uint32_t innerGroupSize;
  std::string debugStr;
};

class BsSoftmaxGradOp : public Op {
public:
  BsSoftmaxGradOp(const BsSoftmaxOp &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  void appendOutlineAttributes(OpSerialiserBase &os) const override;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  
  const std::vector<int64_t>& getMatrixDims() const {return matrixDims;}
  const std::array<int, 2>& getBlockSize() const {return blockSize;}
  const std::vector<unsigned char>& getSparsity() const {return sparsity;}
  std::vector<int64_t> getGroupSizes() const {return groupSizes;}
  const std::vector<SubBlockMask>& getSubBlockMaskTypePerGroup() const {return subBlockMaskTypePerGroup;}
  uint32_t getInnerGroupSize() const { return innerGroupSize; }
  const std::string& getDebugStr() const { return debugStr; }

private:
  std::vector<int64_t> matrixDims;
  std::array<int, 2> blockSize;
  std::vector<unsigned char> sparsity;
  std::vector<int64_t> groupSizes;
  std::vector<SubBlockMask> subBlockMaskTypePerGroup;
  uint32_t innerGroupSize;
  std::string debugStr;
};
} // namespace popart

#endif
