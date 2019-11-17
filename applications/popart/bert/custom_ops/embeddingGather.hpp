// Copyright 2019 Graphcore Ltd.
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/popx/opx.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier EmbeddingGather = {"ai.graphcore", "EmbeddingGather", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
        const popart::OperatorIdentifier EmbeddingGatherGrad = {"ai.graphcore", "EmbeddingGatherGrad", 1};
} // namespace CustomGradOperators

struct MatMulSplit {
  // dim = 2: serialise channels dimension of the fully connected layer
  // dim = 1: serialise internalSize dimension of the fully connected layer
  // dim = 0: serialise inputSize dimension of the fully connected layer
  unsigned dim  = 0;
  // Factor by which the dimension is serialised. The dimension to serialise
  // must be divisible by this factor
  unsigned factor = 1;
  MatMulSplit(unsigned dim, unsigned factor) : dim(dim), factor(factor) {};
  MatMulSplit() = default;
  MatMulSplit(const MatMulSplit &) = default;
};

class EmbeddingGatherOp : public popart::Op
{
public:
    MatMulSplit split;
    EmbeddingGatherOp(const popart::OperatorIdentifier &_opid,
                      const popart::Op::Settings &settings_,
                      const MatMulSplit split);

    std::unique_ptr<popart::Op> clone() const final;
    std::vector<std::unique_ptr<popart::Op>> getGradOps() final;
    void setup() final;

    static popart::InIndex dataInIndex() { return 0; }
    static popart::InIndex indicesInIndex() { return 1; }
    static popart::InIndex outIndex() { return 0; }

    float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class EmbeddingGatherGradOp : public popart::Op
{
public:
    EmbeddingGatherGradOp(const EmbeddingGatherOp &op);

    std::unique_ptr<popart::Op> clone() const final;
    const std::vector<popart::GradInOutMapper> &gradInputInfo() const final;
    const std::map<int, int> &gradOutToNonGradIn() const final;
    void setup() final;

    static popart::InIndex gradInIndex() { return 0; }
    static popart::InIndex indicesInIndex() { return 1; }
    // The data input (dict) is also passed to the Grad but only so it's tile layout can be cloned.
    static popart::InIndex dataInIndex() { return 2; }
    static popart::InIndex gradOutIndex() { return 0; }

    float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class EmbeddingGatherOpx : public popart::popx::Opx
{
public:
    EmbeddingGatherOpx(popart::Op *, popart::popx::Devicex *);
    void grow(poplar::program::Sequence &) const final;

    // create the input poplar::Tensor for input at index
    // default : throw error (not all popart::popx::Opxs can createInput)
    poplar::Tensor createInput(int index, const std::string &name) const override;
    // default return DEADEND, i.e. unable to create input tensor, and
    // cannot use downstream opxs as candidates to create input
    // tensor
    popart::popx::InputCreatorType getInputCreatorType(int index0) const override;
    // If this popart::popx::Opx creates a poplar::Tensor at index0 (via createInput),
    // does it create the same poplar::Tensor as if opx1 creates one at
    // index1?. default behaviour : throws error
    bool createsEquiv(int index0, const popart::popx::Opx *opx1, int index1) const override;
    // To create a poplar::Tensor for input index index0, which
    // poplar::Tensors must already exist?
    std::vector<popart::TensorId> mustExistBeforeCreate(int index0) const override;
};

class EmbeddingGatherGradOpx : public popart::popx::Opx
{
public:
    EmbeddingGatherGradOpx(popart::Op *, popart::popx::Devicex *);
    void grow(poplar::program::Sequence &) const final;
};
