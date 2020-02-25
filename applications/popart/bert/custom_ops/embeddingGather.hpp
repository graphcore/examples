// Copyright 2019 Graphcore Ltd.
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/popx/opx.hpp>
#include <popart/tensornames.hpp>

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
    std::string embedding_tensor_id = "";

    EmbeddingGatherOp(const popart::OperatorIdentifier &_opid,
                      const popart::Op::Settings &settings_,
                      const MatMulSplit split,
                      const std::string embedding_tensor_id);

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
    MatMulSplit split;
    const std::string embedding_tensor_id = "";

    EmbeddingGatherGradOp(const EmbeddingGatherOp &op,
                          const MatMulSplit split,
                          const std::string embedding_tensor_id);

    std::unique_ptr<popart::Op> clone() const final;
    const std::vector<popart::GradInOutMapper> &gradInputInfo() const final;
    const std::map<int, int> &gradOutToNonGradIn() const final;
    void setup() final;

    static popart::InIndex gradInIndex() { return 0; }
    static popart::InIndex indicesInIndex() { return 1; }
    // The data input (dict) is also passed to the Grad but only so its tile layout can be cloned.
    static popart::InIndex dataInIndex() { return 2; }
    static popart::InIndex acclSliceInputFirstIndex() { return 3; }

    static popart::InIndex gradOutIndex() { return 0; }

    static std::string acclTensorPrefix() {
        return std::string(popart::reservedAcclToAccumulatorPrefix()) + "MLM/MatMul";
    }

    float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class EmbeddingGatherOpx : public popart::popx::Opx
{
public:
    EmbeddingGatherOpx(popart::Op *, popart::popx::Devicex *);
    void grow(poplar::program::Sequence &) const final;

    // This will ensure the dictionary is layed out split if required
    poplar::Tensor createInput(int index, const std::string &name) const override;
    popart::popx::InputCreatorType getInputCreatorType(int index0) const override;
    bool createsEquiv(int index0, const popart::popx::Opx *opx1, int index1) const override;
    std::vector<popart::TensorId> mustExistBeforeCreate(int index0) const override;
private:
    poplar::Tensor serialisedGather(const poplar::Tensor &data, const poplar::Tensor &indices, poplar::program::Sequence &prog) const;

};

class EmbeddingGatherGradOpx : public popart::popx::Opx
{
public:
    EmbeddingGatherGradOpx(popart::Op *, popart::popx::Devicex *);

    // This will ensure that if using a split embedding:
    // the accumulators are layed out as copies of the weights, instead of copies of the gradients.
    poplar::Tensor createInput(int index, const std::string &name) const override;
    popart::popx::InputCreatorType getInputCreatorType(int index0) const override;
    std::vector<popart::TensorId> mustExistBeforeCreate(int index0) const override;

    void grow(poplar::program::Sequence &) const final;

private:
    float getScaledDampeningScalar(const popart::TensorId tensorId, float defaultValue = 1.0f) const;
    void untiedGradUpdate(poplar::program::Sequence &prog,
                          const poplar::Tensor &update,
                          const poplar::Tensor &indices,
                          const poplar::Tensor &scale) const;
    void tiedGradUpdate(poplar::program::Sequence &prog,
                        const poplar::Tensor &update,
                        const poplar::Tensor &indices,
                        const poplar::Tensor &scale) const;
};
