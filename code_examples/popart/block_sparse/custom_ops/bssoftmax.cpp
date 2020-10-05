// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include "bssoftmax.hpp" 
#include <vector>
#include <queue>

namespace popart {

BsSoftmaxOp::BsSoftmaxOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &_matrixDims,
                         const std::array<int, 2> &_blockSize,
                         const std::vector<unsigned char> &_sparsity,
                         const std::vector<int64_t> &groupSizes_,
                         const std::vector<SubBlockMask> &_subBlockMaskTypePerGroup,
                         unsigned _innerGroupSize,
                         const Op::Settings &_settings,
                         const std::string &_debugStr)
    : Op(_opid, _settings), matrixDims(_matrixDims),
      blockSize(_blockSize), sparsity(_sparsity), groupSizes(groupSizes_),
      subBlockMaskTypePerGroup(_subBlockMaskTypePerGroup),
      innerGroupSize(_innerGroupSize),
      debugStr(_debugStr){}

std::vector<std::unique_ptr<Op>> BsSoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<BsSoftmaxGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> BsSoftmaxOp::clone() const {
  return std::make_unique<BsSoftmaxOp>(*this);
}

void BsSoftmaxOp::setup() {
  // Output is same shape as input
  outInfo(0) = inInfo(0);
}


void BsSoftmaxOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("matrixDims", matrixDims);
  std::vector<int64_t> vblockSize{blockSize[0], blockSize[1]};
  os.appendAttribute("blocksize", vblockSize);
  os.appendAttribute("groupSizes", groupSizes);
  std::vector<int64_t> vmasks;
  std::transform(subBlockMaskTypePerGroup.begin(), subBlockMaskTypePerGroup.end(),
                std::back_inserter(vmasks), [](SubBlockMask mask){return static_cast<int64_t>(mask);});
  os.appendAttribute("subBlockMaskTypePerGroup", vmasks);
  std::vector<int64_t> vsparsity(sparsity.begin(), sparsity.end());
  os.appendAttribute("sparsity", vsparsity);
	}

BsSoftmaxGradOp::BsSoftmaxGradOp(const BsSoftmaxOp &op_)
    : Op(CustomGradOperators::BsSoftmaxGrad, op_.getSettings()),
      matrixDims(op_.getMatrixDims()), blockSize(op_.getBlockSize()),
      sparsity(op_.getSparsity()), groupSizes(op_.getGroupSizes()),
      subBlockMaskTypePerGroup(op_.getSubBlockMaskTypePerGroup()),
      innerGroupSize(op_.getInnerGroupSize()),
      debugStr(op_.getDebugStr())  {}

std::unique_ptr<Op> BsSoftmaxGradOp::clone() const {
  return std::make_unique<BsSoftmaxGradOp>(*this);
}

void BsSoftmaxGradOp::setup() {
  // First and only output is the gradient of the input
  outInfo(0) = inInfo(0);
}

void BsSoftmaxGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("matrixDims", matrixDims);
  std::vector<int64_t> vblockSize{blockSize[0], blockSize[1]};
  os.appendAttribute("blocksize", vblockSize);
  os.appendAttribute("groupSizes", groupSizes);
  std::vector<int64_t> vsparsity(sparsity.begin(), sparsity.end());
  os.appendAttribute("sparsity", vsparsity);
}


const std::vector<GradInOutMapper> &BsSoftmaxGradOp::gradInputInfo() const {
  // input at index 0 : upstream backprop gradient
  // input at index 1 : output probabilities of the softmax
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GradOut},
      {1, 0, GradOpInType::Out}};
  return inInfo;
}

const std::map<int, int> &BsSoftmaxGradOp::gradOutToNonGradIn() const {
  // The output at index 0 maps to the logits which were fed into the 
  // original op at index 0
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

namespace {
static OpDefinition BsSoftmaxOpDef({});

static OpCreator<BsSoftmaxOp> BsSoftmaxOpCreator(
    OpDefinitions({{CustomOperators::BsSoftmax, BsSoftmaxOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {

      std::vector<int64_t> matrixDims = attr.getAttribute<Attributes::Ints>("matrixDims");

      Attributes::Ints python_blockSize = attr.getAttribute<Attributes::Ints>("blockSize");
      std::vector<int> cast_python_blockSize(python_blockSize.begin(), python_blockSize.end());
      std::array<int, 2> blockSize{cast_python_blockSize[0], cast_python_blockSize[1]};

      Attributes::Ints python_sparsity = attr.getAttribute<Attributes::Ints>("sparsity");
      std::vector<unsigned char> sparsity(python_sparsity.begin(), python_sparsity.end()); 

      Attributes::Ints groupSizes = attr.getAttribute<Attributes::Ints>("groupSizes");

      uint32_t innerGroupSize = static_cast<unsigned>(attr.getAttribute<Attributes::Int>("innerGroupSize", 0U));
      
      // The mask string actually contains one mask per group
      std::string maskAttribute = attr.getAttribute<Attributes::String>("subBlockMaskPerGroup");
      std::string m1 = "ZeroUpperTriangle";
      std::string m2 = "ZeroLowerTriangle";
      std::string m3 = "None";
      // Alterantive to regex lookup
      std::priority_queue<std::pair<int, SubBlockMask>> masksInOrder;
      size_t pos;
      pos = maskAttribute.find(m1);
      while(pos != std::string::npos){
        masksInOrder.push(std::make_pair(-pos, SubBlockMask::ZeroUpperTriangle));
        pos = maskAttribute.find(m1, pos + m1.size());
      }
      pos = maskAttribute.find(m2);
      while(pos != std::string::npos){
        masksInOrder.push(std::make_pair(-pos, SubBlockMask::ZeroLowerTriangle));
        pos = maskAttribute.find(m2, pos + m2.size());
      }
      pos = maskAttribute.find(m3);
      while(pos != std::string::npos){
        masksInOrder.push(std::make_pair(-pos, SubBlockMask::None));
        pos = maskAttribute.find(m3, pos + m3.size());
      }

      // Check length
      if (masksInOrder.size() != groupSizes.size()){
        std::string error_text = "Sublock mask string does not include include one mask per group: " + maskAttribute + "\n";
        error_text += "Please use one of the following strings per group: \n";
        error_text += "\t- None\n";
        error_text += "\t- ZeroUpperTriangle\n";
        error_text += "\t- ZeroLowerTriangle\n";
        error_text += "e.g. [ZeroLowerTriangle, None] for two groups.\n";
        throw error(error_text);      
      }

      // Convert priority queue into a vector of mask enums
      std::vector<SubBlockMask> subBlockMaskTypePerGroup;
      while (!masksInOrder.empty()){
        std::pair<int, SubBlockMask> p = masksInOrder.top();
        masksInOrder.pop();
        subBlockMaskTypePerGroup.push_back(p.second);
      }
      std::string log{"SubBlockMasksPerGroup has values: "};
      for (SubBlockMask sbm : subBlockMaskTypePerGroup){
        if (sbm == SubBlockMask::ZeroUpperTriangle){
          log += "ZeroUpperTriangle ";
        } else if (sbm == SubBlockMask::ZeroLowerTriangle){
          log += "ZeroLowerTriangle ";
        } else if (sbm == SubBlockMask::None){
          log += "None ";
        }
      }
      logging::info("{}:{} {}", __FILE__, __LINE__, log);

      std::string debugStr = attr.getAttribute<Attributes::String>("debug_str", 
                                                                "bs-softmax");
      return std::unique_ptr<Op>(new BsSoftmaxOp(_opid, matrixDims,
                          blockSize, sparsity, groupSizes, 
                          subBlockMaskTypePerGroup, innerGroupSize, settings, debugStr));
    },
    true);

} // namespace
} // namespace popart

