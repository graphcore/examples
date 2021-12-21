// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <popops/Encoding.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <poplin/MatMul.hpp>

#include <iostream>


#define NUM_WORKERS     (int(6))
// change `MAX_TOTAL_SIZE` to max_mel_length if datasets changed.
#define MAX_TOTAL_SIZE  (int(870))
#define PARTION_SIZE    (int(MAX_TOTAL_SIZE/NUM_WORKERS)+1)
#define MIN(a, b)       (int((a) > (b) ? (b) : (a)))


/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
extern "C" {
  int32_t custom_op_api_level = 5;
}

// If an operation takes one or more tensors of the same shape,
// and performs an expression on only corresponding elements in
// the input tensors, and produces a tensor of the same shape,
// then it is elementwise.
extern "C" bool IsElementWise() { return false; }

extern "C"
void Build_metadata(
    std::vector<std::int64_t>& allocating_indices,
    std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
    bool& is_elementwise,
    bool& is_stateless,
    bool& is_hashable,
    std::uint32_t num_inputs){
    is_hashable=true;
}

// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs,  const std::string& attributes,
    const std::string& debugPrefix) {
  if (inputs.size() != 2) {
    throw poputil::poplibs_error("LengthRegulator requires 2 inputs");
  }

  if (inputs[0].numElements() == 0 || inputs[1].numElements() == 0) {
    throw poputil::poplibs_error("Size of both input must larger than 0");
  }

  if (inputs[0].rank() != 3) {
    throw poputil::poplibs_error("inputs[0] must be 3D");
  }

  if (inputs[1].rank() != 2) {
    throw poputil::poplibs_error("inputs[1] must be 2D");
  }

  if (inputs[0].dim(0) != inputs[1].dim(0)) {
    throw poputil::poplibs_error("Batch size mis-match");
  }

  auto &target = graph.getTarget();
  auto numTiles = target.getNumTiles();

  auto dType = inputs[0].elementType();
  auto batchSize = inputs[0].dim(0);
  auto dimension = inputs[0].dim(1);
  auto melCount = inputs[0].dim(2);
  auto maxLength = MAX_TOTAL_SIZE;
  auto tileIndex = 0;

  // Sequence program
  auto prog = poplar::program::Sequence();

  // Convert duration to index
  auto duration = inputs[1];
  auto dur2IdxCS = graph.addComputeSet(debugPrefix + "/LR/dur2IdxCS");
  poplar::Tensor index = graph.addVariable(poplar::INT,
                                          {batchSize, maxLength},
                                          debugPrefix + "/LR/Idx");

  for (auto b = 0; b < batchSize; ++b) {
    for (auto l = 0; l < maxLength; ++l) {
      graph.setTileMapping(index[b][l], tileIndex);

      auto dur2IdxVertex = graph.addVertex(dur2IdxCS,
          poputil::templateVertex("duration2IndexVertex", dType));
      graph.setTileMapping(dur2IdxVertex, tileIndex);

      graph.connect(dur2IdxVertex["duration"], duration[b].flatten());
      graph.connect(dur2IdxVertex["index"], index[b].slice(l, l+1, 0));
      graph.setInitialValue(dur2IdxVertex["hmel_size"], melCount);
      graph.setInitialValue(dur2IdxVertex["i"], l);

      tileIndex = (tileIndex + 1) % numTiles;
    }
  }
  prog.add(poplar::program::Execute(dur2IdxCS));

  // Length Regulator
  auto hmel = inputs[0];

  auto paddingShae = hmel.shape();
  paddingShae[2] = 1;

  auto zeroPadding = graph.addConstant(dType, paddingShae, 0, {"zeroPadding"});
  poputil::mapTensorLinearly(graph, zeroPadding);
  auto hmelPadding = poplar::concat(hmel, zeroPadding, 2);

  auto LRCS = graph.addComputeSet(debugPrefix + "/LR/LRCS");

	poplar::Tensor mel = graph.addVariable(dType,
      { batchSize, dimension, maxLength},
		  debugPrefix + "/LR/mel");

	for (auto b = 0u; b < batchSize; ++b) {
    for (auto d = 0u; d < dimension; ++d) {
      graph.setTileMapping(mel[b][d], tileIndex);
      for (auto w = 0u; w < NUM_WORKERS; ++w) {
        auto s = PARTION_SIZE * w;
        auto e = MIN(s + PARTION_SIZE, maxLength);

        poplar::VertexRef lengthRegulatorVertex =
          graph.addVertex(LRCS,
            poputil::templateVertex("lengthRegulatorVertex", dType), {
              {"hmel", hmelPadding[b][d].flatten()},
              {"index", index[b].slice(s, e, 0).flatten()},
              {"mel_out", mel[b][d].slice(s, e, 0).flatten()}}
            );
        graph.setInitialValue(lengthRegulatorVertex["size"], e - s);
        graph.setInitialValue(lengthRegulatorVertex["mel_count"], melCount);
        graph.setTileMapping(lengthRegulatorVertex, tileIndex);
      }
      tileIndex = (tileIndex + 1) % numTiles;
    }
	}
  prog.add(poplar::program::Execute(LRCS));

  outputs.push_back(mel);

  return prog;
}


extern "C" poplar::program::Program Build_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,  const std::string& attributes,
    const std::string& debugPrefix) {

  if (gradients.size() != 1 || fwd_outputs.size() != 1) {
    throw poputil::poplibs_error("Size of FWD outputs and Gradients must be 1");
  }
  if (gradients[0].numElements() == 0 || fwd_outputs[0].numElements() == 0) {
    throw poputil::poplibs_error("Number of elements of gradients and fwd_outputs muse be equal");
  }
  if (fwd_inputs.size() != 2) {
    throw poputil::poplibs_error("LengthRegulator requires 2 fwd_inputs");
  }
  if (fwd_inputs[0].numElements() == 0 || fwd_inputs[1].numElements() == 0) {
    throw poputil::poplibs_error("Size of both fwd_inputs must larger than 0");
  }
  if (fwd_inputs[0].rank() != 3) {
    throw poputil::poplibs_error("fwd_inputs[0] must be 3D");
  }
  if (fwd_inputs[1].rank() != 2) {
    throw poputil::poplibs_error("fwd_inputs[1] must be 2D");
  }
  if (fwd_inputs[0].dim(0) != fwd_inputs[1].dim(0)) {
    throw poputil::poplibs_error("Batch size mis-match");
  }

  auto &target = graph.getTarget();
  auto numTiles = target.getNumTiles();

  auto dType = fwd_inputs[0].elementType();
  auto batchSize = fwd_inputs[0].dim(0);
  auto dimension = fwd_inputs[0].dim(1);
  auto melCount = fwd_inputs[0].dim(2);
  auto maxLength = MAX_TOTAL_SIZE;
  // auto maxLength = fwd_outputs[0].dim(2);
  auto tileIndex = 0;

  // Sequence program
  auto prog = poplar::program::Sequence();

  // Convert duration to index
  auto duration = fwd_inputs[1];
  auto dur2IdxCS = graph.addComputeSet(debugPrefix + "/LRGrad/dur2IdxCS");
  poplar::Tensor index = graph.addVariable(poplar::INT,
                                          {batchSize, maxLength},
                                          debugPrefix + "/LRGrad/Idx");
  for (auto b = 0; b < batchSize; ++b) {
    for (auto l = 0; l < maxLength; ++l) {
      graph.setTileMapping(index[b][l], tileIndex);

      auto dur2IdxVertex = graph.addVertex(dur2IdxCS,
          poputil::templateVertex("duration2IndexVertex", dType));
      graph.setTileMapping(dur2IdxVertex, tileIndex);

      graph.connect(dur2IdxVertex["duration"], duration[b].flatten());
      graph.connect(dur2IdxVertex["index"], index[b].slice(l, l+1, 0));
      graph.setInitialValue(dur2IdxVertex["hmel_size"], melCount);
      graph.setInitialValue(dur2IdxVertex["i"], l);

      tileIndex = (tileIndex + 1) % numTiles;
    }
  }
  prog.add(poplar::program::Execute(dur2IdxCS));

  // Convert index to OneHot format
  auto oneHot = graph.addVariable(dType,
                                  {batchSize, maxLength, melCount + 1},
                                  debugPrefix + "/LRGrad/OneHot");
  poputil::mapTensorLinearly(graph, oneHot);

  for (auto b = 0; b < batchSize; ++b) {
    popops::encodeOneHot(graph, index[b], oneHot[b], prog, "/LRGrad/OneHotCalc");
  }

  // Calc. hmel grad
  auto grad2Hmel = poplin::matMulGrouped(
      graph, gradients[0], oneHot, prog, dType, debugPrefix + "/LRGrad/Grad2Hmel");
  outputs.push_back(grad2Hmel.slice(0, melCount, 2));

  // Calc. duration grad
  auto grad2Idx = popops::map(
      graph,
      popops::expr::Mul(popops::expr::_1, popops::expr::_2),
      {gradients[0], fwd_outputs[0]},
      prog,
      debugPrefix + "/LRGrad/Grad2Idx");
  auto grad2IdxSum =
      popops::reduce(graph, grad2Idx, dType, {1}, popops::Operation::ADD, prog);

  auto grad2Dur = graph.addVariable(dType,
                                    {batchSize, melCount},
                                     debugPrefix + "/LRGrad/grad2Dur");
  poputil::mapTensorLinearly(graph, grad2Dur);

  tileIndex = 0;
  auto grad2DurCS = graph.addComputeSet(debugPrefix + "/LRGrad/grad2DurCS");
  for (auto b = 0; b < batchSize; ++b) {
    for (auto m = 0; m < melCount; ++m) {
      graph.setTileMapping(grad2Dur[b][m], tileIndex);

      auto grad2DurVertex = graph.addVertex(grad2DurCS,
          poputil::templateVertex("grad2DurationVertex", dType));
      graph.setTileMapping(grad2DurVertex, tileIndex);

      graph.connect(grad2DurVertex["gradient"], grad2IdxSum[b].flatten());
      graph.connect(grad2DurVertex["duration"], duration[b].flatten());
      graph.connect(grad2DurVertex["grad2Dur"], grad2Dur[b].slice(m, m+1, 0));
      graph.setInitialValue(grad2DurVertex["index"], m);

      tileIndex = (tileIndex + 1) % numTiles;
    }
  }
  prog.add(poplar::program::Execute(grad2DurCS));
  outputs.push_back(grad2Dur);

  return prog;
}
