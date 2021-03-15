// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <iostream>

/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
extern "C" {
  int32_t custom_op_api_level = 4;
}

/// This is an elementwise operation, so we tell the framework using the
/// Build_metadata function.
extern "C" void Build_metadata(
  std::vector<std::int64_t>& allocating_indices,
  std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
  bool& is_elementwise,
  bool& is_stateless,
  bool& is_hashable,
  std::uint32_t num_inputs) {
  is_elementwise = true;
}

// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& debugPrefix) {
  if (inputs.size() != 2) {
    throw poputil::poplibs_error("VectorAdd requires 2 inputs");
  }

  if (inputs[0].numElements() == 0) {
    return poplar::program::Sequence();
  }

  if (inputs[0].rank() != 1 || inputs[1].rank() != 1) {
    throw poputil::poplibs_error("All inputs must be vectors");
  }

  if (inputs[0].dim(0) != inputs[1].dim(0)) {
    throw poputil::poplibs_error(
        "Length of input vectors must match");
  }

  if (inputs[0].elementType() != inputs[1].elementType()) {
    throw poputil::poplibs_error(
        "Data types of inputs must match");
  }

  auto dType = inputs[0].elementType();

  // Create a ComputeSet which will be executed, and contains the vertices
  auto cs = graph.addComputeSet(debugPrefix + "/VectorAdd");

  // Get the tile mapping for the complete tensor.  We will map the vertices so
  // that they match the layout of the 'x' input tensor (input[0]).  If the 'x'
  // tensor was layed out differently to the other ones, then Poplar will
  // insert code to move the data in the other tensors to the mapped tile. So
  // ideally we would choose the best mapping for the vertices by analysing
  // all of the tensor mappings.
  auto tileMapping = graph.getTileMapping(inputs[0]);

  // Get the target, which descibes properties of the hardware.
  auto target = graph.getTarget();

  // Get the vector width of the particular data type, so that later we can
  // divide the tensor up between workers in an appropriate way.
  const auto vectorWidth = target.getVectorWidth(dType);

  // Create the output tensors
  outputs.push_back(graph.clone(inputs[0]));

  auto xFlat = inputs[0].flatten();
  auto yFlat = inputs[1].flatten();
  auto xOutputFlat = outputs[0].flatten();

  for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
    // If a tile contains no elements of the tensor then do not create any
    // vertices for it.
    if (tileMapping[tile].empty()) {
      continue;
    }

    // Split up the regions of the inputs tensors so that they are evenly
    // distributed between the workers on the tile.
    auto vertexRegions = poputil::splitRegionsBetweenWorkers(
        target, tileMapping[tile], vectorWidth, 2 * vectorWidth);

    for (const auto& regions : vertexRegions) {
      // If a region has no elements, then there is no need to add a vertex for
      // it.
      if (regions.empty()) {
        continue;
      }

      // Add codelets to tiles which work over the regions in the input
      // tensors.
      auto v = graph.addVertex(cs, poputil::templateVertex("VectorAdd", dType),
                               {{"z", xOutputFlat.slices(regions)},
                                {"x", xFlat.slices(regions)},
                                {"y", yFlat.slices(regions)}});

      // Map the vertex onto the appropriate tile.
      graph.setTileMapping(v, tile);

      // Provide a bogus cycle count estimate for the profiler.
      graph.setPerfEstimate(v, 1);
    }
  }

  return poplar::program::Execute(cs);
}
