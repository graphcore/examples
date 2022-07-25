// Copyright (c) 2021 Graphcore Ltd. All rights reserved.


#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <iostream>

/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
extern "C" {
  int32_t custom_op_api_level = 5;
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
  is_elementwise = false;
}

// The Build function constructs the Poplar graph that computes the custom op.
extern "C" poplar::program::Program Build(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& debugPrefix) {
  if (inputs.size() != 2) {
    throw poputil::poplibs_error("MatrixSolveCE requires 2 inputs");
  }

  if (inputs[0].numElements() == 0) {
    return poplar::program::Sequence();
  }

  if (inputs[0].rank() != 3 || inputs[1].rank() != 3) {
    throw poputil::poplibs_error("All inputs must be vectors");
  }

  auto dim_size = inputs[0].dim(1);
  if ((inputs[0].dim(0) != inputs[1].dim(0)) ||
      (inputs[0].dim(1) != inputs[1].dim(1)) ||
      (inputs[0].dim(1) != inputs[0].dim(2)) ||
      (1 != inputs[1].dim(2))) {
    throw poputil::poplibs_error(
        "Length of input matrix and vectors must match");
  }

  if (inputs[0].elementType() != inputs[1].elementType()) {
    throw poputil::poplibs_error(
        "Data types of inputs must match");
  }

  auto dType        = inputs[0].elementType();

  // Create a ComputeSet which will be executed, and contains the vertices
  auto mat_solve_cs = graph.addComputeSet(debugPrefix + "/MatrixSolveCS");

  // Get the target, which descibes properties of the hardware.
  auto         target      = graph.getTarget();
  unsigned int channel_cnt = inputs[0].dim(0);
  unsigned int numTiles    = target.getNumTiles();

  // Create the output tensors
  outputs.push_back(graph.clone(inputs[1]));

  size_t  active_tile_cnt = 0;
  int*    tile_start      = new int[numTiles];
  int*    tile_count      = new int[numTiles];
  int     tile_idx_last   = -1;
  memset(tile_start, 0, numTiles * sizeof(int));
  memset(tile_count, 0, numTiles * sizeof(int));
  for (unsigned i = 0; i < channel_cnt; ++i)
  {
    int   idx = ((unsigned long long)i * (unsigned long long)numTiles) / ((unsigned long long)channel_cnt);
    auto  A   = inputs[0][i].flatten();
    auto  b   = inputs[1][i].flatten();
    auto  x   = outputs[0][i].flatten();
    graph.setTileMapping(A, idx);
    graph.setTileMapping(b, idx);
    graph.setTileMapping(x, idx);

    if(tile_idx_last != idx)
    {
      tile_start[idx] = i;
      active_tile_cnt++;
    }
    tile_count[idx] ++;
    tile_idx_last = idx;
  }
  
  std::string        vertex_name    = "MatrixSolveVertex";
  int                ele_cnt        = dim_size * dim_size;
  int                ele_cntV       = ((ele_cnt + 1) >> 1) << 1;
  int                dim_sizeV      = ((dim_size + 1) >> 1) << 1;
  int                data_size      = 2;
  poplar::Type       int_type       = poplar::INT;
  if(poplar::FLOAT == dType)
    vertex_name = vertex_name + "<float>";
  else
  {
    ele_cntV    = ((ele_cnt + 3) >> 2) << 2;
    dim_sizeV   = ((dim_size + 3) >> 2) << 2;
    int_type    = poplar::SHORT;
    data_size   = 4;
    vertex_name = vertex_name + "<half>";
  }
  
  int                assist_buf_len = 2 * (ele_cntV + dim_sizeV) * data_size;
  poplar::Tensor     assist_buf     = graph.addVariable(dType,    {active_tile_cnt, (size_t)assist_buf_len},   "buf_");

  active_tile_cnt = 0;
  size_t             total_size     = 0;
  for (unsigned i = 0; i < numTiles; ++i)
  {
    if(0 == tile_count[i])
      continue;
    auto  A = inputs[0].slice(tile_start[i], tile_start[i] + tile_count[i]).flatten();
    auto  b = inputs[1].slice(tile_start[i], tile_start[i] + tile_count[i]).flatten();
    auto  x = outputs[0].slice(tile_start[i], tile_start[i] + tile_count[i]).flatten();
    poplar::VertexRef  matrix_solve_vertex = graph.addVertex(mat_solve_cs, vertex_name, 
                                                      {
                                                        {"A_",      A},
                                                        {"b_",      b},
                                                        {"x_",      x},
                                                        {"buf_",    assist_buf[active_tile_cnt]}
                                                      });
    graph.setTileMapping(matrix_solve_vertex, i);
    graph.setTileMapping(assist_buf[active_tile_cnt], i);
    graph.setPerfEstimate(matrix_solve_vertex, 1);
    graph.setInitialValue(matrix_solve_vertex["dim_size_"], (int)dim_size);
    active_tile_cnt ++;
  }

  delete[] tile_start;
  delete[] tile_count;
  return poplar::program::Execute(mat_solve_cs);
}
