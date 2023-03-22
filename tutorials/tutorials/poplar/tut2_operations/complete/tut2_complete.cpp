// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

/* This file contains the completed version of Poplar tutorial 2.
   See the Poplar user guide for details.
*/

#include <iostream>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

using namespace poplar;
using namespace poplar::program;

int main() {
  // Create the IPU model device
  IPUModel ipuModel;
  Device device = ipuModel.createDevice();
  Target target = device.getTarget();

  // Create the Graph object
  Graph graph(target);
  popops::addCodelets(graph);

  // Add variables to the graph
  Tensor v1 = graph.addVariable(FLOAT, {2, 2}, "v1");
  Tensor v2 = graph.addVariable(FLOAT, {2, 2}, "v2");
  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      graph.setTileMapping(v1[i][j], i * 2 + j);
      graph.setTileMapping(v2[i][j], j * 2 + i);
    }
  }

  // Create a control program that is a sequence of steps
  Sequence prog;

  // Add steps to initialize the variables
  Tensor c1 = graph.addConstant<float>(FLOAT, {4}, {1.0, 1.5, 2.0, 2.5});
  Tensor c2 = graph.addConstant<float>(FLOAT, {4}, {4.0, 3.0, 2.0, 1.0});
  graph.setTileMapping(c1, 0);
  graph.setTileMapping(c2, 0);
  prog.add(Copy(c1, v1.flatten()));
  prog.add(Copy(c2, v2.flatten()));

  // Extend program with elementwise add (this will add to the sequence)
  Tensor v3 = popops::add(graph, v1, v2, prog, "Add");
  prog.add(PrintTensor("v3", v3));

  // Use the result of the previous calculation
  Tensor v4 = popops::add(graph, v3, v2, prog, "Add");
  prog.add(PrintTensor("v4", v4));

  // Example element wise addition using a transposed view of the data
  Tensor v5 = popops::add(graph, v1, v2.transpose(), prog, "Add");
  prog.add(PrintTensor("v5", v5));

  // Create the engine
  Engine engine(graph, prog);
  engine.load(device);

  // Run the control program
  std::cout << "Running program\n";
  engine.run(0);
  std::cout << "Program complete\n";

  return 0;
}
