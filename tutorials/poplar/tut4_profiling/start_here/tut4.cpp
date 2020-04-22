// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

/* This file contains the starting point for Poplar tutorial 4.
   See the Poplar user guide for details.
*/

#include <iostream>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;

int main() {
  // Create the IPU model device
  IPUModel ipuModel;
  Device device = ipuModel.createDevice();
  Target target = device.getTarget();

  // Create the Graph object
  Graph graph(target);
  // Include the device-side library code
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  // Add variables to the graph
  Tensor m1 = graph.addVariable(FLOAT, {800, 500}, "m1");
  Tensor m2 = graph.addVariable(FLOAT, {500, 400}, "m2");
  Tensor m3 = graph.addVariable(FLOAT, {400, 200}, "m3");
  // Distribute the tensors evenly over the tiles
  poputil::mapTensorLinearly(graph, m1);
  poputil::mapTensorLinearly(graph, m2);
  poputil::mapTensorLinearly(graph, m3);

  // Create a control program that is a sequence of steps
  Sequence prog;

  // Create the engine
  Engine engine(graph, prog);
  engine.load(device);

  // Run the control program
  std::cout << "Running program\n";
  engine.run(0);
  std::cout << "Program complete\n";

  engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});

  return 0;
}
