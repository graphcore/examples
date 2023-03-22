// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

/* This file contains the completed version of Poplar tutorial 4,
  which uses the IPU Model.
*/

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <iostream>

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
  poplin::addCodelets(graph);

  // Add variables to the graph
  Tensor m1 = graph.addVariable(FLOAT, {900, 600}, "m1");
  Tensor m2 = graph.addVariable(FLOAT, {600, 300}, "m2");
  Tensor m3 = graph.addVariable(FLOAT, {300, 200}, "m3");
  poputil::mapTensorLinearly(graph, m1);
  poputil::mapTensorLinearly(graph, m2);
  poputil::mapTensorLinearly(graph, m3);

  // Create a control program that is a sequence of steps
  Sequence prog;

  Tensor m4 = poplin::matMul(graph, m1, m2, prog, "m4");
  Tensor m5 = poplin::matMul(graph, m4, m3, prog, "m5");

  // Create the engine
  // By default, in order to reduce host memory consumption, the generation and
  // retention of debug information is not enabled. Because we wish to print a
  // Profile Summary below, we need to tell the Engine at `prog` compile time to
  // retain this information via the `retainDebugInformation` option.
  auto engine = Engine{graph, prog, {{"debug.retainDebugInformation", "true"}}};
  engine.load(device);

  // Run the control program
  std::cout << "Running program\n";
  engine.run(0);
  std::cout << "Program complete\n";

  engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});

  return 0;
}
