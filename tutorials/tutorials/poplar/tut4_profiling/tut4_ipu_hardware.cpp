// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

/* This file contains the completed version of Poplar tutorial 4,
  which uses the IPU Hardware.
*/

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <algorithm>
#include <iostream>

using namespace poplar;
using namespace poplar::program;

int main() {
  // Create the DeviceManager which is used to discover devices
  auto manager = DeviceManager::createDeviceManager();

  // Attempt to attach to a single IPU:
  auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
  std::cout << "Trying to attach to IPU\n";
  auto it = std::find_if(devices.begin(), devices.end(),
                         [](Device &device) { return device.attach(); });

  if (it == devices.end()) {
    std::cerr << "Error attaching to device\n";
    return 1; // EXIT_FAILURE
  }

  auto device = std::move(*it);
  std::cout << "Attached to IPU " << device.getId() << std::endl;

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

  // Create the engine. We instruct the engine to perform instrumentation - this
  // adds cycle counters to the compiled program to enable the execution profile
  // to be retrieved after the program is run.
  auto engine = Engine{graph, prog, {{"debug.instrument", "true"}}};
  engine.load(device);

  // Run the control program
  std::cout << "Running program\n";
  engine.run(0);
  std::cout << "Program complete\n";

  engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});

  return 0;
}
