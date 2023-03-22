// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

/* This file contains the completed version of Poplar tutorial 1
   which uses the IPU Model.
   See the Poplar user guide for details
*/

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <iostream>

using namespace poplar;

int main() {
  // Create the IPU model device
  IPUModel ipuModel;
  Device device = ipuModel.createDevice();
  Target target = device.getTarget();

  // Create the Graph object
  Graph graph(target);

  // Add variables to the graph
  Tensor v1 = graph.addVariable(FLOAT, {4}, "v1");
  Tensor v2 = graph.addVariable(FLOAT, {4}, "v2");
  Tensor v3 = graph.addVariable(FLOAT, {4, 4}, "v3");
  Tensor v4 = graph.addVariable(INT, {10}, "v4");

  // Allocate v1 to reside on tile 0
  graph.setTileMapping(v1, 0);

  // Spread v2 over tiles 0..3
  for (unsigned i = 0; i < 4; ++i)
    graph.setTileMapping(v2[i], i);

  // Allocate v3, t4 to tile 0
  graph.setTileMapping(v3, 0);
  graph.setTileMapping(v4, 0);

  // Create a control program that is a sequence of steps
  program::Sequence prog;

  // Add a constant tensor to the graph
  Tensor c1 = graph.addConstant<float>(FLOAT, {4}, {1.0, 1.5, 2.0, 2.5});
  graph.setTileMapping(c1, 0);

  // Add a step to initialize v1 with the constant value in c1
  prog.add(program::Copy(c1, v1));
  // Debug print the tensor to the host console
  prog.add(program::PrintTensor("v1-debug", v1));

  // Copy the data in v1 to v2
  prog.add(program::Copy(v1, v2));
  // Debug print v2
  prog.add(program::PrintTensor("v2-debug", v2));

  // Create host read/write handles for v3
  graph.createHostWrite("v3-write", v3);
  graph.createHostRead("v3-read", v3);

  // Copy a slice of v1 into v3
  Tensor v1slice = v1.slice(0, 3);
  Tensor v3slice = v3.slice({1, 1}, {2, 4});
  prog.add(program::Copy(v1slice, v3slice));

  // Add a data stream to fill v4
  DataStream inStream = graph.addHostToDeviceFIFO("v4-input-stream", INT, 10);

  // Add program steps to copy from the stream
  prog.add(program::Copy(inStream, v4));
  prog.add(program::PrintTensor("v4-0", v4));
  prog.add(program::Copy(inStream, v4));
  prog.add(program::PrintTensor("v4-1", v4));

  // Create the engine
  Engine engine(graph, prog);
  engine.load(device);

  // Copy host data via the write handle to v3 on the device
  std::vector<float> h3(4 * 4, 0);
  engine.writeTensor("v3-write", h3.data(), h3.data() + h3.size());

  // Create a buffer to hold data to be fed via the data stream
  std::vector<int> inData(10 * 3);
  for (unsigned i = 0; i < 10 * 3; ++i)
    inData[i] = i;

  // Connect the data stream
  engine.connectStream("v4-input-stream", &inData[0], &inData[10 * 3]);

  // Run the control program
  std::cout << "Running program\n";
  engine.run(0);
  std::cout << "Program complete\n";

  // Copy v3 back to the host via the read handle
  engine.readTensor("v3-read", h3.data(), h3.data() + h3.size());

  // Output the copied back values of v3
  std::cout << "\nh3 data:\n";
  for (unsigned i = 0; i < 4; ++i) {
    std::cout << "  ";
    for (unsigned j = 0; j < 4; ++j) {
      std::cout << h3[i * 4 + j] << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
