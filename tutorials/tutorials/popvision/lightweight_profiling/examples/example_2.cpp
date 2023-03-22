// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifdef __POPC__

#include <poplar/Vertex.hpp>
using namespace poplar;

class DummyVertex : public SupervisorVertex {
  poplar::InOut<unsigned> iterations;

public:
  __attribute__((target("supervisor"))) bool compute() {
    for (unsigned i{}; i < iterations; ++i) {
      // Simulate some computation
      for (volatile int i{}; i < 100; ++i)
        ;
    }
    // Increase the number of iterations the next execution will have to do
    iterations += 1;
    return true;
  }
};

#else

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <pva/pva.hpp>

using namespace poplar;
using namespace poplar::program;

// Utility code to acquire a real IPU device
poplar::Device getIpuHwDevice(std::size_t numIpus) {
  auto dm = poplar::DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(poplar::TargetType::IPU, numIpus);
  auto it =
      std::find_if(hwDevices.begin(), hwDevices.end(),
                   [](poplar::Device &device) { return device.attach(); });
  if (it != hwDevices.end()) {
    return std::move(*it);
  }
  throw std::runtime_error("No IPU hardware available.");
}

int main(int argc, char **argv) {
  const unsigned NUM_IPUS = 1;
  Device device = getIpuHwDevice(NUM_IPUS);
  Target target = device.getTarget();
  Graph graph(target);

  // Compile the dummy vertex
  graph.addCodelets(__FILE__);

  // Create a compute set with a vertex that is connected
  // to a variable that regulates the internal iterations
  // and execute that compute set five times
  Tensor vIterations = graph.addVariable(UNSIGNED_INT, {});
  graph.setTileMapping(vIterations, 0);
  ComputeSet cs = graph.addComputeSet("operation");
  VertexRef v = graph.addVertex(cs, "DummyVertex");
  graph.setTileMapping(v, 0);

  graph.connect(v["iterations"], vIterations);
  graph.setInitialValue(vIterations, 1);

  Repeat loop{5, Execute(cs)};

  // Compile and execute the model
  OptionFlags engineOpts{
      {"profiler.programs.filter", "operation"},
  };
  Engine engine(graph, loop, engineOpts);
  engine.load(device);
  engine.run();

  // Use libpva to retrieve the blocks
  auto report = engine.getReport();
  const unsigned tile{0};
  for (const auto &block : report.execution().blocks(tile)) {
    const auto cycles = block.cyclesTo() - block.cyclesFrom();
    std::cout << block.program()->name() << ": " << cycles << "\n";
  }

  return 0;
}
#endif
