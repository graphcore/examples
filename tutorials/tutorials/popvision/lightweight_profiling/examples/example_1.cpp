// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifdef __POPC__

#include <poplar/Vertex.hpp>
using namespace poplar;

class DummyVertex : public SupervisorVertex {
public:
  __attribute__((target("supervisor"))) bool compute() {
    // Simulate some computation
    for (volatile int i{}; i < 100; ++i)
      ;
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
  const unsigned NUM_IPUS = 2;
  Device device = getIpuHwDevice(NUM_IPUS);
  Target target = device.getTarget();
  Graph graph(target);

  // Compile the dummy vertex
  graph.addCodelets(__FILE__);

  const auto tilesPerIpu = device.getTarget().getTilesPerIPU();
  auto createOperation = [&](const std::string &name, const unsigned ipu) {
    ComputeSet op = graph.addComputeSet(name);
    VertexRef v = graph.addVertex(op, "DummyVertex");
    graph.setTileMapping(v, tilesPerIpu * ipu);
    // return Execute(op);
    return Block(Execute(op), name);
  };

  // Construct a fake pipeline with two stages and run it five times
  const auto stage1 = Block(
      Sequence{createOperation("op1", 0), createOperation("op2", 0)}, "stage1");
  const auto stage2 = Block(
      Sequence{createOperation("op1", 1), createOperation("op2", 1)}, "stage2");

  Repeat mainLoop{5, Sequence{stage1, stage2}};

  // Compile and execute the model
  OptionFlags engineOpts{{"profiler.blocks.filter", "stage.|op2"}};
  Engine engine(graph, mainLoop, engineOpts);
  engine.load(device);
  engine.run();

  // Use libpva to retrieve the blocks
  auto report = engine.getReport();
  for (unsigned ipu{}; ipu < NUM_IPUS; ++ipu) {
    const auto tile = ipu * tilesPerIpu;
    for (const auto &block : report.execution().blocks(tile)) {
      const auto from = block.cyclesFrom();
      const auto to = block.cyclesTo();
      std::cout << std::setw(10) << block.program()->name() << ": "
                << (to - from) << " (" << from << " - " << to << ")\n";
    }
  }

  return 0;
}
#endif
