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
  const unsigned NUM_IPUS = 1;
  Device device = getIpuHwDevice(NUM_IPUS);
  Target target = device.getTarget();
  Graph graph(target);

  // Compile the dummy vertex
  graph.addCodelets(__FILE__);

  const unsigned computeTile{0};

  // Construct a compute set that gets executed in a simple loop
  ComputeSet op = graph.addComputeSet("operation");
  VertexRef v = graph.addVertex(op, "DummyVertex");
  graph.setTileMapping(v, computeTile);

  Sequence seq{Repeat{100, Sequence{Execute(op)}}};

  // Compile and execute the model
  OptionFlags engineOpts{
      {"profiler.programs.filter", "operation"},
      {"debug.instrumentExternalExchange", "true"},
      {"profiler.blocks.implicitFlush", "true"},
  };
  // Make sure the engine is destroyed before accessing the report
  auto report = [&]() {
    Engine engine(graph, seq, engineOpts);
    engine.load(device);
    engine.run();
    return engine.getReport();
  }();

  // Use libpva to count the blocks
  unsigned numCommonBlocks{};
  unsigned numStreamCopies{};
  unsigned numOverflows{};
  unsigned numFlushes{};
  unsigned numBlocks{};
  for (const auto &block : report.execution().blocks(computeTile)) {
    if (block.isCommon()) {
      ++numCommonBlocks;
    } else if (block.isOverflow()) {
      ++numOverflows;
    } else if (block.isBlockFlush()) {
      ++numFlushes;
    } else if (block.isStreamCopy()) {
      ++numStreamCopies;
    }
    ++numBlocks;
  }
  std::cout << "numBlocks: " << numBlocks << "\n";
  std::cout << "numCommonBlocks: " << numCommonBlocks << "\n";
  std::cout << "numFlushes: " << numFlushes << "\n";
  std::cout << "numStreamCopies: " << numStreamCopies << "\n";
  std::cout << "numOverflows: " << numOverflows << "\n";

  return 0;
}
#endif
