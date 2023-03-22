// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifdef __POPC__

#include <poplar/Vertex.hpp>
using namespace poplar;

class DummyVertex : public SupervisorVertex {
  poplar::Input<int> input;

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

// Simple callback that sends the contents of an array from
// the host to the IPU
struct HostToDeviceCallback : public LegacyStreamCallback {
  poplar::ArrayRef<int> in;
  HostToDeviceCallback(poplar::ArrayRef<int> in) : in{in} {}
  void fetch(void *p) override {
    std::copy(in.begin(), in.end(), static_cast<int *>(p));
  }
};

std::string getType(const std::shared_ptr<pva::Program> prog) {
  if (prog->type() == pva::Program::Type::StreamCopyBegin) {
    return "StreamCopyBegin";
  }
  if (prog->type() == pva::Program::Type::StreamCopyMid) {
    return "StreamCopyMid";
  }
  if (prog->type() == pva::Program::Type::StreamCopyEnd) {
    return "StreamCopyEnd";
  }
  throw std::runtime_error("Program is not a StreamCopy phase");
}

int main(int argc, char **argv) {
  const unsigned NUM_IPUS = 1;
  Device device = getIpuHwDevice(NUM_IPUS);
  Target target = device.getTarget();
  Graph graph(target);

  // Compile the dummy vertex
  graph.addCodelets(__FILE__);

  const unsigned computeTile{0};
  const unsigned ioTile{100};

  ComputeSet op = graph.addComputeSet("operation");
  VertexRef v = graph.addVertex(op, "DummyVertex");
  graph.setTileMapping(v, computeTile);

  // Variable to receive input data from the host
  Tensor in1 = graph.addVariable(INT, {});
  // Variable to compute with
  Tensor in2 = graph.addVariable(INT, {});
  graph.setTileMapping(in1, ioTile);
  graph.setTileMapping(in2, computeTile);

  graph.connect(v["input"], in2);

  auto inStream = graph.addHostToDeviceFIFO("in", INT, 1);

  Sequence seq{
      Copy(inStream, in1),
      Repeat{4, Sequence{Copy(in1, in2), Copy(inStream, in1), Execute(op)}},
      Copy(in1, in2), Execute(op)};

  // Compile the model
  OptionFlags engineOpts{{"profiler.programs.filter", "operation"},
                         {"debug.instrumentExternalExchange", "true"},
                         {"profiler.blocks.implicitFlush", "false"}};
  Engine engine(graph, seq, engineOpts);

  // Dummy callback that sends zeroes
  const int value{0};
  auto callback =
      std::make_unique<HostToDeviceCallback>(poplar::ArrayRef<int>(&value, 1));
  engine.connectStreamToCallback("in", 0, std::move(callback));

  // Execute the model
  engine.load(device);
  engine.run();

  // Use libpva to retrieve the blocks
  auto report = engine.getReport();
  for (const unsigned tile : {computeTile, ioTile}) {
    for (const auto &block : report.execution().blocks(tile)) {
      const auto from = block.cyclesFrom();
      const auto to = block.cyclesTo();
      if (block.isStreamCopy()) {
        std::cout << std::setw(20) << getType(block.program());
      } else {
        std::cout << std::setw(20) << block.program()->name();
      }
      std::cout << " in tile " << std::setw(4) << block.tile() << ": "
                << std::setw(8) << (to - from) << " (" << from << " - " << to
                << ")\n";
    }
  }

  return 0;
}
#endif
