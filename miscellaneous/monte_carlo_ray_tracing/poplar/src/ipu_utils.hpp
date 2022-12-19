// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <functional>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>

#include <pvti/pvti.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "logging.hpp"

namespace ipu_utils {

/// Return the application's shared logger object.
inline std::shared_ptr<spdlog::logger> logger() {
  static auto logger = spdlog::stdout_logger_mt("ipu_trace_logger");
  return spdlog::get("ipu_trace_logger");
}

inline std::string makeExeFileName(const std::string& name) {
    return name + ".poplar.exe";
}

inline std::string makeProgramsFileName(const std::string& name) {
    return name + ".poplar.progs";
}

inline poplar::Executable loadExe(const std::string& name) {
  const auto exeName = makeExeFileName(name);
  logger()->info("Loading precompiled graph from: '{}'", exeName);
  try {
    auto inf = std::ifstream(exeName);
    return poplar::Executable::deserialize(inf);
  } catch (const poplar::poplar_error& e) {
    logger()->error("Error: Failed to load executable: '{}'", exeName);
    throw;
  }
}

inline void saveExe(const poplar::Executable& exe, const std::string& name) {
  const auto fileName = makeExeFileName(name);
  auto outf = std::ofstream(fileName);
  exe.serialize(outf);
  logger()->info("Saved Poplar executable as: '{}'", fileName);
}

/// Abstract Interface for a device within the builder framework.
class DeviceInterface {
public:
  DeviceInterface() {}
  virtual ~DeviceInterface() {}
  virtual const poplar::Target& getTarget() = 0;
  virtual const poplar::Device& getPoplarDevice() = 0;
  virtual void attach() = 0;
  virtual void detach() = 0;
};

/// Utility class for managing delayed attach to hardware devices.
class DeferredDevice : public DeviceInterface {
public:
  /// Create a device specifying whether hardware attach should
  /// be immediate (defer = false) or deferred (defer = true).
  DeferredDevice(bool defer) : attached (nullptr), deferredAttach(defer) {}
  virtual ~DeferredDevice() {}

  /// Create a device with specified IPU model configuration.
  void getIpuModel(std::size_t numIpus) {
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = numIpus;
    devices.clear();
    devices.push_back(ipuModel.createDevice());
    logger()->info("Using IPU model");
    // IPU models do not need to attach so always record as attached:
    attached = &devices.front();
  }

  /// Create a lazy device with specified hardware IPU configuration. If no
  /// device with the requested config is available then an exception is thrown.
  void getIpuHardware(std::size_t numIpus) {
    auto dm = poplar::DeviceManager::createDeviceManager();
    devices = dm.getDevices(poplar::TargetType::IPU, numIpus);
    if (devices.empty()) {
      logger()->error("No devices found with {} ipus.", numIpus);
      throw std::runtime_error("No IPU hardware with requested configuration.");
    }
    logger()->info("Found {} compatible IPU devices.", devices.size());
    if (deferredAttach == false) {
      attach();
    }
  }

  const poplar::Target& getTarget() override {
    if (attached) {
      return attached->getTarget();
    } else {
      if (devices.empty()) {
        logger()->error("Device list was empty during attempt to attach.");
        throw std::runtime_error("Could not attach to an IPU device.");
      }
      return devices.front().getTarget();
    }
  }

  const poplar::Device& getPoplarDevice() override {
    if (attached) {
      return *attached;
    }
    logger()->error("No device attached ({} available).", devices.size());
    throw std::runtime_error("No device attached.");
  }

  void attach() override {
    if (attached) { return; }

    for (auto &d : devices) {
      if (d.attach()) {
        auto ipus = d.getTarget().getNumIPUs();
        auto tilesPerIpu = d.getTarget().getTilesPerIPU();
        auto workers = d.getTarget().getNumWorkerContexts();
        bool remoteBuffers = d.supportsRemoteBuffers();
        logger()->info("Attached to device with {} IPUs, {} tiles, {} workers. remote-buffers-supported: {}", ipus, ipus * tilesPerIpu, workers, remoteBuffers);
        attached = &d;
        return;
      }
    }
    logger()->error("None of the {} IPU devices were available.", devices.size());
    throw std::runtime_error("Could not attach to an IPU device.");
  }

  void detach() override {
    if (attached) {
      attached->detach();
      attached = nullptr;
    }
  }

private:
  std::vector<poplar::Device> devices;
  poplar::Device* attached;
  bool deferredAttach;
};

struct RuntimeConfig {
  std::size_t numIpus;
  std::string exeName;
  bool useIpuModel;
  bool saveExe;
  bool loadExe;
  bool compileOnly;
  bool deferredAttach;
};

/// Determine whether to acquire a HW device or IPU model, and number of IPUs
/// for either, from the relevant command line options.
inline
std::unique_ptr<DeviceInterface> getDeviceFromConfig(RuntimeConfig config) {
    std::unique_ptr<DeferredDevice> device(new DeferredDevice(config.deferredAttach));
    if(config.useIpuModel) {
      device->getIpuModel(config.numIpus);
    } else {
      device->getIpuHardware(config.numIpus);
    }
    return device;
}

using ProgramList = std::vector<poplar::program::Program>;
using ProgramMap = std::map<std::string, poplar::program::Program>;
using ProgramOrdinals = std::map<std::string, int>;

/// Class which manages the set of programs for a Poplar graph.
/// Hides the details of managing program ordinal IDs by providing
/// a consistent mapping from program name strings to ordinals.
class ProgramManager {
  ProgramMap map;
  ProgramOrdinals ordinals;

public:
  /// Register a program with a given name with the manager.
  void add(const std::string& name, poplar::program::Program prog) {
    map.insert(std::make_pair(name, prog));
  }

  /// Check the named program exists and run it.
  void run(poplar::Engine& e, const std::string progName) const {
    auto found = ordinals.find(progName);
    if (found == ordinals.cend()) {
      throw std::runtime_error("No such program: '" + progName + "'");
    }
    e.run(found->second);
  }

  /// Get the list of programs ordered by their ordinals (e.g. pass
  /// the result to poplar::compileGraph).
  ProgramList getList() {
    ProgramList list;
    int i = 0;
    list.reserve(map.size());
    for (const auto& p : map) {
      list.push_back(p.second);
      // Construct a consistent set of ordinals to go with the list:
      ordinals.insert(std::make_pair(p.first, i));
      i += 1;
    }

    return list;
  }

  /// Return a map from strings to ordinals. Use this to call a program by its
  /// string name e.g. `engine.run(progs.getOrdinals().at("prog_name"))`
  ProgramOrdinals getOrdinals() const {
    if (ordinals.empty()) {
      throw std::logic_error("Program ordinals map is empty. "
                             "Did you call getList() or deserialise() first?");
    }
    return ordinals;
  }

  void serialise(std::ostream& os) const {
    boost::property_tree::ptree progs;
    const auto ordinals = getOrdinals();
    for (const auto& p : ordinals) {
      progs.put(p.first, std::to_string(p.second));
    }
    boost::property_tree::ptree root;
    root.add_child("programs", progs);
    boost::property_tree::write_json(os, root);
  }

  void deserialise(std::istream& is) {
    boost::property_tree::ptree root;
    boost::property_tree::read_json(is, root);
    auto progs = root.get_child("programs");
    logger()->info("Loading program list:");
    ordinals.clear();
    for (auto& p : progs) {
      std::string name = p.first;
      auto ordinal = progs.get<unsigned>(p.first);
      logger()->info("Program: {}: {}", name, ordinal);
      ordinals.insert(std::make_pair(name, ordinal));
    }
  }
};

/// Utility to connect a stream to a std::vector.
template <class T>
void connectStream(poplar::Engine& e, const std::string& handle, std::vector<T>& v) {
  e.connectStream(handle, v.data(), v.data() + v.size());
}

/// Utility for managing a tensor and its IO streams. We need to retain the handle
/// name to use when we come to execute the graph (and rebuild it when loading an
/// exe) so this encapsulates the necessary string management, which otherwise
/// becomes unweildy.
struct StreamableTensor {
  StreamableTensor(const std::string& s) : name(s) {}

  std::string getWriteHandle() const { return name + "/write_stream"; }
  std::string getReadHandle() const { return name + "/read_stream"; }

  template<typename... Args>
  void buildTensor(poplar::Graph& graph,
                   poplar::Type type, poplar::ArrayRef<std::size_t> shape,
                   poplar::VariableMappingMethod mapping = poplar::VariableMappingMethod::NONE) {
    checkedAssign(graph.addVariable(type, shape, mapping, name + "/tensor"));
  }

  const poplar::Tensor& operator = (poplar::Tensor&& t) {
    checkedAssign(t);
    return tensor;
  }

  // Create a host to device stream and return a copy program that writes to it.
  poplar::program::Program buildWrite(poplar::Graph& graph, bool optmiseMemory) {
    if (!tensor.valid()) {
      throw std::logic_error("Tensor must be assigned before calling buildWrite().");
    }
    writeStream = graph.addHostToDeviceFIFO(getWriteHandle(), tensor.elementType(), tensor.numElements());
    return poplar::program::Copy(writeStream, tensor, optmiseMemory, name + "/write");
  }

  // Create a device to host stream and return a copy program that reads from it.
  poplar::program::Program buildRead(poplar::Graph& graph, bool optmiseMemory) {
    if (!tensor.valid()) {
      throw std::logic_error("Tensor must be assigned before building buildRead().");
    }
    readStream = graph.addDeviceToHostFIFO(getReadHandle(), tensor.elementType(), tensor.numElements());
    return poplar::program::Copy(tensor, readStream, optmiseMemory, name + "/read");
  }

  void connectWriteStream(poplar::Engine& e, void* data) const {
    e.connectStream(getWriteHandle(), data);
  }

  void connectReadStream(poplar::Engine& e, void* data) const {
    e.connectStream(getReadHandle(), data);
  }

  std::size_t numElements() const { return get().numElements(); }
  poplar::Type elementType() const { return get().elementType(); }
  std::vector<std::size_t> shape() const { return get().shape(); }
  operator poplar::Tensor() { return get(); } // Overload cast to poplar::Tensor
  const poplar::Tensor& get() const {
    if (!tensor.valid()) {
      throw std::logic_error("Tensor must be assigned before access.");
    }
    return tensor;
  }

private:
  void checkedAssign(poplar::Tensor t) {
    if (tensor.valid()) {
      throw std::logic_error("StreamableTensor may only assigned once.");
    }
    tensor = t;
  }

  const std::string name;
  poplar::DataStream writeStream;
  poplar::DataStream readStream;
  poplar::Tensor tensor;
};

/// Abstract interface for graph builders. A class that implements the
/// BuilderInterface must be passed to GraphManager that will then call
/// the interface methods at the correct time to build/compile/load/execute
/// the graph.
struct BuilderInterface {
  virtual RuntimeConfig getRuntimeConfig() const = 0;
  virtual ProgramManager& getPrograms() = 0;

  /// Methods below are private as they should only be called by the GraphManager.
private:
  friend class GraphManager;
  virtual void build(poplar::Graph& graph, const poplar::Target& target) = 0;
  virtual void execute(poplar::Engine& engine, const poplar::Device& device) = 0;

  /// The default implementation creates a device from the config. This
  /// can be overridden by derived classes if more sophisticated behaviour
  /// is needed.
  virtual std::unique_ptr<DeviceInterface> getDevice() {
    return getDeviceFromConfig(getRuntimeConfig());
  }
};

/// Utility class that can be used to wrap a poplar::Engine::ProgressFunc
/// callback in a filter that reduces the amount of output produced.
class CallbackFilter {

public:
  CallbackFilter(std::function<void(int, int)> progressFunc, int stageGap = 15, double timeGap = 10.0)
    : wrappedCallback(progressFunc),
      stageFilterCount(stageGap), timeFilterSecs(timeGap),
      lastStage(0), lastTime(std::chrono::system_clock::now()) {}

  // Return the filtered callback function:
  std::function<void(int,int)> getFilteredCallback() {
    return std::bind(&CallbackFilter::callback, this, std::placeholders::_1, std::placeholders::_2);
  }

private:
  // Callback that can be used as a poplar::Engine::ProgressFunc (via std::bind):
  void callback(int done, int todo) {
    if (done == 0) {
      // Reset in case this ever gets used in multiple calls to compile:
      lastStage = 0;
      lastTime = std::chrono::system_clock::now();
    }

    // If nothing has been reported for some time print
    // the progress regardless of the other filtering:
    auto currentTime = std::chrono::system_clock::now();
    double secs = std::chrono::duration<double>(currentTime - lastTime).count();

    // Filters to reduce excessive logging:
    if (done - lastStage > stageFilterCount || secs > timeFilterSecs || done == todo) {
      lastStage = done;
      lastTime  = currentTime;
      wrappedCallback(done, todo);
    }
  }

  std::function<void(int, int)> wrappedCallback;

  // Member variables used in filtering the callback rate:
  const int stageFilterCount;
  const double timeFilterSecs;
  int lastStage;
  std::chrono::time_point<std::chrono::system_clock> lastTime;
};

/// Class that manages graph creation, engine creation and execution, and provides a consistent interface
/// for saving and loading graph executables. In effect it manages all the host side and runtime activity
/// of a Poplar program.
class GraphManager {
public:
  GraphManager() {}

  /// Run the graph (i.e. build, then compile (or load) and execute the graph.
  /// Takes a reference to an object that implements the builder interface. The
  /// builder object completely describes the Poplar program to build and run.
  /// Returns the exit code for the host program.
  int run(BuilderInterface& builder) {
    try {
      pvti::TraceChannel traceChannel = {"ipu_utils::GraphManager"};

      auto config = builder.getRuntimeConfig();
      auto device = builder.getDevice();
      poplar::Graph graph(device->getTarget());

      if (config.loadExe) {
        // When loading, we simply load-construct the executable and run it:
        pvti::Tracepoint::begin(&traceChannel, "loading_graph");
        poplar::Executable exe = loadExe(config.exeName);
        // Need to load a ProgramManager also:
        auto progsFileName = makeProgramsFileName(config.exeName);
        try {
          std::ifstream fs(progsFileName);
          builder.getPrograms().deserialise(fs);
        } catch (const std::exception& e) {
          logger()->error("Error: failed to load program list from '{}'", progsFileName);
          throw;
        }
        pvti::Tracepoint::end(&traceChannel, "loading_graph");
        executeGraphProgram(exe, *device, builder);
      } else {
        // Otherwise we must build and compile the graph:
        logger()->info("Graph construction started");
        pvti::Tracepoint::begin(&traceChannel, "constructing_graph");
        builder.build(graph, device->getTarget());
        pvti::Tracepoint::end(&traceChannel, "constructing_graph");
        logger()->info("Graph construction finished");

        logger()->info("Graph compilation started");
        pvti::Tracepoint::begin(&traceChannel, "compiling_graph");
        CallbackFilter progress([] (int done, int todo) {
          logger()->debug("Compilation step {}/{}", done, todo);
        });
        poplar::Executable exe = poplar::compileGraph(graph, builder.getPrograms().getList(), {},
                                                      progress.getFilteredCallback(), "ipu_utils_engine");
        pvti::Tracepoint::end(&traceChannel, "compiling_graph");
        logger()->info("Graph compilation finished");

        if (config.saveExe) {
          saveExe(exe, config.exeName);
          std::ofstream fs(makeProgramsFileName(config.exeName));
          // Need to serialise the ProgramManager also:
          builder.getPrograms().serialise(fs);
        }

        if (config.compileOnly) {
            ipu_utils::logger()->info("Compile only mode selected: finished.");
            return EXIT_SUCCESS;
        }

        // Run the graph we just built and compiled.
        executeGraphProgram(exe, *device, builder);
      }

    } catch (const std::exception& e) {
      ipu_utils::logger()->error("Exception: {}", e.what());
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

private:
  void executeGraphProgram(poplar::Executable& exe,
                           DeviceInterface& device,
                           BuilderInterface& builder) {
    // Prepare the execution engine and connect
    // data streams to/from IPU:
    poplar::Engine engine(std::move(exe));
    device.attach();
    engine.load(device.getPoplarDevice());
    builder.execute(engine, device.getPoplarDevice());
  }
};

} // end namespace utils
