// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>

#include <pvti/pvti.hpp>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace utils {

/// Return the application's shared logger object.
std::shared_ptr<spdlog::logger> logger() {
  static auto logger = spdlog::stdout_logger_mt("ipu_trace_logger");
  return spdlog::get("ipu_trace_logger");
}

inline std::string makeExeFileName(const std::string& name) {
    return name + ".poplar.exe";
}

inline std::string makeProgramsFileName(const std::string& name) {
    return name + ".poplar.progs";
}

poplar::Executable loadExe(const std::string& name) {
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

void saveExe(const poplar::Executable& exe, const std::string& name) {
  const auto fileName = makeExeFileName(name);
  auto outf = std::ofstream(fileName);
  exe.serialize(outf);
  logger()->info("Saved Poplar executable as: '{}'", fileName);
}

// Return a HW device with the requested number of IPUs.
// If no devices with the requested number are available
// then an exception is thrown.
inline
poplar::Device getIpuHwDevice(std::size_t numIpus) {
  auto dm = poplar::DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(poplar::TargetType::IPU, numIpus);
  if (hwDevices.size() > 0) {
    for (auto &d : hwDevices) {
      if (d.attach()) {
        return std::move(d);
      }
    }
  }
  throw std::runtime_error("No IPU hardware available.");
}

// Return an IPU Model device with the requested number of IPUs.
inline
poplar::Device getIpuModelDevice(std::size_t numIpus) {
  poplar::IPUModel ipuModel;
  ipuModel.numIPUs = numIpus;
  return ipuModel.createDevice();
}

struct RuntimeConfig {
  std::size_t numIpus;
  std::string exeName;
  bool useIpuModel;
  bool saveExe;
  bool loadExe;
  bool compileOnly;
};

/// Determine whether to acquire a HW device or IPU model,
/// and number of IPUs for either, from the relevant command
/// line options.
inline poplar::Device getDeviceFromConfig(RuntimeConfig config) {
    poplar::Device device;
    if(config.useIpuModel) {
      device = getIpuModelDevice(config.numIpus);
      logger()->info("Using IPU model");
    } else {
      device = getIpuHwDevice(config.numIpus);
      logger()->info("Using IPU device ID: {}", device.getId());
    }
    auto ipus = device.getTarget().getNumIPUs();
    auto tilesPerIpu = device.getTarget().getTilesPerIPU();
    auto workers = device.getTarget().getNumWorkerContexts();
    bool remoteBuffers = device.supportsRemoteBuffers();
    logger()->info("IPUs {}, tiles {}, workers {}, remote-buffers-supported: {}",
                   ipus, ipus * tilesPerIpu, workers, remoteBuffers);
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

/// Utility for managing Poplar data streams. (It is a stream name and handle pair).
struct StreamRecord {
  StreamRecord(const std::string& s) : name(s) {}

  void buildWriteToTensor(poplar::Graph& graph, poplar::Tensor t) {
    stream = graph.addHostToDeviceFIFO(name, t.elementType(), t.numElements());
  }

  void buildReadFromTensor(poplar::Graph& graph, poplar::Tensor t) {
    stream = graph.addDeviceToHostFIFO(name, t.elementType(), t.numElements());
  }

  const std::string name;
  poplar::DataStream stream;
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
  virtual void build(poplar::Graph& graph, const poplar::Device& device) = 0;
  virtual void execute(poplar::Engine& engine, const poplar::Device& device) = 0;

  /// The default getDevice() creates a device from the config. This
  /// can be overridden by derived classes if more sophisticated behaviour
  /// is needed.
  virtual poplar::Device getDevice() {
    return getDeviceFromConfig(getRuntimeConfig());
  }
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
      pvti::TraceChannel traceChannel = {"utils::GraphManager"};

      auto config = builder.getRuntimeConfig();
      auto device = builder.getDevice();
      poplar::Graph graph(device.getTarget());

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
        executeGraphProgram(exe, device, builder);
      } else {
        // Otherwise we must build and compile the graph:
        logger()->info("Graph construction started");
        pvti::Tracepoint::begin(&traceChannel, "constructing_graph");
        builder.build(graph, device);
        pvti::Tracepoint::end(&traceChannel, "constructing_graph");
        logger()->info("Graph construction finished");

        logger()->info("Graph compilation started");
        pvti::Tracepoint::begin(&traceChannel, "compiling_graph");
        poplar::Executable exe = poplar::compileGraph(graph, builder.getPrograms().getList());
        pvti::Tracepoint::end(&traceChannel, "compiling_graph");
        logger()->info("Graph compilation finished");

        if (config.saveExe) {
          saveExe(exe, config.exeName);
          std::ofstream fs(makeProgramsFileName(config.exeName));
          // Need to serialise the ProgramManager also:
          builder.getPrograms().serialise(fs);
        }

        if (config.compileOnly) {
            utils::logger()->info("Compile only mode selected: finished.");
            return EXIT_SUCCESS;
        }

        // Run the graph we just built and compiled.
        executeGraphProgram(exe, device, builder);
      }

    } catch (const std::exception& e) {
      utils::logger()->error("Exception: {}", e.what());
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

private:
  void executeGraphProgram(poplar::Executable& exe, poplar::Device& device,
                           BuilderInterface& builder) {
    // Prepare the execution engine and connect
    // data streams to/from IPU:
    poplar::Engine engine(std::move(exe));
    engine.load(device);
    builder.execute(engine, device);
  }
};

} // end namespace utils
