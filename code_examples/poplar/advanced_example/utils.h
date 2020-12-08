// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>

#include <boost/program_options.hpp>

namespace utils {

struct Options {
  std::size_t numIpus;
  std::string exeName;
  std::string profileName;
  bool useIpuModel;
  bool saveExe;
  bool loadExe;
  bool profile;
};

inline
Options parseOptions(int argc, char** argv) {
  Options options;
  std::string modeString;

  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
  ("help", "Show command help.")
  ("model",
   po::bool_switch(&options.useIpuModel)->default_value(false),
   "If set then use IPU model instead of hardware."
  )
  ("ipus",
   po::value<std::size_t>(&options.numIpus)->default_value(1),
   "Number of IPUs to use."
  )
  ("exe-name",
   po::value<std::string>(&options.exeName)->default_value(""),
   "Save the graph executable under a file name with this prefix. "
   "This option is required when loading/saving executables."
  )
  ("save-exe",
   po::bool_switch(&options.saveExe)->default_value(false),
   "Save the Poplar graph executable after compilation. "
   "You must also set 'exe-name'."
  )
  ("load-exe",
   po::bool_switch(&options.loadExe)->default_value(false),
   "Load a previosuly saved executable and skip graph and program construction. "
   "You must also set 'exe-name'."
  )
  ("profile",
   po::bool_switch(&options.profile)->default_value(false),
   "Enable profile output."
  )
  ("profile-name",
   po::value<std::string>(&options.profileName)->default_value("profile.txt"),
   "Name of profile output file."
  );

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    throw std::runtime_error("Show help");
  }

  po::notify(vm);

  // Check options are set correctly:
  if ((options.saveExe || options.loadExe) && options.exeName.empty()) {
    throw std::logic_error("To save/load an executable you must set 'exe-name'");
  }

  if (options.loadExe && options.profile) {
    throw std::logic_error("It is not possible to run profiling on a loaded executable.");
  }

  return options;
}

inline
std::string getExeFileName(const Options &options) {
  return options.exeName + ".poplar";
}

inline
poplar::Executable compileOrLoadExe(
    poplar::Graph &graph,
    const std::vector<poplar::program::Program> &progs,
    const Options &options) {
  if (options.loadExe) {
    const auto exeName = getExeFileName(options);
    try {
      auto inf = std::ifstream(exeName);
      return poplar::Executable::deserialize(inf);
    } catch (const poplar::poplar_error& e) {
      std::cerr << "Error: Failed to load executable '" << exeName << "'\n";
      throw;
    }
  } else {
    return poplar::compileGraph(graph, progs);
  }
}

// Return a HW device with the requested number of IPUs.
// Exception is thrown if no devices with the requested
// number are available.
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

inline
poplar::Device getDeviceFromOptions(const Options& options) {
    poplar::Device device;
    if(options.useIpuModel) {
      device = getIpuModelDevice(options.numIpus);
      std::cerr << "Using IPU model\n";
    } else {
      device = getIpuHwDevice(options.numIpus);
      std::cerr << "Using HW device ID: " << device.getId() << "\n";
    }
    return device;
}

} // end namespace utils
