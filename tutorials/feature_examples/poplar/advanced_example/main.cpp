// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <cstdlib>
#include <vector>

#include "utils.h"

#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>

enum Progs {
  WRITE_INPUTS,
  CUSTOM_PROG,
  REDUCTION_PROG,
  READ_RESULTS,
  NUM_PROGRAMS
};

std::vector<poplar::program::Program>
buildGraphAndPrograms(poplar::Graph &g, const utils::Options &options) {
  // Use the namespace here to make graph construction code less verbose:
  using namespace poplar;

  // Create some tensor variables. In this case they are just vectors:
  Tensor v1 = g.addVariable(FLOAT, {4}, "a");
  Tensor v2 = g.addVariable(FLOAT, {4}, "b");
  Tensor v3 = g.addVariable(FLOAT, {4}, "c");

  // Variables need to be explcitly mapped to tiles. Put each variable on
  // a different tile (regardless of whether it is sensible in this case):
  g.setTileMapping(v1, 0);
  g.setTileMapping(v2, 1);
  g.setTileMapping(v3, 2);

  // In order to do any computation we need a compute set and a compute
  // vertex that is placed in that compute set:
  ComputeSet cs1 = g.addComputeSet("cs1");

  // Before we can add a custom vertex to the graph we need to load its
  // code. NOTE: .gp files are precompiled codelets but we could also
  // have loaded and compiled source directly here:
  g.addCodelets("codelets.gp"); // g.addCodelets("codelets.cpp");
  auto v = g.addVertex(cs1, "VectorAdd");

  // Vertices must also be mapped to tiles. This computation will
  // run on tile 0. Exchanges will automatically be generated to
  // get the inputs and results to the correct locations:
  g.setTileMapping(v, 0);

  // Next connect the variables to the vertex's fields by name:
  g.connect(v["x"], v1);
  g.connect(v["y"], v2);
  g.connect(v["z"], v3);

  if (options.useIpuModel) {
    // For generating IPUmodel profiles a cycle estimate must
    // be set for each custom vertex:
    g.setPerfEstimate(v, 20);
  }

  // Create streams that allow reading and writing of the variables:
  auto stream1 = g.addHostToDeviceFIFO("write_x", FLOAT, v1.numElements());
  auto stream2 = g.addHostToDeviceFIFO("write_y", FLOAT, v2.numElements());
  auto stream3 = g.addHostToDeviceFIFO("write_z", FLOAT, v3.numElements());
  auto stream4 = g.addDeviceToHostFIFO("read_z", FLOAT, v3.numElements());

  // Add a second compute set that will perform the same calculation using
  // Poplib's reduction API:
  popops::addCodelets(g);
  std::vector<ComputeSet> reductionSets;
  // Concatenate the vectors into a matrix so we can reduce one axis:
  auto tc = poplar::concat(v1.reshape({v1.numElements(), 1}),
                           v2.reshape({v2.numElements(), 1}), {1});
  auto reduce = popops::ReduceParams(popops::Operation::ADD);
  // Create compute sets that perform the reduction, storing th eresult in v3:
  popops::reduceWithOutput(g, tc, v3, {1}, reduce, reductionSets, "reduction");

  // Now can start constructing the programs. Construct a vector of
  // separate programs that can be called individually:
  std::vector<program::Program> progs(Progs::NUM_PROGRAMS);

  // Add program which initialises the inputs. Poplar is able to merge these
  // copies for efficiency:
  progs[WRITE_INPUTS] =
      program::Sequence({program::Copy(stream1, v1), program::Copy(stream2, v2),
                         program::Copy(stream3, v3)});

  // Program that executes custom vertex in compute set 1:
  progs[CUSTOM_PROG] = program::Execute(cs1);

  // Program that executes all the reduction compute sets:
  auto seq = program::Sequence();
  for (auto &cs : reductionSets) {
    seq.add(program::Execute(cs));
  }
  progs[REDUCTION_PROG] = seq;

  // Add a program to read back the result:
  progs[READ_RESULTS] = program::Copy(v3, stream4);

  return progs;
}

void executeGraphProgram(poplar::Device &device, poplar::Executable &exe,
                         const utils::Options &options) {
  poplar::Engine engine(std::move(exe));
  engine.load(device);

  std::vector<float> x = {1, 2, 3, 4};
  std::vector<float> y = {4, 3, 2, 1};
  std::vector<float> zInit = {-1, -1, -1, -1};
  std::vector<float> zResult1 = {0, 0, 0, 0};
  std::vector<float> zResult2 = {0, 0, 0, 0};
  engine.connectStream("write_x", x.data());
  engine.connectStream("write_y", y.data());
  engine.connectStream("write_z", zInit.data());

  // Run using custom vertex:
  engine.connectStream("read_z", zResult1.data());
  engine.run(WRITE_INPUTS);
  engine.run(CUSTOM_PROG);
  engine.run(READ_RESULTS);

  // Run program using PopLibs reduction:
  engine.connectStream("read_z", zResult2.data());
  engine.run(WRITE_INPUTS);
  engine.run(REDUCTION_PROG);
  engine.run(READ_RESULTS);

  // Check both methods give same result:
  for (auto i = 0u; i < zResult1.size(); ++i) {
    if (zResult1[i] != zResult2[i]) {
      throw std::runtime_error("Results do not match");
    }
  }
  std::cerr << "Results match.\n";

  if (options.profile) {
    // Retrieve and save profiling information from the engine:
    std::ofstream of(options.profileName);
    engine.printProfileSummary(of, {{"showExecutionSteps", "true"}});
  }
}

int main(int argc, char **argv) {
  try {
    auto options = utils::parseOptions(argc, argv);
    auto device = utils::getDeviceFromOptions(options);
    poplar::Graph graph(device.getTarget());

    // If we are loading the graph program we do not need
    // to construct it (which can be time consuming
    // for large programs):
    std::vector<poplar::program::Program> progs;
    if (!options.loadExe) {
      progs = buildGraphAndPrograms(graph, options);
    }

    auto exe = utils::compileOrLoadExe(graph, progs, options);

    if (options.saveExe && !options.loadExe) {
      auto outf = std::ofstream(getExeFileName(options));
      exe.serialize(outf);
    }

    executeGraphProgram(device, exe, options);

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
