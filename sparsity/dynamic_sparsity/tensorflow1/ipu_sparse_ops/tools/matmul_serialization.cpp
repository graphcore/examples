// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <random>
#include <vector>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <poputil/exceptions.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include "../utils.hpp"

using namespace poplar;


std::vector<float> randomVector(std::size_t size) {
    std::vector<float> vec(size);
    std::generate(vec.begin(), vec.end(), []() {
        return rand();
    });
    return vec;
}

Device setupDevice(bool runOnModel) {
    if (runOnModel) {
        // Create the IPU model device
        IPUModel ipuModel;
        return ipuModel.createDevice();
    }
    // Create the DeviceManager which is used to discover devices
    DeviceManager manager = DeviceManager::createDeviceManager();

    // Attempt to attach to a single IPU:
    Device device;
    bool success = false;
    // Loop over all single IPU devices on the host
    // Break the loop when an IPU is successfully acquired
    for (auto &hwDevice : manager.getDevices(TargetType::IPU, 1)) {
       device = std::move(hwDevice);
       std::cerr << "Trying to attach to IPU " << device.getId() << std::endl;
       if ((success = device.attach())) {
           std::cerr << "Attached to IPU " << device.getId() << std::endl;
           break;
        }
    }
    if (!success) {
        throw poputil::poplibs_error("Error attaching to device");
    }
  return device;
}


int main(int argc, char **argv) {
    if (argc < 6) {
        throw poputil::poplibs_error("Expected 5 arguments: size along dim 0, size along dim 1, "
                                     "hiden size, splits along dim 0, splits along dim 1");
    }

    std::size_t sizeX = std::atoi(argv[1]);
    std::size_t sizeY = std::atoi(argv[2]);
    std::size_t hiddenSize = std::atoi(argv[3]);

    std::size_t ASplits = std::atoi(argv[4]);
    std::size_t BSplits = std::atoi(argv[5]);

    std::cout << "Running matmul serialization test for input size A={" 
              << sizeX << ", " << hiddenSize << "} x B={" << hiddenSize 
              << ", " << sizeY << "}, and splits {" << ASplits << ", " 
              << BSplits << "}." << std::endl;

    if (sizeX < 1) {
        throw poputil::poplibs_error("Size along dim 0 should be at least 1");
    }
    if (sizeY < 1) {
        throw poputil::poplibs_error("Size along dim 1 should be at least 1");
    }
    if (sizeX % ASplits != 0) {
        throw poputil::poplibs_error("Size along dim 0 should be a multiple of the number of splits.");
    }
    if (sizeY % BSplits != 0) {
        throw poputil::poplibs_error("Size along dim 1 should be a multiple of the number of splits.");
    }

    //Get device or model
    auto device = setupDevice(true);
    // Create the Graph object
    Graph graph(device.getTarget());
    poplin::addCodelets(graph);
    popops::addCodelets(graph);

    // Add variables to the graph
    auto A = graph.addVariable(FLOAT, std::vector<std::size_t>{sizeX, hiddenSize},
                               VariableMappingMethod::LINEAR, "in_A");
    auto B = graph.addVariable(FLOAT, std::vector<std::size_t>{hiddenSize, sizeY},
                               VariableMappingMethod::LINEAR, "in_B");

    program::Sequence prog;

    // Add streams
    auto AStream = graph.addHostToDeviceFIFO("A-stream", FLOAT, sizeX * hiddenSize);
    auto BStream = graph.addHostToDeviceFIFO("B-stream", FLOAT, hiddenSize * sizeY);
    auto regularStream = graph.addDeviceToHostFIFO("regular-stream", FLOAT, sizeX * sizeY);
    auto serializedStream = graph.addDeviceToHostFIFO("serialized-stream", FLOAT, sizeX * sizeY);

    // Copy input from stream
    prog.add(program::Copy(AStream, A));
    prog.add(program::Copy(BStream, B));

    // Default Options
    poplar::OptionFlags options = getDenseGradMulDefaultOptions("");

    // Perform matmuls
    auto serializedOut = serializedMatmul(graph, prog, A, B, ASplits, BSplits,
                                          "serialized", true, options, nullptr);
    auto regularOut = serializedMatmul(graph, prog, A, B, ASplits, BSplits,
                                       "regular", false, options, nullptr);

    // Copy output to stream
    prog.add(program::Copy(serializedOut, serializedStream));
    prog.add(program::Copy(regularOut, regularStream));

    // Create the engine
    Engine engine(graph, prog);
    engine.load(device);

    auto a = randomVector(sizeX * hiddenSize);
    auto b = randomVector(sizeY * hiddenSize);
    std::vector<float> serial(sizeY * sizeX);
    std::vector<float> regular(sizeY * sizeX);

    engine.connectStream("A-stream", a.data());
    engine.connectStream("B-stream", b.data());
    engine.connectStream("regular-stream", serial.data());
    engine.connectStream("serialized-stream", regular.data());

    // Run
    std::cout << "Running" << std::endl;
    engine.run(0);
    std::cout << "Done." << std::endl;

    for (std::size_t i = 0; i < serial.size(); i++) {
        if (serial[i] != regular[i]) {
            throw poputil::poplibs_error("Result Mismatch. Expected " + std::to_string(regular[i])
                                         + " and got " + std::to_string(serial[i]));
        }
    }

    std::cout << "All results match." << std::endl;
    return 0;
}
