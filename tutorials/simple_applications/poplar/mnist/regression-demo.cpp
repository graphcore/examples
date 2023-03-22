// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include "mnist.h"

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popnn/Loss.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace poplar;
using namespace poplar::program;

std::mt19937 randomEngine;

// Create a vector of normally distributed random numbers (mean 0, stddev 1).
static std::vector<float> createRandomInitializers(unsigned numElements) {
  std::vector<float> inits(numElements);
  std::normal_distribution<> dist(0.0, 1.0);
  for (unsigned i = 0; i < numElements; ++i) {
    inits[i] = dist(randomEngine);
  }
  return inits;
}

// A simple program to train the model Wx + b = y on the MNIST dataset using
// gradient descent.
// If -IPU is passed as the first argument then use an IPU; otherwise use an
// IPUModel
// Any other arguments will be used to change the number of epochs to run and
// (optionally) the percentage of the images to use for training For example,
// regression-demo -IPU 2 10.0
int main(int argc, char **argv) {
  unsigned numberOfImages = 0, imageSize = 0;
  std::vector<float> data =
      readMNISTData(numberOfImages, imageSize, "data/train-images-idx3-ubyte");
  std::vector<unsigned> labels =
      readMNISTLabels("data/train-labels-idx1-ubyte");
  assert(numberOfImages == 60000);
  assert(imageSize == 784);

  bool useModel = true;

  // Default number of epochs which can be changed by passing an argument
  unsigned epochs = 50;

  // If -IPU is passed as the first argument then use an IPU; otherwise use an
  // IPUModel
  // Any other arguments will be used to change the number of epochs to run and
  // (optionally) the percentage of the images to use
  if (argc > 1 && std::strcmp(argv[1], "-IPU") == 0) {
    useModel = false;
    std::cout << "Using the IPU" << std::endl;
    if (argc > 2) {
      epochs = std::stoi(argv[2]);
      if (argc == 4)
        numberOfImages = floor(std::stof(argv[3]) / 100.0 * numberOfImages);
    }
  } else {
    std::cout << "Using the IPU Model" << std::endl;
    if (argc > 1) {
      epochs = std::stoi(argv[1]);
      if (argc == 3)
        numberOfImages = floor(std::stof(argv[2]) / 100.0 * numberOfImages);
    }
  }

  Device device;
  if (useModel) {
    IPUModel ipuModel;
    ipuModel.numIPUs = 1;
    ipuModel.tilesPerIPU = 4;
    device = ipuModel.createDevice();
  } else {
    auto manager = DeviceManager::createDeviceManager();

    // Attempt to attach to a single IPU:
    auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
    std::cout << "Trying to attach to IPU\n";
    auto it = std::find_if(devices.begin(), devices.end(),
                           [](Device &device) { return device.attach(); });

    if (it == devices.end()) {
      std::cerr << "Error attaching to device\n";
      return 1; // EXIT_FAILURE
    }

    device = std::move(*it);
    std::cout << "Attached to IPU " << device.getId() << std::endl;
  }

  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);

  // Create tensors in the graph.
  Tensor x = graph.addVariable(FLOAT, {imageSize, 1}, "x");
  poputil::mapTensorLinearly(graph, x);
  Tensor W = graph.addVariable(FLOAT, {10, imageSize}, "W");
  poputil::mapTensorLinearly(graph, W);
  Tensor b = graph.addVariable(FLOAT, {10, 1}, "b");
  poputil::mapTensorLinearly(graph, b);

  // Make the weights and biases host writable for initialization.
  graph.createHostWrite("weights", W);
  graph.createHostWrite("biases", b);

  Tensor numCorrect = graph.addVariable(UNSIGNED_INT, {1}, "numCorrect");
  poputil::mapTensorLinearly(graph, numCorrect);
  Tensor expected = graph.addVariable(UNSIGNED_INT, {1}, "expected");
  poputil::mapTensorLinearly(graph, expected);

  // Create the graph and program to execute the model, calculate the gradients
  // of W, b and subtract the scaled gradients from the parameters
  Sequence mProg;

  // Calculate y = Wx + b
  Tensor t = poplin::matMul(graph, W, x, mProg, "Wx");
  Tensor y = popops::add(graph, t, b, mProg, "Wx+b");

  // Calculate the loss
  Tensor loss = graph.addVariable(FLOAT, {1}, "loss");
  poputil::mapTensorLinearly(graph, loss);
  // The loss gradient with respect to y
  Tensor delta = graph.addVariable(FLOAT, {1, 10}, "delta");
  poputil::mapTensorLinearly(graph, delta);
  auto softmaxY = popnn::nonLinearity(graph, popnn::NonLinearityType::SOFTMAX,
                                      y.transpose(), mProg, "softmax(Wx+b)");
  mProg.add(popnn::calcLoss(graph, softmaxY, expected, loss, delta, numCorrect,
                            popnn::CROSS_ENTROPY_LOSS, "dE/d(Wx+b)"));
  // Update: b -= eta * dE/db, where dE/db = dE/dy
  float eta = 0.0009;
  popops::scaledAddTo(graph, b, delta.transpose(), -eta, mProg,
                      "b += -eta * dE/db");

  // Update: W -= eta * dE/dW
  Tensor wGrad =
      poplin::matMul(graph, delta.transpose(), x.transpose(), mProg, "dE/dW");
  popops::scaledAddTo(graph, W, wGrad, -eta, mProg, "W += -eta * dE/dW");

  // Create a control program to execute the SGD training algorithm.
  DataStream dataIn = graph.addHostToDeviceFIFO("data", FLOAT, imageSize);
  DataStream labelIn = graph.addHostToDeviceFIFO("labels", UNSIGNED_INT, 1);
  DataStream numCorrectOut =
      graph.addDeviceToHostFIFO("hostNumCorrect", UNSIGNED_INT, 1);

  Sequence trainProg;
  unsigned int hNumCorrect = 0;

  // Initialize the numCorrect tensor to 0
  Tensor zero = graph.addConstant(UNSIGNED_INT, {1}, 0);
  graph.setTileMapping(zero, 0);
  trainProg.add(Copy(zero, numCorrect));
  const unsigned batchSize = 300;
  trainProg.add(Repeat(
      batchSize, Sequence({Copy(dataIn, x), Copy(labelIn, expected), mProg})));
  trainProg.add(Copy(numCorrect, numCorrectOut));

  // Create a Poplar engine.
  // By default, in order to reduce host memory consumption, the generation and
  // retention of debug information is not enabled. Because we wish to print a
  // Profile Summary below, we need to tell the Engine at `prog` compile time to
  // retain this information via the `retainDebugInformation` option.
  auto engine =
      Engine{graph, trainProg, {{"debug.retainDebugInformation", "true"}}};
  engine.load(device);

  // Connect up the data streams
  engine.connectStream("data", &data[0], &data[numberOfImages * imageSize]);
  engine.connectStream("labels", &labels[0], &labels[numberOfImages]);
  engine.connectStream("hostNumCorrect", &hNumCorrect);

  // Initialize the weights and biases
  std::vector<float> initW = createRandomInitializers(W.numElements());
  std::vector<float> initB = createRandomInitializers(b.numElements());
  engine.writeTensor("weights", initW.data(), initW.data() + initW.size());
  engine.writeTensor("biases", initB.data(), initB.data() + initB.size());

  // Run the training algorithm, printing out the accuracy regularly
  unsigned totalCorrect = 0, totalTested = 0;

  const unsigned batches = numberOfImages / batchSize;
  for (unsigned epoch = 1; epoch <= epochs; ++epoch) {
    for (unsigned batch = 1; batch <= batches; ++batch) {
      engine.run(0); // trainProg
      totalCorrect += hNumCorrect;
      totalTested += batchSize;
      if (epoch == 1 && batch == 1) {
        engine.printProfileSummary(std::cout);
      }
      // Status update if we've done at least another 20th of an epoch
      if (totalTested > numberOfImages / 20) {
        unsigned percentCorrect = totalCorrect * 100 / totalTested;
        unsigned epochPercent = batch * 100 / batches;
        std::cout << "Epoch " << epoch << " (" << epochPercent
                  << "%), accuracy " << percentCorrect << "%\n";
        totalCorrect = totalTested = 0;
      }
    }
  }

  return 0;
}
