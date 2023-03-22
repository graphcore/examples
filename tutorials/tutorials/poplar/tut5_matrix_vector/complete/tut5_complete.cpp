// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>

#include <iostream>
#include <vector>

using namespace poplar;
using namespace poplar::program;

// Function to check the result of multiplying the matrix by the vector.
int checkResult(float *matrix, float *input, float *output, unsigned numRows,
                unsigned numCols) {
  for (unsigned row = 0; row < numRows; ++row) {
    float sum = 0;
    for (unsigned col = 0; col < numCols; ++col) {
      sum += matrix[row * numCols + col] * input[col];
    }
    if (output[row] != sum) {
      std::cout << "ERROR: output " << row << ": expected=" << sum
                << ", actual=" << output[row] << "\n";
      return 1;
    }
  }
  std::cout << "Multiplication result OK\n";
  return 0;
}

// This function returns a device side program that will multiply
// the data in the 2-d tensor 'matrix' with the 1-d vector held
// in the 'in' tensor. When the program executes
// the result is placed in the 'out' 1-d tensor.
Program buildMultiplyProgram(Graph &graph, Tensor matrix, Tensor in,
                             Tensor out) {
  // Create a compute set to hold the vertices to perform the calculation
  ComputeSet mulCS = graph.addComputeSet("mulCS");

  // The compute set holds a vertex for every output value. Each vertex
  // takes a row of the matrix as input and the whole input vector and
  // performs a dot-product placing the result in an element of the
  // output vector.
  auto numRows = matrix.dim(0);
  for (unsigned i = 0; i < numRows; ++i) {
    auto v = graph.addVertex(mulCS,              // Put the vertex in the
                                                 // 'mulCS' compute set.
                             "DotProductVertex", // Create a vertex of this
                                                 // type.
                             {{"a", matrix[i]},  // Connect input 'a' of the
                                                 // vertex to a row of the
                                                 // matrix.
                              {"b", in},         // Connect input 'b' of the
                                                 // vertex to whole
                                                 // input vector.
                              {"out", out[i]}}); // Connect the output 'out'
                                                 // of the vertex to a single
                                                 // element of the output
                                                 // vector.
    graph.setTileMapping(v, i);
    graph.setPerfEstimate(v, 20);
  }
  // The returned program just executes the 'mulCS' compute set that is,
  // executes every vertex calculation in parallel.
  return Execute(mulCS);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " numRows numCols\n";
    return 1;
  }

  unsigned numRows = std::atoi(argv[1]);
  unsigned numCols = std::atoi(argv[2]);
  std::cout << "Multiplying matrix of size " << numRows << "x" << numCols
            << " by vector of size " << numCols << "\n";

  // Create the IPU model device
  IPUModel ipuModel;
  Device device = ipuModel.createDevice();
  Target target = device.getTarget();

  std::cout
      << "Creating new graph object and compiling vertex program additions\n";

  Graph graph(target);
  graph.addCodelets("matrix-mul-codelets.cpp");

  std::cout << "Constructing full compute graph and control program\n";

  // Create tensors in the graph to hold the input/output data.
  Tensor matrix = graph.addVariable(FLOAT, {numRows, numCols}, "matrix");
  Tensor inputVector = graph.addVariable(FLOAT, {numCols}, "inputVector");
  Tensor outputVector = graph.addVariable(FLOAT, {numRows}, "outputVector");
  poputil::mapTensorLinearly(graph, matrix);
  poputil::mapTensorLinearly(graph, inputVector);
  poputil::mapTensorLinearly(graph, outputVector);

  // Create host buffers for the inputs and outputs and fill the inputs
  // with sample data.
  auto hMatrix = std::vector<float>(numRows * numCols);
  auto hInput = std::vector<float>(numCols);
  auto hOutput = std::vector<float>(numRows);

  for (unsigned col = 0; col < numCols; ++col) {
    hInput[col] = col;
    for (unsigned row = 0; row < numRows; ++row) {
      hMatrix[row * numCols + col] = row * col;
    }
  }

  // Create a device program to multiply two tensors together.
  auto mulProg = buildMultiplyProgram(graph, matrix, inputVector, outputVector);

  // Set up data streams to copy data in and out of graph
  auto inStreamV = graph.addHostToDeviceFIFO("inputVector", FLOAT, numCols);
  auto inStreamM =
      graph.addHostToDeviceFIFO("inputMatrix", FLOAT, numCols * numRows);
  auto outStream = graph.addDeviceToHostFIFO("out", FLOAT, numRows);

  // Create a program that copies data from the host buffers, multiplies
  // the result and copies the result back to the host.
  auto prog = Sequence({Copy(inStreamV, inputVector), Copy(inStreamM, matrix),
                        mulProg, Copy(outputVector, outStream)});

  // Create an engine from the compute graph and control program.
  Engine engine(graph, prog);
  engine.load(device);
  engine.connectStream("inputVector", hInput.data());
  engine.connectStream("inputMatrix", hMatrix.data());
  engine.connectStream("out", hOutput.data());

  // Execute the program
  std::cout << "Running graph program to multiply matrix by vector\n";
  engine.run();

  // Check the results match what is expected.
  return checkResult(&hMatrix[0], &hInput[0], &hOutput[0], numRows, numCols);
}
