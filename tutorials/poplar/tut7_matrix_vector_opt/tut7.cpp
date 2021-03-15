// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <memory>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>

// This example performs matrix multiplication on the IPU by decomposing
// the column axis (i.e. the number of columns) into N partial sums.
// The partial sums are then added up to get the final result. For example,
// consider the following multiplication:
//
// ( 0 1 2 3 )   ( A )
// ( 4 5 6 7 ) x ( B )
//               ( C )
//               ( D )
//
// With a column axis split of two there would be four
// partial sums:
//
//  P1 = 0xA + 1xB, P2 = Cx2 + Dx3, P3 = 4xA + 5xB, P4 = Cx6 + Dx7
//
// The final vector would be created by adding these partial sums:
//
//  ( P1 + P2, P3 + P4)
//
// On the IPU, the code needs to work out the optimial column axis split for
// a particular matrix. This is done by estimating the compute cost for
// every possible split and choosing the best.

#define VERBOSE_LEVEL 1

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
    auto relativeErr = std::fabs(1 - sum / (output[row]));
    if (relativeErr > 1e-5) {
      std::cout << "ERROR: output " << row << ": expected=" << sum
                << ", actual=" << output[row]
                << ", relative error=" << relativeErr << "\n";
      return 1;
    }
  }
  std::cout << "Multiplication result OK\n";
  return 0;
}

// Utility function to divide two integers rounding up the result.
inline unsigned ceilDiv(unsigned a, unsigned b) { return (a + b - 1) / b; }

// Function to estimate the number of cycles required to perform the
// multiplication given a particular column axis split.
unsigned estimateCycles(const Graph &graph, unsigned numRows, unsigned numCols,
                        unsigned split, bool verbose) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  // There are numRows * split partial sums, each will have a vertex to
  // calculate it and these vertices will be spread evenly over the tiles.
  // The compute cost is equal to the time of the longest running tile (i.e.
  // the tile with the maximum number of vertices).
  unsigned maxVerticesPerTile =
      std::max(ceilDiv(numRows * split, numTiles), 1U);
  // Each vertex has to calculate a dot product which it can do using
  // a vector unit of width 2 (i.e. 2 macs per cycle).
  auto cyclesPerVertex = 5 + (ceilDiv(numCols, split) + 1) / 2;
  // The compute cost is the time of the longest running worker context on
  // the longest running tile. To calculate this we can round up the
  // number of vertices to the nearest multiple of the number of worker
  // contexts.
  auto computeCost =
      ceilDiv(maxVerticesPerTile, numWorkers) * numWorkers * cyclesPerVertex;
  // The number of exchange cycles before the partial sum calculation is
  // bound by the maximum number of bytes a tile can receive.
  // Each tile will receive multiple sections of the input vector
  // (of size 'numCols / split'). Each tile will be allocated all the rows
  // of a section before moving on to the next so the number of sections can
  // be estimated as 'numVertices / numRows'.
  auto exchangeCost = ceilDiv(maxVerticesPerTile, numRows) *
                      ceilDiv(numCols, split) * sizeof(float) /
                      exchangeBytesPerCycle;
  // After the partial sums are calculated, a second compute set will
  // calculate the reduction with a vertex for each element of the output (i.e
  // 'numRows' elements). The compute cost for the reduction can be calculated
  // with similar logic to above.
  auto maxReduceVerticesPerTile = std::max(ceilDiv(numRows, numTiles), 1U);
  auto cyclesPerReduceVertex = 5 + (split + 1) / 2;
  auto reduceComputeCost = ceilDiv(maxReduceVerticesPerTile, numWorkers) *
                           numWorkers * cyclesPerReduceVertex;
  // Each tile reducing will have to receive 'split' number of elements per
  // set of partial sums it is reducing.
  auto reduceExchangeCost =
      maxReduceVerticesPerTile * split * sizeof(float) / exchangeBytesPerCycle;
  // The estimated total number of cycles it the sum of the compute and
  // exchange cycles for the dot product and reduce phases.
  auto totalCost =
      computeCost + exchangeCost + reduceComputeCost + reduceExchangeCost;
  if (verbose) {
    std::cout << "colsAxisSplit=" << split << ", total cost=" << totalCost
              << " (compute cost=" << computeCost
              << ", exchange cost=" << exchangeCost
              << ", reduce exchange cost=" << reduceExchangeCost
              << ", reduce compute cost=" << reduceComputeCost << ")\n";
  }
  return totalCost;
}

// Given a specific number of rows and columns, calculate the optimal
// column axis split to use.
unsigned calcOptimalColAxisSplit(const Graph &graph, unsigned numRows,
                                 unsigned numCols) {
  unsigned best = 1;
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  unsigned worstCost = std::numeric_limits<unsigned>::min();
  for (unsigned split = 1; split < numCols; ++split) {
    auto cost =
        estimateCycles(graph, numRows, numCols, split, VERBOSE_LEVEL >= 2);
    if (bestCost > cost) {
      best = split;
      bestCost = cost;
    }
    worstCost = std::max(cost, worstCost);
  }
  if (VERBOSE_LEVEL >= 1) {
    std::cout << "Best split chosen:\n";
    estimateCycles(graph, numRows, numCols, best, true);
    std::cout << "Worst cost seen: " << worstCost << "\n";
  }
  return best;
}

// This function returns a device side program that will multiply
// the data in the 2-d tensor 'matrix' with the 1-d vector held
// in the 'in' tensor. When the program executes
// the result is placed in the 'out' 1-d tensor.
Program buildMultiplyProgram(Graph &graph, Tensor matrix, Tensor in,
                             Tensor out) {
  auto numRows = matrix.dim(0);
  auto numCols = matrix.dim(1);
  // Get the optimal column axis split to split the number of columns
  // into partial sums.
  unsigned colAxisSplit = calcOptimalColAxisSplit(graph, numRows, numCols);

  // Create a tensor to hold the intermediate calculated partial sums.
  auto partials = graph.addVariable(FLOAT, {numRows, colAxisSplit}, "partials");

  const auto numTiles = graph.getTarget().getNumTiles();

  // The input vector is used by all tiles. So their is no obvious place
  // to put it that reduces communications. To balance memory, just
  // spread it over all the tiles.
  for (unsigned i = 0; i < numCols; ++i) {
    graph.setTileMapping(in[i], i * numTiles / numCols);
  }

  // Create a compute set to hold the vertices to perform the
  // partial sum calculations.
  ComputeSet mulCS = graph.addComputeSet("mulCS");

  // Create a vertex for each segment, for each row.
  for (unsigned i = 0; i < colAxisSplit; ++i) {
    // The split may not divide the number of columns exactly. So the
    // columns in this segment need to be quantized.
    unsigned beginCol = (i * numCols) / colAxisSplit;
    unsigned endCol = ((i + 1) * numCols) / colAxisSplit;
    if (beginCol == endCol)
      continue;
    for (unsigned row = 0; row < numRows; ++row) {
      // The matrix elements for the dot product are the slice of the
      // row between 'beginCol' and 'endCol'.
      auto matrixElements = matrix[row].slice(beginCol, endCol);
      // The input elements for the dot product are just the slice of the
      // input vector between 'beginCol' and 'endCol'.
      auto inputElements = in.slice(beginCol, endCol);
      // Create a 'DotProductVertex' vertex in the 'mulCS' compute set and
      // connect the inputs 'a' and 'b' to the matrix and input elements to
      // be multiplied. Connect the output 'out' to an element of the
      // tensor 'partials'
      auto v = graph.addVertex(mulCS, "DotProductVertex",
                               {{"a", matrixElements},
                                {"b", inputElements},
                                {"out", partials[row][i]}});

      // This vertices are evenly spread over the tiles.
      unsigned tile = (i * numRows + row) * numTiles / (colAxisSplit * numRows);
      // Map the vertex and its associated data to the current tile.
      graph.setTileMapping(v, tile);
      graph.setTileMapping(matrix[row].slice(beginCol, endCol), tile);
      graph.setTileMapping(partials[row][i], tile);
      // Guess 5 cycles for the overhead of starting the vertex and setting
      // up the loop.
      // The dot product can be performed by vectorized mac instructions that
      // can perform 2 macs per cycle.
      graph.setPerfEstimate(v, 5 + ((endCol - beginCol) + 1) / 2);
    }
  }
  // Create a compute set to calculate the reduction.
  auto reduceCS = graph.addComputeSet("reduceCS");

  // For each output element create a vertex.
  for (unsigned row = 0; row < numRows; ++row) {
    // Create a 'ReduceVertex' that takes all the partial sums for a row
    // and reduces them to a single value placed in the output vector.
    auto v = graph.addVertex(reduceCS, "ReduceVertex",
                             {{"in", partials[row]}, {"out", out[row]}});
    // Map the computation to a tile such that the vertices are spread
    // evenly over the tiles. Map associated tensor data to the same tile.
    auto tile = (row * numTiles) / numRows;
    graph.setTileMapping(v, tile);
    graph.setTileMapping(out[row], tile);
    // Addition can be vectorized at 2 float additions per cycle.
    graph.setPerfEstimate(v, 5 + (partials[row].numElements() + 1) / 2);
  }

  // The program to perform the multiplication consists of executing the
  // compute set that calculates the partial sums followed by the compute
  // set that performs the reduction.
  return Sequence(Execute(mulCS), Execute(reduceCS));
}

int main(int argc, char **argv) {
  if ((argc < 3 || argc > 4) || (argc == 4 && strcmp(argv[3], "mk1") != 0)) {
    std::cerr << "usage: " << argv[0] << " numRows numCols mk1(optional)\n";
    return 1;
  }
  unsigned numRows = std::atoi(argv[1]);
  unsigned numCols = std::atoi(argv[2]);
  std::string model;

  if (argc == 4 && strcmp(argv[3], "mk1") == 0) {
    model = "ipu1";
  }
  else {
    model = "ipu2";
  }

  std::cout << "Multiplying matrix of size " << numRows << "x" << numCols
            << " by vector of size " << numCols << "\n";

  std::cout << "Constructing compute graph and control program\n";
  // This graph is going to target an simulated IPU. For simplicity in
  // this example the configuration of the exchange in this simulated model
  // is set to be more simplistic to reduce some latencies/delays in
  // the exchange fabric.

  char modelChar[model.length() + 1];
  strcpy(modelChar, model.c_str());

  IPUModel ipuModel(modelChar);
  ipuModel.minIPUSyncDelay = 0;
  ipuModel.relativeSyncDelay = IPUModel::RelativeSyncDelayType::NO_DELAY;
  auto device = ipuModel.createDevice();
  Graph graph(device);
  graph.addCodelets("matrix-mul-codelets.cpp");

  // Create tensors in the graph to hold the input/output data.
  Tensor matrix = graph.addVariable(FLOAT, {numRows, numCols}, "matrix");
  Tensor inputVector = graph.addVariable(FLOAT, {numCols}, "inputVector");
  Tensor outputVector = graph.addVariable(FLOAT, {numRows}, "outputVector");

  // Create host buffers for the inputs and outputs and fill the inputs
  // with sample data.
  auto hMatrix = std::unique_ptr<float[]>(new float[numRows * numCols]);
  auto hInput = std::unique_ptr<float[]>(new float[numCols]);
  auto hOutput = std::unique_ptr<float[]>(new float[numRows]);

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

  // Create an engine from the compute graph and control program.
  Engine engine(graph,
                Sequence(Copy(inStreamV, inputVector), Copy(inStreamM, matrix),
                         mulProg, Copy(outputVector, outStream)));
  engine.load(device);
  engine.connectStream("inputVector", hInput.get());
  engine.connectStream("inputMatrix", hMatrix.get());
  engine.connectStream("out", hOutput.get());

  std::cout << "Running graph program to multiply matrix by vector\n";
  engine.run();

  // Check the results match what is expected.
  int err = checkResult(&hMatrix[0], &hInput[0], &hOutput[0], numRows, numCols);
  if (err)
    return err;

  // Write graph and execution profiles to files
  ProfileValue graphProfile = engine.getGraphProfile();
  ProfileValue executionProfile = engine.getExecutionProfile();
  std::ofstream graphFile("graph.json");
  if (graphFile.is_open()) {
    poplar::serializeToJSON(graphFile, graphProfile, true);
    graphFile.close();
  }
  std::ofstream executionFile("execution.json");
  if (executionFile.is_open()) {
    poplar::serializeToJSON(executionFile, executionProfile, true);
    executionFile.close();
  }

  // Product a report showing the profile of execution on the simulated IPU.
  engine.printProfileSummary(std::cout,
                             OptionFlags{{"showExecutionSteps", "true"}});
  return 0;
}
