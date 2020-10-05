// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>
#include <tuple>
#include <cstdint>
#include <algorithm>
#include <random>

#include <poplar/DeviceManager.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <popsparse/SparsePartitioner.hpp>
#include <popsparse/FullyConnected.hpp>
#include <popsparse/SparseStorageFormats.hpp>
#include "utils.hpp"

namespace py = pybind11;

using PopSparseSplits = std::tuple<std::size_t, std::size_t, std::size_t>;
using PopSparseTensorSizes = std::tuple<std::size_t, std::size_t, PopSparseSplits>;

// Return a planning cache that is global in this shared
// object (but have one per thread):
popsparse::dynamic::PlanningCache* getCache() {
  thread_local popsparse::dynamic::PlanningCache global_cache;
  return &global_cache;
}

const auto get_sparse_tensor_sizes_doc =
"Return the tensor sizes and serial splits required for "
"a specific sparse matrix size and sparsity.";
PopSparseTensorSizes get_sparse_tensor_sizes(std::size_t num_ipus,
                                            std::size_t maxNonZeros,
                                            std::size_t numGroups, std::size_t batchSize,
                                            std::size_t inputSize, std::size_t outputSize,
                                            const std::string& dtype,
                                            const std::string& matmulOptions) {
  using namespace popsparse::dynamic;

  // Construct a small graph to get the information from Poplibs:
  auto dm = poplar::DeviceManager::createDeviceManager();

  auto hwDevices = dm.getDevices(poplar::TargetType::IPU, num_ipus);
  if (hwDevices.empty()) {
    throw std::runtime_error("No device found");
  }
  poplar::Device *device = &hwDevices[0];
  poplar::Graph graph(device->getTarget());

  const auto dataType = type_from_string(dtype);
  auto options = getSparseMulDefaultOptions(matmulOptions);
  auto sparsityFactor = maxNonZeros / float(inputSize * outputSize);

  SparsityParams sparsityParams(SparsityType::Element, SparsityStructure::Unstructured);
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);

  const SparseTensor weights =
      createFullyConnectedWeights(graph, dataType, params, "weights", options, getCache());

  auto splits = fullyConnectedDenseGradWSerialSplits(
    graph, weights.getNzValuesTensor().elementType(), params, options, getCache());

  return std::make_tuple(
    weights.getMetaInfoTensor().numElements(),
    weights.getNzValuesTensor().numElements(),
      std::make_tuple(
        std::get<0>(splits),
        std::get<1>(splits),
        std::get<2>(splits))
    );
}

template <typename T>
using PyArrayCLayout = py::array_t<T, py::array::c_style>;
using PopSparseMatrix = std::tuple<PyArrayCLayout<std::uint16_t>, PyArrayCLayout<float>>;
using Triplets = std::tuple<PyArrayCLayout<std::size_t>, PyArrayCLayout<std::size_t>, PyArrayCLayout<float>>;

template <typename T>
std::vector<T> numpyToVector(const py::array_t<T>& array) {
  if (array.ndim() != 1) {
    throw std::runtime_error("Can only convert 1D arrays.");
  }

  std::vector<T> result(array.size());
  for (auto i = 0u; i < array.size(); ++i) {
    result[i] = *array.data(i);
  }
  return result;
}

// A class to store the partitioner alongside
// the IPUmodel that it relies on:
class PartitionerContext {
public:
  PartitionerContext(
    std::size_t num_ipus,
    std::size_t maxNonZeros,
    std::size_t numGroups, std::size_t batchSize,
    std::size_t inputSize, std::size_t outputSize,
    poplar::Type dataType,
    const std::string& jsonConfig) {
      auto dm = poplar::DeviceManager::createDeviceManager();

      auto hwDevices = dm.getDevices(poplar::TargetType::IPU, num_ipus);
      if (hwDevices.empty()) {
        throw std::runtime_error("No device found");
      }
      poplar::Device *device = &hwDevices[0];
      auto target = device->getTarget();
      auto options = getSparseMulDefaultOptions(jsonConfig);
      auto sparsityFactor = maxNonZeros / float(inputSize * outputSize);

      popsparse::dynamic::SparsityParams sparsityParams(
        popsparse::dynamic::SparsityType::Element,
        popsparse::dynamic::SparsityStructure::Unstructured);
      auto params = popsparse::dynamic::FullyConnectedParams::createWithNzRatio(
        sparsityParams, sparsityFactor, batchSize, numGroups, inputSize, outputSize);
      partitioner.reset(new popsparse::dynamic::Partitioner<float>(
        params, dataType, target, options, getCache()));
  }

  popsparse::dynamic::Partitioner<float>& get() { return *partitioner; }

private:
  poplar::IPUModel ipuModel;
  std::unique_ptr<popsparse::dynamic::Partitioner<float>> partitioner;
};

const auto representation_from_triplets_doc =
"Create the host side sparse representation of a matrix from "
"triplets of row indices, column indices, and values. The Triplets "
"must be sorted by the row index.";
PopSparseMatrix representation_from_triplets(
    std::size_t num_ipus,
    std::size_t maxNonZeros,
    std::size_t numGroups, std::size_t batchSize,
    std::size_t inputSize, std::size_t outputSize,
    const std::string& dtype,
    py::array_t<std::size_t> npRowIndices, py::array_t<std::size_t> npColIndices,
    py::array_t<float> npValues,
    const std::string& matmulOptions) {

  using namespace popsparse::dynamic;

  if (npRowIndices.shape(0) != npColIndices.shape(0) ||
      npColIndices.shape(0) != npValues.shape(0)) {
    throw std::runtime_error("Triplets arrays have inconsistent sizes");
  }

  /// Construct a COOMatrix (triplets):
  // Note that we have to swap rows and columns here as
  // popsparse has some odd conventions:
  auto rowIndices = numpyToVector(npColIndices);
  auto colIndices = numpyToVector(npRowIndices);
  auto values = numpyToVector(npValues);
  auto cooMatrix = popsparse::COOMatrix<float>(values, colIndices, rowIndices);

  const auto dataType = type_from_string(dtype);
  auto partitioner = PartitionerContext(
    num_ipus, maxNonZeros, numGroups, batchSize,
    inputSize, outputSize, dataType, matmulOptions);
  auto pnBuckets = partitioner.get().createSparsityDataImpl(cooMatrix);

  // Copy the flat representation to numpy arrays amd return them:
  auto metainfo = PyArrayCLayout<std::uint16_t>(pnBuckets.metaInfo.size());
  auto nzvalues = PyArrayCLayout<float>(pnBuckets.nzValues.size());

  for( int i = 0; i < pnBuckets.metaInfo.size(); ++i) {
    *metainfo.mutable_data(i) = pnBuckets.metaInfo[i];
  }

  for( int i = 0; i < pnBuckets.nzValues.size(); ++i) {
    *nzvalues.mutable_data(i) = pnBuckets.nzValues[i];
  }

  return std::make_tuple(metainfo, nzvalues);
}

const auto triplets_from_representation_doc =
"Convert from host side sparse representation back to triplets.";
Triplets triplets_from_representation(
    std::size_t num_ipus,
    std::size_t maxNonZeros,
    std::size_t numGroups, std::size_t batchSize,
    std::size_t inputSize, std::size_t outputSize,
    const std::string& dtype,
    const py::array_t<std::size_t>& metainfo,
    const py::array_t<float>& nzvalues,
    const std::string& matmulOptions) {

  const auto dataType = type_from_string(dtype);
  auto partitioner = PartitionerContext(
    num_ipus, maxNonZeros, numGroups, batchSize,
    inputSize, outputSize, dataType, matmulOptions);
  popsparse::dynamic::SparsityDataImpl<float> buckets;
  buckets.metaInfo = numpyToVector(metainfo);
  buckets.nzValues = numpyToVector(nzvalues);

  auto cooMatrix = partitioner.get().sparsityDataImplToCOOMatrix(buckets);

  auto rowIndices = PyArrayCLayout<std::size_t>(cooMatrix.rowIndices.size());
  auto colIndices = PyArrayCLayout<std::size_t>(cooMatrix.columnIndices.size());
  auto nzValues = PyArrayCLayout<float>(cooMatrix.nzValues.size());

  for( int i = 0; i < rowIndices.size(); ++i) {
    *rowIndices.mutable_data(i) = cooMatrix.rowIndices[i];
  }

  for( int i = 0; i < colIndices.size(); ++i) {
    *colIndices.mutable_data(i) = cooMatrix.columnIndices[i];
  }

  for( int i = 0; i < nzValues.size(); ++i) {
    *nzValues.mutable_data(i) = cooMatrix.nzValues[i];
  }

  return std::make_tuple(colIndices, rowIndices, nzValues);
}

PYBIND11_MODULE(host_utils, m) {
  m.doc() = "Host C++ code for ipu_sparse_ops module";
  m.def("get_sparse_tensor_sizes", &get_sparse_tensor_sizes, get_sparse_tensor_sizes_doc);
  m.def("representation_from_triplets", &representation_from_triplets, py::return_value_policy::move, representation_from_triplets_doc);
  m.def("triplets_from_representation", &triplets_from_representation, py::return_value_policy::move, triplets_from_representation_doc);
}
