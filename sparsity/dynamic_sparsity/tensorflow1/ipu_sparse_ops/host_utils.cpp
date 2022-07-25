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
#include <poplar/Target.hpp>
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

poplar::DeviceManager* getDeviceManager() {
  thread_local poplar::DeviceManager device_manager = poplar::DeviceManager::createDeviceManager();
  return &device_manager;
}

struct MatmulSpec {
  std::size_t maxNonZeros;
  std::size_t blockSize;
  std::size_t numGroups;
  std::size_t batchSize;
  std::size_t inputSize;
  std::size_t outputSize;
};

static float computeDensity(const MatmulSpec& spec) {
  float numBlocks = (spec.inputSize / spec.blockSize) * (spec.outputSize / spec.blockSize);
  return spec.maxNonZeros / numBlocks;
}

static poplar::Target getTarget(
    std::size_t num_ipus,
    std::size_t ipu_version) {
  switch(ipu_version) {
    case 0: {
      // Construct a small graph to get the information from Poplibs:

      auto hwDevices = getDeviceManager()->getDevices(poplar::TargetType::IPU, num_ipus);
      if (hwDevices.empty()) {
        throw std::runtime_error("No device found");
      }
      poplar::Device *device = &hwDevices[0];
      return device->getTarget();
    }

    case 1:  // intentional fall-through
    case 2: {
      return poplar::Target::createIPUTarget(num_ipus, "ipu" + std::to_string(ipu_version));
    }

    default: {
      throw std::runtime_error("Unknown ipu_version " + std::to_string(ipu_version));
    }
  }
}

static MatmulSpec matmulSpecFromNamedTuple(const py::object& obj) {
  MatmulSpec spec;
  spec.maxNonZeros = obj.attr("max_non_zero_blocks").cast<std::size_t>();
  spec.blockSize = obj.attr("block_size").cast<std::size_t>();
  spec.numGroups = obj.attr("num_groups").cast<std::size_t>();
  spec.batchSize = obj.attr("batch_size").cast<std::size_t>();
  spec.inputSize = obj.attr("input_size").cast<std::size_t>();
  spec.outputSize = obj.attr("output_size").cast<std::size_t>();
  return spec;
}

const auto initialize_device_manager_doc =
"Initialize the Poplar device manager.";
void initialize_device_manager() {
  (void*)getDeviceManager();
}

const auto get_sparse_tensor_sizes_doc =
"Return the tensor sizes and serial splits required for "
"a specific sparse matrix size and sparsity.";
PopSparseTensorSizes get_sparse_tensor_sizes(std::size_t num_ipus,
                                             std::size_t ipu_version,
                                             const py::object& matmulSpec,
                                             const std::string& dtype,
                                             const std::string& matmulOptions) {
  using namespace popsparse::dynamic;

  poplar::Graph graph(getTarget(num_ipus, ipu_version));

  const auto spec = matmulSpecFromNamedTuple(matmulSpec);

  const auto dataType = type_from_string(dtype);
  auto options = getSparseMulDefaultOptions(matmulOptions);
  auto sparsityFactor = computeDensity(spec);

  SparsityParams sparsityParams(
    SparsityType::Element, SparsityStructure::Unstructured, {spec.blockSize, spec.blockSize});
  auto params = FullyConnectedParams::createWithNzRatio(
    sparsityParams, sparsityFactor,
    spec.batchSize, spec.numGroups, spec.inputSize, spec.outputSize);

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
    const std::size_t num_ipus,
    const std::size_t ipu_version,
    const MatmulSpec spec,
    const poplar::Type dataType,
    const std::string& jsonConfig,
    const std::string& name = "") {
      auto target = getTarget(num_ipus, ipu_version);
      auto options = getSparseMulDefaultOptions(jsonConfig);
      auto sparsityFactor = computeDensity(spec);

      popsparse::dynamic::SparsityParams sparsityParams(
        popsparse::dynamic::SparsityType::Element,
        popsparse::dynamic::SparsityStructure::Unstructured,
        {spec.blockSize, spec.blockSize});
      auto params = popsparse::dynamic::FullyConnectedParams::createWithNzRatio(
        sparsityParams, sparsityFactor,
        spec.batchSize, spec.numGroups, spec.inputSize, spec.outputSize);
      partitioner.reset(new popsparse::dynamic::Partitioner<float>(
        params, dataType, target, options, getCache(), name));
  }

  popsparse::dynamic::Partitioner<float>& get() { return *partitioner; }

private:
  std::unique_ptr<popsparse::dynamic::Partitioner<float>> partitioner;
};

const auto representation_from_triplets_doc =
"Create the host side sparse representation of a matrix from "
"triplets of row indices, column indices, and (block) values.";
PopSparseMatrix representation_from_triplets(
    std::size_t num_ipus,
    std::size_t ipu_version,
    const py::object& matmulSpec,
    const std::string& dtype,
    py::array_t<std::size_t> npRowIndices, py::array_t<std::size_t> npColIndices,
    py::array_t<float> npValues,
    const std::string& matmulOptions,
    const std::string& debugName = "") {

  const auto spec = matmulSpecFromNamedTuple(matmulSpec);

  using namespace popsparse::dynamic;

  const std::size_t blockElements = spec.blockSize * spec.blockSize;
  if (npRowIndices.shape(0) != npColIndices.shape(0) ||
      npColIndices.shape(0) * blockElements != npValues.shape(0)) {
    throw std::runtime_error("Triplets arrays have inconsistent sizes");
  }

  /// Construct a COOMatrix (triplets):
  // Note that we have to swap rows and columns here as
  // popsparse has some odd conventions:
  auto rowIndices = numpyToVector(npColIndices);
  auto colIndices = numpyToVector(npRowIndices);
  auto values = numpyToVector(npValues);

  // Popsparse uses element indices where as this module's API
  // uses block indices so we convert here:
  if (spec.blockSize != 1) {
    for (auto &r: rowIndices) {
      r *= spec.blockSize;
    }
    for (auto &c: colIndices) {
      c *= spec.blockSize;
    }
  }

  auto cooMatrix = popsparse::COOMatrix<float>(
    values, colIndices, rowIndices, {spec.blockSize, spec.blockSize});

  const auto dataType = type_from_string(dtype);
  auto partitioner = PartitionerContext(num_ipus, ipu_version, spec, dataType, matmulOptions, debugName);
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
"Convert from host side sparse representation back to (block) triplets.";
Triplets triplets_from_representation(
    std::size_t num_ipus,
    std::size_t ipu_version,
    const py::object& matmulSpec,
    const std::string& dtype,
    const py::array_t<std::size_t>& metainfo,
    const py::array_t<float>& nzvalues,
    const std::string& matmulOptions,
    const std::string& debugName = "") {
  const auto spec = matmulSpecFromNamedTuple(matmulSpec);
  const auto dataType = type_from_string(dtype);
  auto partitioner = PartitionerContext(num_ipus, ipu_version, spec, dataType, matmulOptions, debugName);
  popsparse::dynamic::SparsityDataImpl<float> buckets;
  buckets.metaInfo = numpyToVector(metainfo);
  buckets.nzValues = numpyToVector(nzvalues);

  auto cooMatrix = partitioner.get().sparsityDataImplToCOOMatrix(buckets);

  auto rowIndices = PyArrayCLayout<std::size_t>(cooMatrix.rowIndices.size());
  auto colIndices = PyArrayCLayout<std::size_t>(cooMatrix.columnIndices.size());
  auto nzValues = PyArrayCLayout<float>(cooMatrix.nzValues.size());

  for( int i = 0; i < rowIndices.size(); ++i) {
    *rowIndices.mutable_data(i) = cooMatrix.rowIndices[i] / spec.blockSize;
  }

  for( int i = 0; i < colIndices.size(); ++i) {
    *colIndices.mutable_data(i) = cooMatrix.columnIndices[i] / spec.blockSize;
  }

  for( int i = 0; i < nzValues.size(); ++i) {
    *nzValues.mutable_data(i) = cooMatrix.nzValues[i];
  }

  return std::make_tuple(colIndices, rowIndices, nzValues);
}

PYBIND11_MODULE(host_utils, m) {
  m.doc() = "Host C++ code for ipu_sparse_ops module";
  m.def("initialize_device_manager", &initialize_device_manager, initialize_device_manager_doc);
  m.def("get_sparse_tensor_sizes", &get_sparse_tensor_sizes, get_sparse_tensor_sizes_doc);
  m.def("representation_from_triplets", &representation_from_triplets, py::return_value_policy::move, representation_from_triplets_doc);
  m.def("triplets_from_representation", &triplets_from_representation, py::return_value_policy::move, triplets_from_representation_doc);
}
