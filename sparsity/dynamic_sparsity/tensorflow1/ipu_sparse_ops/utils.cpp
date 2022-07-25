// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "utils.hpp"

#include <popnn/Pooling.hpp>
#include <popnn/PoolingDef.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/Loss.hpp>
#include <popops/ElementWise.hpp>

poplar::Type type_from_string(const std::string& dtype) {
  static const std::map<std::string, poplar::Type> types = {
    {"<dtype: 'float16'>", poplar::HALF},
    {"<dtype: 'float32'>", poplar::FLOAT}
  };

  try {
    return types.at(dtype);
  } catch (const std::exception& e) {
    throw std::runtime_error("Conversion to Poplar type not supported for: " + dtype);
  }
}

float computeDensity(const SparseArgs& args) {
  if (args.input_size % args.block_size != 0) {
    throw std::logic_error("Input size must be divisible by block size.");
  }
  if (args.output_size % args.block_size != 0) {
    throw std::logic_error("Input size must be divisible by block size.");
  }
  float numBlocks = (args.input_size / args.block_size) * (args.output_size / args.block_size);
  return args.max_non_zeros / numBlocks;
}

SparseArgs read_json_args(const std::string& attributes) {
  SparseArgs args;
  try {
    std::stringstream json_args(attributes);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json_args, pt);
    args.block_size = pt.get<std::size_t>("block_size");
    args.batch_size = pt.get<std::size_t>("batch_size");
    args.input_size = pt.get<std::size_t>("input_size");
    args.output_size = pt.get<std::size_t>("output_size");
    args.max_non_zeros = pt.get<std::size_t>("max_non_zero_blocks");
    args.group_size = pt.get<std::size_t>("group_size");
    args.data_type = type_from_string(pt.get<std::string>("data_type"));
    args.matmul_options = pt.get<std::string>("matmul_options");
    args.dense_grad_matmul_options = pt.get<std::string>("dense_grad_matmul_options");
    args.pooling_type = pt.get<std::string>("pooling_type");
    args.embedding_grad_scale = pt.get<float>("embedding_grad_scale");
    args.debug_printing = pt.get<bool>("debug_printing");
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(
      "Error reading custom op's JSON attributes: ") + e.what()
      + "\nJSON input: " + attributes);
  }

  return args;
}

static poplar::OptionFlags getDefaultOptions(const std::string& defaults, const std::string& jsonOptions) {
  poplar::OptionFlags options;
  poplar::readJSON(defaults, options);

  // Overwrite defaults with contents of config string:
  if (!jsonOptions.empty()) {
    poplar::readJSON(jsonOptions, options);
  }

  return options;
}

poplar::OptionFlags getSparseMulDefaultOptions(const std::string& jsonOptions) {
  return getDefaultOptions(sparse_defaults, jsonOptions);
}

poplar::OptionFlags getDenseGradMulDefaultOptions(const std::string& jsonOptions) {
  return getDefaultOptions(dense_defaults, jsonOptions);
}

poplar::Tensor topKIndices(
    poplar::Graph& graph,
    unsigned k,
    const poplar::Tensor gradient,
    poplar::program::Sequence &prog,
    const std::string& debug_prefix) {
  // top-k only works on flat indices:
  auto absGradsFlat = popops::abs(
    graph, gradient.reshape({1, gradient.numElements()}), prog);
  auto indices = poplar::Tensor();
  popnn::topK(graph, absGradsFlat, indices, k, true, prog, debug_prefix + "/indices");
  return indices;
}

poplar::Tensor serializedMatmul(poplar::Graph& graph, poplar::program::Sequence& prog,
                                poplar::Tensor& A, poplar::Tensor& B,
                                std::size_t inSplits, std::size_t outSplits,
                                const std::string& debug_prefix,
                                bool enableSerialization,
                                const poplar::OptionFlags& options,
                                poplin::matmul::PlanningCache* cache) {
  if (enableSerialization && (inSplits > 1 || outSplits > 1)) {
    // Define the shape of the output Tensor
    std::vector<std::size_t> resultShape {A.dim(0), B.dim(1)};

    // Create the vector of dimensions along which to slice and the slice size
    std::size_t inSliceSize = resultShape[0] / inSplits;
    std::size_t outSliceSize = resultShape[1] / outSplits;


    auto result = graph.addVariable(A.elementType(),
                                    resultShape,
                                    poplar::VariableMappingMethod::LINEAR,
                                    debug_prefix + "/result");

    for (std::size_t i = 0; i < inSplits; i++) {
      auto A_slice_i = A.slice(i * inSliceSize, (i + 1) * inSliceSize, 0);
      for (std::size_t j = 0; j < outSplits; j++) {
        auto B_slice_j = B.slice(j * outSliceSize, (j + 1) * outSliceSize, 1);
        // Compute matmul for chunk ij
        auto chunk_ij =
          poplin::matMul(graph, A_slice_i, B_slice_j,
                         prog, debug_prefix +
                         "/chunk_" + std::to_string(i)
                         + "_" + std::to_string(j),
                         options, cache);

        // Assign resulting slice
        auto resultSlice_ij =
          result.slice({(unsigned)(i * inSliceSize), (unsigned)(j * outSliceSize)},
                                     {(unsigned)((i + 1) * inSliceSize), (unsigned)((j + 1) * outSliceSize)});

        prog.add(poplar::program::Copy(chunk_ij, resultSlice_ij));
      }
    }
    return result;
  }
  else {
    return poplin::matMul(graph, A, B, prog, debug_prefix + "/matmul", options, cache);
  }
}

poplar::Tensor pool(
  poplar::Graph& graph,
    std::string poolingType,
    std::size_t blockSize,
    const poplar::Tensor gradient,
    poplar::program::Sequence &prog,
    const std::string& debug_prefix) {

  // If no pooling is required or the blocksize is one
  // (meaning this is not block sparse)
  // return the gradients as is
  if (poolingType == "NONE" || blockSize == 1) {
    return gradient;
  }

  // We use the SUM pooling type for SUM, and also AVG, to allow granular control
  // over when the scaling occurs
  auto poolTypeParam = popnn::PoolingType::SUM;

  if (poolingType == "MAX") {
    poolTypeParam = popnn::PoolingType::MAX;
  }
  else if (poolingType == "AVG") {
    poolTypeParam = popnn::PoolingType::AVG;
  }

  // Reduce across each block.
  // Reshape to give blocks as input field, with batch size
  // the number of blocks and 1 channel.
  auto reshaped_grads =
    gradient.reshape({gradient.shape()[0] / blockSize, blockSize,
                      gradient.shape()[1] / blockSize, blockSize})
            .dimShuffle({0, 2, 1, 3}).flatten(0, 2).expand({1});
  // Take abs value of grads before any of the pooling operations:
  auto abs_reshaped_grads = popops::abs(graph, reshaped_grads, prog);

  // Create pooling parameters
  const std::vector<std::size_t> inputFieldShape = {blockSize, blockSize};
  const auto kernelShape = inputFieldShape;
  const std::vector<unsigned> stride = {1, 1};
  const auto inputTruncationOrPaddingLower = std::vector<int> {0, 0};
  const auto inputTruncationOrPaddingUpper = std::vector<int> {0, 0};
  const std::size_t numChannels = 1;
  const std::size_t batchSize = abs_reshaped_grads.dim(0);
  auto poolParams =
    popnn::pooling::PoolParams(poolTypeParam, inputFieldShape,
                               kernelShape, stride,
                               inputTruncationOrPaddingLower,
                               inputTruncationOrPaddingUpper,
                               numChannels, batchSize, gradient.elementType());
  // Perform pooling
  // Set introspection option to false to work around compile time issue
  // when it is enabled (the default as it avoids extra re-arrangement):
  const poplar::OptionFlags poolOptions({{"poolUseIntrospectiveMapping", "false"}});
  auto pooledTensor = popnn::pooling::pool(
      graph, poolParams, abs_reshaped_grads, prog,
      debug_prefix + "/gradient_pooling", poolOptions);

  // Recover output shape.
  pooledTensor = pooledTensor.reshape({gradient.shape()[0] / blockSize,
                                       gradient.shape()[1] / blockSize});

  return pooledTensor;
}
