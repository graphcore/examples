// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poprand/RandomGen.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <poprand/codelets.hpp>
#include <popops/codelets.hpp>

#include <light/src/light.hpp>
#include <light/src/jobs.hpp>

#include "ipu_utils.hpp"

#include <boost/program_options.hpp>

struct IpuPathTraceJob;
using ProgramList = std::vector<poplar::program::Program>;
using IpuJobList = std::vector<IpuPathTraceJob>;

using Interval = std::pair<std::size_t, std::size_t>;

/// Compute the start and end indices that can be used to slice the
/// tile's pixels into chunks that each worker will process:
std::vector<Interval> splitTilePixelsOverWorkers(std::size_t rows, std::size_t cols,
                                                 std::size_t workers) {
  const auto rowsPerWorker = rows / workers;
  const auto leftOvers = rows % workers;
  std::vector<std::size_t> work(workers, rowsPerWorker);

  // Distribute leftovers amongst workers:
  for (auto i = 0u; i < leftOvers; ++i) {
    work[i] += 1;
  }

  // Turn list of rows per worker into element intervals:
  std::vector<Interval> intervals;
  intervals.reserve(workers);
  auto start = 0u;
  for (auto w : work) {
    auto end = start + (cols * w);
    intervals.emplace_back(start, end);
    start = end;
  }
  return intervals;
}

poplar::Tensor buildAaNoise(poplar::Graph& graph, poplar::Tensor& layoutTensor,
                            poplar::program::Sequence& prog, std::string aaNoiseType,
                            std::string debugString) {
  if (aaNoiseType == "uniform") {
    return poprand::uniform(
      graph, nullptr, 0u, layoutTensor, poplar::HALF, -1.f, 1.f,
      prog, debugString);
  } else if (aaNoiseType == "normal") {
    return poprand::normal(
      graph, nullptr, 0u, layoutTensor, poplar::HALF, 0.f, 1.f,
      prog, debugString);
  } else if (aaNoiseType == "truncated-normal") {
    return poprand::truncatedNormal(
      graph, nullptr, 0u, layoutTensor, poplar::HALF, 0.f, 1.f, 3.f,
      prog, debugString);
  } else {
    throw std::runtime_error("Invalid AA noise type: " + aaNoiseType);
  }
}

// Class that describes and builds the compute graph and programs for a
// path tracing job. Each job traces rays for a small sub-tile of the
// whole image.
struct IpuPathTraceJob {
  TraceTileJob& jobSpec;
  const std::size_t ipuCore; // Core instead of 'Tile' to avoid confusion with the image tiles.
  std::vector<light::Vector> frameData;
  ipu_utils::StreamableTensor frameBuffer;

  // Hard code the maximum number of samples that a single path
  // could need if it reached maximum depth.
  static constexpr std::size_t maxRandomNumbersPerSample = 6 * 6;
  static constexpr std::size_t numChannels = 3;
  static constexpr std::size_t numRayDirComponents = 2;

  // Utility to add a scalar constant to the graph and map it to the IPU tile
  // for this job:
  template <typename T>
  poplar::Tensor
  addScalarConstant(poplar::Graph& graph, poplar::VertexRef v, std::string field, poplar::Type type, T value) {
    auto t = graph.addConstant(type, {}, value);
    graph.connect(v[field], t);
    graph.setTileMapping(t, ipuCore);
    return t;
  }

  // Constructor only initialises values that are independent of graph construction.
  // The buildGraph() method constructs the Poplar graph components: graph execution and graph
  // construction are completely separated so that buildGraph() can be skipped when loading a
  // pre-compiled executable.
  IpuPathTraceJob(TraceTileJob& spec,
                 const boost::program_options::variables_map& args,
                 std::size_t core)
  :
    jobSpec(spec),
    ipuCore(core),
    frameData(spec.rows() * spec.cols()),
    frameBuffer("frame_buffer" + jobStringSuffix())
  {
    // If we use a half-precision frame buffer then we need
    // an extra buffer to hold the data we read back
    // before conversion to single precision:
    auto fp16Framebuffer = args.at("float16-frame-buffer").as<bool>();
    if (fp16Framebuffer) {
      halfFrameData.resize(numChannels * spec.rows() * spec.cols());
    }
  }

  bool useSimd(poplar::Graph& graph, const boost::program_options::variables_map& args) const {
    if (graph.getTarget().getTargetType() == poplar::TargetType::IPU) {
      return args.at("use-simd").as<bool>();
    }
    return false;
  }

  void buildGraph(poplar::Graph& graph, poplar::Tensor aaScaleTensor,
                  const boost::program_options::variables_map& args) {
    const auto suffix = jobStringSuffix();

    // Create buffers for primary rays and the output frame:
    auto fp16Framebuffer = args.at("float16-frame-buffer").as<bool>();
    const auto frameBufferType = fp16Framebuffer ? poplar::HALF : poplar::FLOAT;
    cameraRays  = graph.addVariable(poplar::HALF, {numRayDirComponents * frameData.size()}, "cam_rays" + suffix);
    frameBuffer.buildTensor(graph, frameBufferType, {numChannels * frameData.size()});
    // Create stream that allows reading of the frame buffer:
    auto frameBufferCopy = frameBuffer.buildRead(graph, true);

    const auto randomCount = jobSpec.rows() * jobSpec.cols() * maxRandomNumbersPerSample;
    randUniform_0_1 = graph.addVariable(poplar::HALF, {randomCount}, "random" + suffix);
    auto genRays = graph.addComputeSet("ray_gen" + suffix);
    const bool enableSimd = useSimd(graph, args);
    if (enableSimd && ((jobSpec.endCol - jobSpec.startCol) % 2 || numRayDirComponents % 2)) {
      // Tile width must be a multiple of 2 so we can guarantee output row pointers for every worker are
      // always offset by a multiple of 2*numRayDirComponents*sizeof(half) == 8-bytes (see codelets.cpp).
      throw std::logic_error("The tile width must be a multiple of 2 to use the SIMD codelets.");
    }
    const auto rayGenVertexName = enableSimd ? "GenerateCameraRaysSIMD" : "GenerateCameraRays";
    rayGenVertex = graph.addVertex(genRays, rayGenVertexName);
    graph.setPerfEstimate(rayGenVertex, 1); // Fake perf estimate (for IpuModel only).
    addScalarConstant<unsigned>(graph, rayGenVertex, "startRow", poplar::UNSIGNED_INT, jobSpec.startRow);
    addScalarConstant<unsigned>(graph, rayGenVertex, "startCol", poplar::UNSIGNED_INT, jobSpec.startCol);
    addScalarConstant<unsigned>(graph, rayGenVertex, "endRow", poplar::UNSIGNED_INT, jobSpec.endRow);
    addScalarConstant<unsigned>(graph, rayGenVertex, "endCol", poplar::UNSIGNED_INT, jobSpec.endCol);
    addScalarConstant<unsigned>(graph, rayGenVertex, "imageWidth", poplar::UNSIGNED_INT, jobSpec.imageWidth);
    addScalarConstant<unsigned>(graph, rayGenVertex, "imageHeight", poplar::UNSIGNED_INT, jobSpec.imageHeight);
    graph.connect(rayGenVertex["rays"], cameraRays);

    // Make a local copy of AA scale:
    auto localAaScale =
      graph.addVariable(aaScaleTensor.elementType(), aaScaleTensor.shape(), "antiAliasScale" + suffix);
    graph.setTileMapping(localAaScale, ipuCore);
    graph.connect(rayGenVertex["antiAliasScale"], localAaScale);

    // Decide which chunks of the image-tile workers will process:
    const auto workers = graph.getTarget().getNumWorkerContexts();
    auto pathTrace = graph.addComputeSet("path_trace" + suffix);
    tracerVertices.reserve(workers);
    auto computeVertexName = fp16Framebuffer ? "RayTraceKernel<half>" : "RayTraceKernel<float>";
    const auto intervals = splitTilePixelsOverWorkers(jobSpec.rows(), jobSpec.cols(), workers);
    for (const auto &interval : intervals) {
      tracerVertices.push_back(graph.addVertex(pathTrace, computeVertexName));
      auto& v = tracerVertices.back();
      addScalarConstant(graph, v, "refractiveIndex", poplar::HALF,
                        args.at("refractive-index").as<float>());
      addScalarConstant(graph, v, "rouletteDepth", poplar::UNSIGNED_SHORT,
                        args.at("roulette-depth").as<std::uint16_t>());
      addScalarConstant(graph, v, "stopProb", poplar::HALF,
                        args.at("stop-prob").as<float>());
      graph.connect(v["cameraRays"], cameraRays.slice(interval.first * numRayDirComponents, interval.second * numRayDirComponents));
      graph.connect(v["frameBuffer"], frameBuffer.get().slice(interval.first * numChannels, interval.second * numChannels));
    }

    auto aaSampleCount = numRayDirComponents * jobSpec.rows() * jobSpec.cols();
    // Make sure number of samples is a multiple of the number of workers
    // number of rows doesn't need to be even though it maximises utilisation.
    const auto numWorkers = graph.getTarget().getNumWorkerContexts();
    if (aaSampleCount % numWorkers) {
      aaSampleCount += numWorkers - aaSampleCount % numWorkers;
    }
    randForAntiAliasing = graph.addVariable(poplar::HALF, {aaSampleCount}, "aa_noise_layout");

    setTileMappings(graph);

    // Build the programs:

    // Assign any modifiable parameters and init the frame buffer to zero:
    initSeq.add(poplar::program::Copy(aaScaleTensor, localAaScale));
    popops::fill(graph, frameBuffer, initSeq, 0.f, "zero_framebuffer");

    // Program to generate the anti-aliasing samples:
    auto aaNoiseType = args.at("aa-noise-type").as<std::string>();
    randForAntiAliasing = buildAaNoise(graph, randForAntiAliasing, genAaSamples, aaNoiseType, "generate_aa_noise" + suffix);
    graph.connect(rayGenVertex["antiAliasNoise"], randForAntiAliasing);

    // Program to generate the path tracing samples:
    poplar::program::Sequence genUniformSamples;
    randUniform_0_1 = poprand::uniform(
      graph, nullptr, 0u, randUniform_0_1, poplar::HALF, 0.f, 1.f,
      genUniformSamples, "generate_uniform_0_1" + suffix);

    // Need to slice the random numbers between vertices. This is simpler than
    // splitting the pixels because we chose num elements to divide exactly:
    if (randUniform_0_1.numElements() % workers != 0) {
      throw std::logic_error("Size of random data must be divisible by number of workers.");
    }
    auto start = 0u;
    const auto inc = randUniform_0_1.numElements() / workers;
    auto end = inc;
    for (auto i = 0u; i < workers; ++i) {
      graph.connect(tracerVertices[i]["uniform_0_1"], randUniform_0_1.slice(start, end));
      start = end;
      end += inc;
    }

    // Program to perform path tracing:
    execSeq.add(poplar::program::Execute(genRays));
    execSeq.add(poplar::program::WriteUndef(randForAntiAliasing));
    // Generate the uniform samples after we have undefined the anti-aliasing
    // noise to reduce live memory:
    execSeq.add(genUniformSamples);
    execSeq.add(poplar::program::Execute(pathTrace));
    execSeq.add(poplar::program::WriteUndef(randUniform_0_1));

    // Program to read back the frame buffer:
    readSeq.add(frameBufferCopy);
  }

  poplar::program::Sequence initSequence() const { return initSeq; }
  poplar::program::Sequence randSequence() const { return genAaSamples; }
  poplar::program::Sequence executeSequence() const { return execSeq; }
  poplar::program::Sequence readSequence() const { return readSeq; }

  void connectStreams(poplar::Engine& engine) {
    if (!halfFrameData.empty()) {
      // If using half frame-buffer we need to read back into a
      // separate buffer that can later be cast to single precision
      // frame data:
      frameBuffer.connectReadStream(engine, halfFrameData.data());
    } else {
      // Otherwise we can just read directly into the single precision frame data:
      frameBuffer.connectReadStream(engine, frameData.data());
    }
  }

  void prepareFrameBufferForAccess(const poplar::Target &target) {
    if (!halfFrameData.empty()) {
      poplar::copyDeviceHalfToFloat(
        target, halfFrameData.data(), reinterpret_cast<float*>(frameData.data()), halfFrameData.size());
    }
  }

  ~IpuPathTraceJob() {}

private:
  std::vector<std::uint16_t> halfFrameData;

  // Member variables below only get assigned during graph construction
  // (which is skipped if we load a precompiled executable):
  poplar::Tensor cameraRays;
  poplar::Tensor randUniform_0_1;
  poplar::Tensor randForAntiAliasing;

  poplar::VertexRef  rayGenVertex;
  std::vector<poplar::VertexRef> tracerVertices;

  poplar::program::Sequence initSeq;
  poplar::program::Sequence genAaSamples;
  poplar::program::Sequence execSeq;
  poplar::program::Sequence readSeq;

  /// Set the tile mapping for all variables and vertices:
  void setTileMappings(poplar::Graph& graph) {
    graph.setTileMapping(cameraRays, ipuCore);
    graph.setTileMapping(frameBuffer, ipuCore);
    graph.setTileMapping(randUniform_0_1, ipuCore);
    graph.setTileMapping(randForAntiAliasing, ipuCore);
    graph.setTileMapping(rayGenVertex, ipuCore);
    for (auto& v : tracerVertices) {
      graph.setTileMapping(v, ipuCore);
      graph.setPerfEstimate(v, 1); // Fake perf estimate (for IpuModel only).
    }
  }

  std::string jobStringSuffix() {
    return "_core" + std::to_string(ipuCore);
  }
};
