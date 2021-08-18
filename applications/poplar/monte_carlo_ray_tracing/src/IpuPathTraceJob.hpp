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

struct IpuPathTraceJob;
using ProgramList = std::vector<poplar::program::Program>;
using IpuJobList = std::vector<IpuPathTraceJob>;

struct StreamHandle {
  std::string name;
  poplar::DataStream stream;
};

using Interval = std::pair<std::size_t, std::size_t>;

/// Compute the start and end indices that can be used to slice the
/// tile's pixels into chunks that each worker will process:
std::vector<Interval> splitTilePixelsOverWorkers(std::size_t rows, std::size_t cols,
                                                 std::size_t numChannels,
                                                 std::size_t workers) {
  const auto rowsPerWorker = rows / workers;
  const auto leftOvers = rows % workers;
  std::vector<std::size_t> work(workers, rowsPerWorker);

  // Distribute leftovers amongst workers:
  for (auto i=0u; i < leftOvers; ++i) {
    work[i] += 1;
  }

  // Turn list of rows per worker into element intervals:
  std::vector<Interval> intervals;
  intervals.reserve(workers);
  auto start = 0u;
  const auto pixelsPerRow = numChannels * cols;
  for (auto w : work) {
    auto end = start + (pixelsPerRow * w);
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

  // Hard code the maximum number of samples that a single path
  // could need if it reached maximum depth.
  static constexpr std::size_t maxRandomNumbersPerSample = 6 * 6;
  static constexpr std::size_t numChannels = 3;
  static constexpr std::size_t numComponents = 3;

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

  IpuPathTraceJob(TraceTileJob& spec,
                 const boost::program_options::variables_map& args,
                 std::size_t core)
  :
    jobSpec(spec),
    ipuCore(core),
    frameData(spec.rows() * spec.cols())
  {
    // Constructor only initialises values that are independent of graph construction.
    // The buildGraph() method constructs the Poplar graph components: graph execution and graph
    // construction are completely separated so that buildGraph() can be skipped when loading a
    // pre-compiled executable:
    const auto suffix = jobStringSuffix();
    frameBufferRead.name = "fb_read" + suffix;
    // If we use a half-precision frame buffer then we need
    // an extra buffer to hold the data we read back
    // before conversion to single precision:
    auto fp16Framebuffer = args.at("float16-frame-buffer").as<bool>();
    if (fp16Framebuffer) {
      halfFrameData.resize(numChannels * spec.rows() * spec.cols());
    }
  }

  void buildGraph(poplar::Graph& graph, poplar::Tensor aaScaleTensor,
                  const boost::program_options::variables_map& args) {
    const auto suffix = jobStringSuffix();

    // Create buffers for primary rays and the output frame:
    auto fp16Framebuffer = args.at("float16-frame-buffer").as<bool>();
    const auto frameBufferType = fp16Framebuffer ? poplar::HALF : poplar::FLOAT;
    cameraRays  = graph.addVariable(poplar::HALF, {numComponents * frameData.size()}, "cam_rays" + suffix);
    frameBuffer = graph.addVariable(frameBufferType, {numChannels * frameData.size()}, "frame_buffer" + suffix);
    // Create stream that allows reading of the frame buffer:
    frameBufferRead.stream  = graph.addDeviceToHostFIFO(frameBufferRead.name, frameBufferType, frameBuffer.numElements());

    const auto randomCount = jobSpec.rows() * jobSpec.cols() * maxRandomNumbersPerSample;
    randUniform_0_1 = graph.addVariable(poplar::HALF, {randomCount}, "random" + suffix);
    auto genRays = graph.addComputeSet("ray_gen" + suffix);
    rayGenVertex = graph.addVertex(genRays, "GenerateCameraRays");
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
    const auto intervals = splitTilePixelsOverWorkers(jobSpec.rows(), jobSpec.cols(), numChannels, workers);
    auto pathTrace = graph.addComputeSet("path_trace" + suffix);
    tracerVertices.reserve(workers);
    auto computeVertexName = fp16Framebuffer ? "RayTraceKernel<half>" : "RayTraceKernel<float>";
    for (auto i = 0u; i < workers; ++i) {
      tracerVertices.push_back(graph.addVertex(pathTrace, computeVertexName));
      auto& v = tracerVertices.back();
      addScalarConstant(graph, v, "refractiveIndex", poplar::HALF,
                        args.at("refractive-index").as<float>());
      addScalarConstant(graph, v, "rouletteDepth", poplar::UNSIGNED_SHORT,
                        args.at("roulette-depth").as<std::uint16_t>());
      addScalarConstant(graph, v, "stopProb", poplar::HALF,
                        args.at("stop-prob").as<float>());

      auto interval = intervals[i];
      graph.connect(v["cameraRays"], cameraRays.slice(interval.first, interval.second));
      graph.connect(v["frameBuffer"], frameBuffer.slice(interval.first, interval.second));
    }

    setTileMappings(graph);

    // Build the programs:

    // Assign any modifiable parameters and init the frame buffer to zero:
    initSeq.add(poplar::program::Copy(aaScaleTensor, localAaScale));
    popops::fill(graph, frameBuffer, initSeq, 0.f, "zero_framebuffer");

    // Program to generate enough random numbers to supply one path tracing sample per pixel:
    auto aaNoiseType = args.at("aa-noise-type").as<std::string>();
    auto randForAntiAliasing = buildAaNoise(
      graph, randUniform_0_1, randSeq, aaNoiseType, "generate_aa_noise" + suffix);
    graph.connect(rayGenVertex["antiAliasNoise"], randForAntiAliasing);

    randUniform_0_1 = poprand::uniform(
      graph, nullptr, 0u, randUniform_0_1, poplar::HALF, 0.f, 1.f,
      randSeq, "generate_uniform_0_1" + suffix);

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
    execSeq.add(poplar::program::Execute(pathTrace));
    execSeq.add(poplar::program::WriteUndef(randUniform_0_1));

    // Program to read back the frame buffer:
    readSeq.add(poplar::program::Copy(frameBuffer, frameBufferRead.stream));
  }

  poplar::program::Sequence initSequence() const { return initSeq; }
  poplar::program::Sequence randSequence() const { return randSeq; }
  poplar::program::Sequence executeSequence() const { return execSeq; }
  poplar::program::Sequence readSequence() const { return readSeq; }

  void connectStreams(poplar::Engine& engine) {
    if (!halfFrameData.empty()) {
      // If using half frame-buffer we need to read back into a
      // separate buffer that can later be cast to single precision
      // frame data:
      engine.connectStream(frameBufferRead.name, halfFrameData.data());
    } else {
      // Otherwise we can just read directly into the single precision frame data:
      engine.connectStream(frameBufferRead.name, frameData.data());
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
  StreamHandle frameBufferRead;

  // Member variables below only get assigned during graph construction
  // (which is skipped if we load a precompiled executable):
  poplar::Tensor cameraRays;
  poplar::Tensor frameBuffer;
  poplar::Tensor randUniform_0_1;

  poplar::VertexRef  rayGenVertex;
  std::vector<poplar::VertexRef> tracerVertices;

  poplar::program::Sequence initSeq;
  poplar::program::Sequence randSeq;
  poplar::program::Sequence execSeq;
  poplar::program::Sequence readSeq;

  /// Set the tile mapping for all variables and vertices:
  void setTileMappings(poplar::Graph& graph) {
    graph.setTileMapping(cameraRays, ipuCore);
    graph.setTileMapping(frameBuffer, ipuCore);
    graph.setTileMapping(randUniform_0_1, ipuCore);
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
