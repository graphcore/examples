// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cstdlib>
#include <vector>
#include <limits>
#include <chrono>

#include "IpuPathTraceJob.hpp"
#include "AsyncTask.hpp"
#include "ipu_utils.hpp"

#include <light/src/jobs.hpp>
#include <light/src/exr/exr.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pvti/pvti.hpp>

/// Convert the job lists' results (framebuffer tiles) into
/// an OpenCV image, multiplying all pixel values by the
/// specified scale factor. The image must have been
/// pre-allocated to the correct size.
void cvImageFromJobs(const std::vector<IpuPathTraceJob>& jobs,
                     cv::Mat& image, float scale) {
	#pragma omp parallel for schedule(dynamic)
	for (std::size_t j = 0; j < jobs.size(); ++j) {
		auto& job = jobs[j].jobSpec;
		job.visitPixels([&] (std::size_t row, std::size_t col, light::Vector& p) {
			const light::Vector v = p * scale;
      // We do RGB -> BGR conversion here also (OpenCV defaults to BGR):
			const auto value = cv::Vec3b(
				std::min(v(2), 255.f),
				std::min(v(1), 255.f),
				std::min(v(0), 255.f)
			);
			image.at<cv::Vec3b>(row, col) = value;
		});
	}
}

/// Extract the framebuffer tiles from the job list into a vector of vectors
/// (where each inner vector is a flattened array of the pixels from each tile).
/// Also multiplies every pixel value by the specified scale factor.
std::vector<std::vector<float>> pixelsFromJobs(const IpuJobList& jobs, float scale) {
  std::vector<std::vector<float>> tiles;
  tiles.reserve(jobs.size());
	for (std::size_t j = 0; j < jobs.size(); ++j) {
		const auto tiledPixelCount = 3 * jobs[j].jobSpec.rows() * jobs[j].jobSpec.cols();
		tiles.emplace_back(tiledPixelCount);
	}

	#pragma omp parallel for schedule(dynamic)
	for (std::size_t j = 0; j < jobs.size(); ++j) {
		auto& tile = tiles[j];
		std::size_t c = 0;
		jobs[j].jobSpec.visitPixels([&] (std::size_t, std::size_t, light::Vector& p) {
			tile[c] = p.x * scale;
			tile[c + 1] = p.y * scale;
			tile[c + 2] = p.z * scale;
			c += 3;
		});
  }
  return tiles;
}

/// Adjust samples per pixel to be multiple of samples per ipu step:
std::size_t roundSamplesPerPixel(std::size_t samplesPerPixel,
                                 std::size_t samplesPerIpuStep) {
  if (samplesPerPixel % samplesPerIpuStep) {
    samplesPerPixel += samplesPerIpuStep - (samplesPerPixel % samplesPerIpuStep);
    utils::logger()->info("Rounding SPP to next multiple of {}  (Rounded SPP :=  {})",
                          samplesPerIpuStep, samplesPerPixel);
  }
  return samplesPerPixel;
}

/// Accumulate the frame buffer data currently held in
/// each IPU job into its internal result buffer.
void accumulateResults(IpuJobList& jobs, const poplar::Target &target) {
  #pragma omp parallel for schedule(dynamic)
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    auto& job = jobs[j];
    job.prepareFrameBufferForAccess(target);
    for (auto p = 0u; p < job.frameData.size(); ++p) {
      job.jobSpec.pixels[p] += job.frameData[p];
    }
  }
}

/// Write an image in a format determined from the
/// given file name plus an HDR image in EXR format.
void writeImages(const IpuJobList& jobs,
                 std::string fileName,
                 std::size_t samplesPerPixel) {
  auto width = jobs[0].jobSpec.imageWidth;
  auto height = jobs[0].jobSpec.imageHeight;
  auto tileWidth = jobs[0].jobSpec.cols();
  auto tileHeight = jobs[0].jobSpec.rows();

  cv::Mat image(height, width, CV_8UC3);
  const auto scale = 1.f / samplesPerPixel;
  cvImageFromJobs(jobs, image, scale);
  cv::imwrite(fileName, image);
  exr::writeTiled(fileName + ".exr",
                width, height, tileWidth, tileHeight,
                pixelsFromJobs(jobs, scale));
}

/// This is the main application object. It implements the BuilderInterface
/// so that execution can be marshalled by a GraphManager object:
struct ApplicationBuilder : public utils::BuilderInterface {

  /// Constructor sets up everything that does not require the graph.
  ApplicationBuilder(const boost::program_options::variables_map& options)
  : traceChannel("ipu_path_tracer"),
    args(options),
    samplesPerPixel(args.at("samples").as<std::uint32_t>()),
    samplesPerIpuStep(args.at("samples-per-step").as<std::uint32_t>())
  {
    auto imageWidth  = args.at("width").as<std::uint32_t>();
    auto imageHeight = args.at("height").as<std::uint32_t>();
    auto tileWidth   = args.at("tile-width").as<std::uint32_t>();
    auto tileHeight  = args.at("tile-height").as<std::uint32_t>();
    auto seed = args.at("seed").as<std::uint64_t>();

    samplesPerPixel = roundSamplesPerPixel(samplesPerPixel, samplesPerIpuStep);

    pvti::Tracepoint::begin(&traceChannel, "create_path_tracing_jobs");
    traceJobs = createTracingJobs(
      imageWidth, imageHeight, tileWidth, tileHeight, samplesPerIpuStep, seed);
    utils::logger()->info("Image contains {} tiles", traceJobs.size());

    ipuJobs.reserve(traceJobs.size());
    for (auto c = 0u; c < traceJobs.size(); ++c) {
      ipuJobs.emplace_back(traceJobs[c], args, c);
    }
    pvti::Tracepoint::end(&traceChannel, "create_path_tracing_jobs");
  }

  utils::RuntimeConfig getRuntimeConfig() const override {
    auto exeName = args.at("save-exe").as<std::string>();
    if (exeName.empty()) {
      exeName = args.at("load-exe").as<std::string>();;
    }

    return utils::RuntimeConfig{
      args.at("ipus").as<std::size_t>(),
      exeName,
      args.at("model").as<bool>(),
      !args.at("save-exe").as<std::string>().empty(),
      !args.at("load-exe").as<std::string>().empty(),
      args.at("compile-only").as<bool>()
    };
  }

  /// Construct the graph and programs.
  void build(poplar::Graph& g, const poplar::Device& device) override {
    using namespace poplar;

    poprand::addCodelets(g);
    popops::addCodelets(g);
    g.addCodelets(args.at("codelet-path").as<std::string>() + "/codelets.gp");

    poplar::program::Sequence writeData;
    poplar::program::Sequence preTraceInit;
    poplar::program::Sequence initParams;
    poplar::program::Sequence allRandGenJobs;
    poplar::program::Sequence allRayTraceJobs;
    poplar::program::Sequence readResults;

    // Allow the HW RNG seed to be streamed to the IPU at runtime:
    auto seedTensor = g.addVariable(poplar::UNSIGNED_INT, {2}, "seed");
    g.setTileMapping(seedTensor, 0);
    auto seedStream = g.addHostToDeviceFIFO(
      "seed_stream", poplar::UNSIGNED_INT, seedTensor.numElements());
    initParams.add(poplar::program::Copy(seedStream, seedTensor));
    poprand::setSeed(g, seedTensor, 1u, initParams, "set_seed");

    // Allow the anti-alias scale to be streamed to the IPU at runtime:
    auto aaScaleTensor = g.addVariable(poplar::HALF, {}, "antiAliasScale");
    g.setTileMapping(aaScaleTensor, 1);
    auto aaScaleStream = g.addHostToDeviceFIFO("aa_scale_stream", poplar::HALF, 1);
    initParams.add(poplar::program::Copy(aaScaleStream, aaScaleTensor));

    pvti::Tracepoint::begin(&traceChannel, "build_path_trace_jobs");
    for (auto& j : ipuJobs) {
      j.buildGraph(g, aaScaleTensor, args);
      preTraceInit.add(j.initSequence());
      allRandGenJobs.add(j.randSequence());
      allRayTraceJobs.add(j.executeSequence());
      readResults.add(j.readSequence());
    }
    pvti::Tracepoint::end(&traceChannel, "build_path_trace_jobs");

    auto rayTraceSequence = poplar::program::Sequence(allRandGenJobs, allRayTraceJobs);
    auto executeRayTrace = poplar::program::Repeat(samplesPerIpuStep, rayTraceSequence);
    programs.add("setup", preTraceInit);
    programs.add("init_params", initParams);
    programs.add("path_trace", executeRayTrace);
    programs.add("read_framebuffer", readResults);
  }

  utils::ProgramManager& getPrograms() override { return programs; }

  /// Run the path tracing program.
  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    auto imageWidth = args.at("width").as<std::uint32_t>();
    auto imageHeight = args.at("height").as<std::uint32_t>();
    auto samplesPerPixel = args.at("samples").as<std::uint32_t>();
    auto seed = args.at("seed").as<std::uint64_t>();
    auto antiAliasingScale = args.at("aa-noise-scale").as<float>();
    auto fileName = args.at("outfile").as<std::string>();
    auto saveInterval = args.at("save-interval").as<std::uint32_t>();
    samplesPerPixel = roundSamplesPerPixel(samplesPerPixel, samplesPerIpuStep);
    const auto steps = samplesPerPixel / samplesPerIpuStep;

    engine.connectStream("seed_stream", &seed);
    std::uint16_t aaScaleHalf;
    poplar::copyFloatToDeviceHalf(device.getTarget(), &antiAliasingScale, &aaScaleHalf, 1);
    engine.connectStream("aa_scale_stream", &aaScaleHalf);
    for (auto& j : ipuJobs) {
      j.connectStreams(engine);
    }

    const auto& progs = getPrograms().getOrdinals();
    const auto TRACE_SETUP = progs.at("setup");
    const auto INIT_PARAMS = progs.at("init_params");
    const auto RAY_TRACE = progs.at("path_trace");
    const auto READ_FRAMEBUFFERS = progs.at("read_framebuffer");

    AsyncTask hostProcessing;
    const auto pixelSamplesPerStep = imageWidth * imageHeight * samplesPerIpuStep;

    utils::logger()->info("Render started");
    pvti::Tracepoint::begin(&traceChannel, "rendering");
    auto startTime = std::chrono::steady_clock::now();
    engine.run(INIT_PARAMS);
    auto pixelSamples = 0u;

    // Loop over the requisite number of steps with each step
    // computing many samples per pixel on IPU. The framebuffers
    // on the IPU are zeroed before every step and the results
    // accumulated on the CPU. This reduces the chance of overflow
    // when using fp16 framebuffers:
    for (auto step = 1u; step <= steps; ++step) {
      auto loopStartTime = std::chrono::steady_clock::now();

      // Run ray tracing on the IPU and then wait for host processing of the
      // last result to complete before reading back the next result from the IPU:
      engine.run(TRACE_SETUP);
      engine.run(RAY_TRACE);
      hostProcessing.waitForCompletion();
      engine.run(READ_FRAMEBUFFERS);

      // Asynchronously process the result on the host in a lambda function.
      // Captured variables must not be used elsewhere until waitForCompletion()
      // has returned:
      pvti::TraceChannel hostTraceChannel = {"host_processing"};
      hostProcessing.run([&, step]() {
        pvti::Tracepoint::begin(&hostTraceChannel, "accumulate_framebuffers");
        pixelSamples += samplesPerIpuStep;
        accumulateResults(ipuJobs, device.getTarget());
        pvti::Tracepoint::end(&hostTraceChannel, "accumulate_framebuffers");
        if (step % saveInterval == 0 || step == steps) {
          pvti::Tracepoint scopedTrace(&hostTraceChannel, "save_images");
          writeImages(ipuJobs, fileName, pixelSamples);
          utils::logger()->info("Saved images at step {}", step);
        }
      });

      auto loopEndTime = std::chrono::steady_clock::now();
      auto secs = std::chrono::duration<double>(loopEndTime - loopStartTime).count();
      utils::logger()->info("Completed render step {}/{} in {} seconds (Samples/sec {})",
                            step, steps, secs, pixelSamplesPerStep / secs);
    }

    hostProcessing.waitForCompletion();
    pvti::Tracepoint::end(&traceChannel, "rendering");

    auto endTime = std::chrono::steady_clock::now();
    const auto elapsedSecs = std::chrono::duration<double>(endTime - startTime).count();
    utils::logger()->info("Render finished: {} seconds", elapsedSecs);

    const std::size_t pixelsPerFrame = imageWidth * imageHeight;
    const std::size_t numTiles = ipuJobs.size();
    const double samplesPerSec = (pixelsPerFrame / elapsedSecs) * samplesPerPixel;
    const double samplesPerSecPerTile = samplesPerSec / numTiles;
    const std::size_t tilesPerIpu = device.getTarget().getTilesPerIPU();
    utils::logger()->info("Samples/sec: {}", samplesPerSec);
    utils::logger()->info("Samples/sec/tile: {}", samplesPerSecPerTile);
    utils::logger()->info("Samples/sec/IPU (projected for a fully utilised IPU): {}",
                          tilesPerIpu * samplesPerSecPerTile);
  }

private:
  pvti::TraceChannel traceChannel = {"ipu_path_tracer"};
  const boost::program_options::variables_map args;
  std::uint32_t samplesPerPixel;
  std::uint32_t samplesPerIpuStep;
  utils::ProgramManager programs;
  std::vector<TraceTileJob> traceJobs;
  IpuJobList ipuJobs;
};

/// Process the command line options for the path tracing application.
boost::program_options::variables_map parseOptions(int argc, char** argv) {
  std::string modeString;

  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
  ("help", "Show command help.")
  ("model",
   po::bool_switch()->default_value(false),
   "If set then use IPU model instead of hardware."
  )
  ("ipus",
   po::value<std::size_t>()->default_value(1),
   "Number of IPUs to use."
  )
  ("save-exe",
   po::value<std::string>()->default_value(""),
   "Save the Poplar graph executable after compilation using this name (prefix)."
  )
  ("load-exe",
   po::value<std::string>()->default_value(""),
   "Load a previously saved executable with this name (prefix) and skip graph and program construction. "
  )
  ("compile-only", po::bool_switch()->default_value(false),
   "If set and save-exe is also set then exit after compiling and saving the graph.")
  ("outfile,o", po::value<std::string>()->required(), "Set output file name.")
  ("save-interval", po::value<std::uint32_t>()->default_value(1))
  ("width,w", po::value<std::uint32_t>()->default_value(256), "Output image width (total pixels).")
  ("height,h", po::value<std::uint32_t>()->default_value(256), "Output image height (total pixels).")
  ("tile-width", po::value<std::uint32_t>()->default_value(16), "Width of tile (pixels).")
  ("tile-height", po::value<std::uint32_t>()->default_value(16), "Height of tile (pixels).")
  ("samples,s", po::value<std::uint32_t>()->default_value(512), "Samples per pixel.")
  ("samples-per-step", po::value<std::uint32_t>()->default_value(512), "Samples per IPU step.")
  ("refractive-index,n", po::value<float>()->default_value(1.5), "Refractive index.")
  ("roulette-depth", po::value<std::uint16_t>()->default_value(3), "Number of bounces before rays are randomly stopped.")
  ("stop-prob", po::value<float>()->default_value(0.2), "Probability of a ray being stopped.")
	("aa-noise-scale,a", po::value<float>()->default_value(1.0/700), "Scale for pixel space anti-aliasing noise.")
	("seed", po::value<std::uint64_t>()->default_value(1), "Seed for random number generation.")
  ("aa-noise-type", po::value<std::string>()->default_value("truncated-normal"),
   "Choose distribution for anti-aliasing noise ['uniform', 'normal', 'truncated-normal'].")
  ("float16-frame-buffer", po::bool_switch()->default_value(false),
   "Use a float16 frame buffer instead of the default float32.")
  ("codelet-path", po::value<std::string>()->default_value("./"), "Path to ray tracing codelets.");
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    throw std::runtime_error("Show help");
  }

  po::notify(vm);

  auto samplesPerStep = vm.at("samples-per-step").as<std::uint32_t>();
  if (vm.at("float16-frame-buffer").as<bool>() && samplesPerStep >= 256) {
    utils::logger()->warn("Using float16 frame-buffer with a large number of samples per step "
                          "risks oversaturating the frame-buffer. (samples-per-step: {}).",
                          samplesPerStep);
  }

  // Check options are set correctly:
  auto saveExe = !vm.at("save-exe").as<std::string>().empty();
  auto loadExe = !vm.at("load-exe").as<std::string>().empty();
  if (saveExe && loadExe) {
    throw std::logic_error("You can not set both save-exe and load-exe.");
  }

  return vm;
}

/// Boiler plate code to set-up logging and formatting then
/// run the application via a GraphManager:
int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%H:%M:%S.%f] [%L] [%t] %v");

  ApplicationBuilder builder(parseOptions(argc, argv));
  return utils::GraphManager().run(builder);
}
