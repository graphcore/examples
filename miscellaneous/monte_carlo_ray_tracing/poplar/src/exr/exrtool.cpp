// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// Tool for applying various simple processing operations
// to EXR image files.
#include <cstdlib>
#include <vector>
#include <iostream>

#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../logging.hpp"
#include <light/src/exr/exr.hpp>

std::shared_ptr<spdlog::logger> logger() {
    static auto logger = spdlog::stdout_logger_mt("exrtool_logger");
    return spdlog::get("exrtool_logger");
}

boost::program_options::variables_map
parseOptions(int argc, char** argv) {

  namespace po = boost::program_options;
  po::options_description desc("Program for manipulating EXR files. Options");
  desc.add_options()
  ("help", "Show command help.")
  ("inputs,i", po::value<std::vector<std::string>>()->required()->multitoken(),
   "Input file(s)")
  ("output,o", po::value<std::string>()->required(), "Set the output file name.")
  ("op", po::value<std::string>()->default_value("error"),
    "Select processing operation from ['error', 'mean']."
    "\t\nerror: Compute the MSE and PSNR between two images (the first image is considered to be the noise free one)."
    "\t\nmean: Compute the mean of a list of images.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    throw std::runtime_error("Show help");
  }

  po::notify(vm);

  auto numInputs = vm.at("inputs").as<std::vector<std::string>>().size();
  auto op = vm.at("op").as<std::string>();
  if (op == "error" && numInputs != 2) {
    logger()->error("'{}' operation requires exactly 2 inputs.", op);
    throw std::logic_error("Incorrect number of inputs for op.");
  }

  if (op != "error" && op != "mean") {
    logger()->error("Unknown operation '{}'.", op);
    throw std::logic_error("Unknown image operation");
  }

  return vm;
}

// Compute mean of all the images (e.g. use to combine
// parallel renders of same scene):
exr::Image meanImage(const std::vector<exr::Image>& images) {
  exr::Image mean(images.front().width, images.front().height);
  for (const auto& image : images) {
    for (auto& p : image.slices) {
      mean.slices[p.first] = std::vector<float>(p.second.size());
    }
  }

  logger()->info("Averaging {} images", images.size());
  const float scale = 1.f / images.size();
  for (const auto& image : images) {
    for (auto& p : image.slices) {
      const auto& name = p.first;
      auto& result = mean.slices.at(name);
      const auto& data = image.slices.at(name);
      for (auto i = 0u; i < data.size(); ++i) {
        result[i] += scale * data[i];
      }
    }
  }
  return mean;
}

struct Stats {
  double MSE;
  double PSNR;
};

// Compute difference image, mean squared error (MSE), and
// the peak signal to noise ratio (PSNR) between two input
// images. PSNR assumes the first image is the target/ground
// truth and so takes the max value from that image:
exr::Image errorImage(const std::vector<exr::Image>& images, Stats& errors) {
  auto err = images.front();
  auto& b = images.back();
  errors.MSE = 0.f;
  double max = 0.f;
  for (auto& p : err.slices) {
    const auto& name = p.first;
    auto& dataA = err.slices.at(name);
    auto& dataB = b.slices.at(name);
    if (dataA.size() != dataB.size()) {
      logger()->error("Image sizes do not match.");
      throw std::runtime_error("Incompatible image dimensions");
    }
    for (auto i = 0u; i < dataA.size(); ++i) {
      max = std::max(max, (double)dataA[i]);
      dataA[i] -= dataB[i];
      errors.MSE += dataA[i] * dataA[i];
    }
  }
  errors.MSE /= err.width * err.height;
  errors.PSNR = -10.0 * log10(errors.MSE / (max * max));
  return err;
}

int main(int argc, char** argv) {
  try {
    auto args = parseOptions(argc, argv);
    auto inputFiles = args.at("inputs").as<std::vector<std::string>>();
    auto outputFile = args.at("output").as<std::string>();
    auto op = args.at("op").as<std::string>();

    std::vector<exr::Image> images;
    images.reserve(inputFiles.size());
    for (const auto& fileName : inputFiles) {
      images.push_back(exr::read(fileName));
    }

    if (op == "error") {
      Stats errors;
      auto diff = errorImage(images, errors);
      logger()->info("Statistics: MSE {} PSNR {} dB", errors.MSE, errors.PSNR);
      exr::write(outputFile, diff);
    }

    if (op == "mean") {
      auto mean = meanImage(images);
      exr::write(outputFile, mean);
    }
  } catch (const std::exception& e) {
    logger()->error("{}", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
