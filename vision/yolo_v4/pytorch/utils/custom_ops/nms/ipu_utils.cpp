// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "ipu_utils.hpp"

#include <cmath>
#include <iostream>

Device getDevice() {

  auto dm = DeviceManager::createDeviceManager();
  auto devices = dm.getDevices(TargetType::IPU, 1);
  std::cerr << "Found " << devices.size() << " devices ..." << std::endl;

  if (devices.size() > 0) {
    for (auto &d : devices) {
      if (d.attach()) {
        std::cerr << "Using hw device " << d.getId() << " ...\n";
        return std::move(d);
      }
    }
  }
  std::cerr << "Using IpuModel ...\n";
  IPUModel ipuModel;
  return ipuModel.createDevice({}, false);
}

Device getDevice(uint32_t tilesPerIPU) {
  auto dm = DeviceManager::createDeviceManager();
  auto devices = dm.getDevices(TargetType::IPU, 1);
  std::cout << "Found " << devices.size() << " devices ..." << std::endl;

  for (auto &device : devices) {
    if (device.attach()) {
      std::cout << "Using hw device " << device.getId() << " ...\n";
      if (tilesPerIPU != 0 &&
          tilesPerIPU != device.getTarget().getTilesPerIPU()) {
        device = device.createVirtualDevice(tilesPerIPU);
      }
      return std::move(device);
    }
  }
  std::cerr << "Using IpuModel ...\n";
  IPUModel ipuModel;
  return ipuModel.createDevice({}, false);
}

std::vector<uint16_t> to_half(const Target &target,
                              const std::vector<float> &vec) {
  std::vector<uint16_t> res(vec.size());
  poplar::copyFloatToDeviceHalf(target, vec.data(), res.data(), vec.size());
  return res;
}

std::vector<float> to_float(const Target &target,
                            const std::vector<uint16_t> &vec) {
  std::vector<float> res(vec.size());
  poplar::copyDeviceHalfToFloat(target, vec.data(), res.data(), vec.size());
  return res;
}

std::vector<std::vector<poplar::Interval>>
createTileMapping(uint32_t size, uint32_t numTiles, uint32_t totalTiles,
                  uint32_t firstTile) {
  std::vector<std::vector<poplar::Interval>> res(totalTiles);
  uint32_t offset = 0;
  uint32_t step = size / numTiles;
  uint32_t remainder = size % numTiles;
  size_t threshold = numTiles - remainder;
  uint32_t last = 0;
  for (size_t i = 0; i < numTiles; ++i) {
    if (i == threshold)
      ++step;
    last = offset + step;
    res[i + firstTile].push_back({offset, last});
    offset += step;
  }
  assert(last == size);
  return res;
}

uint32_t getNumTiles(uint32_t size, uint32_t totalTiles, uint32_t numWorkers,
                     uint32_t loadWorker) {
  float p = std::max(std::log(float(size) / float(numWorkers * loadWorker)) /
                         std::log(2.0),
                     0.0);
  uint32_t numTiles = uint32_t(std::pow(2.0, std::floor(p)));
  numTiles = std::min(numTiles, totalTiles);
  return numTiles;
}

std::vector<uint32_t> tileSize(const std::vector<uint32_t> &size,
                               uint32_t totalTiles, uint32_t numWorkers,
                               uint32_t loadWorker) {
  uint32_t total = 0;
  for (const auto &s : size)
    total += s;

  float p =
      std::log(float(total) / float(numWorkers * loadWorker)) / std::log(2.0);
  uint32_t numTiles = std::min(
      std::max(uint32_t(std::pow(2.0, std::floor(p))), uint32_t(size.size())),
      totalTiles);
  std::cerr << "P=" << p << " t=" << numTiles << std::endl;
  uint32_t remainder = std::max(numTiles, uint32_t(size.size()));
  std::vector<uint32_t> res;
  for (const auto &s : size) {
    uint32_t tile =
        uint32_t(std::max(float(s) * float(numTiles) / float(total), 1.0f));
    res.push_back(tile);
    remainder -= tile;
  }
  std::cerr << "Remainder " << remainder << " nt " << numTiles << " t "
            << to_string(res) << " tot " << total << std::endl;
  for (size_t i = 0; i < remainder; ++i) {
    ++res[res.size() - 1 - i];
  }
  return res;
}

std::string to_string(const Interval &inter) {
  return "(" + std::to_string(inter.begin()) + "," +
         std::to_string(inter.end()) + ")";
}

std::string to_string(const std::vector<poplar::Interval> &input) {
  std::string res = "[";
  for (size_t i = 0; i < input.size(); ++i) {
    res += to_string(input[i]);
    if (i < input.size() - 1)
      res += ",";
  }
  return res += "]";
}

void printMapping(const std::vector<std::vector<Interval>> &mapping) {
  for (size_t i = 0; i < mapping.size(); ++i) {
    if (!mapping[i].empty())
      std::cerr << "T=" << i << ": " << to_string(mapping[i]) << " ";
  }
  std::cerr << std::endl;
}

void dumpTileMapping(const Graph &graph, const Tensor &t, std::ostream &os) {
  auto ttm = graph.getTileMapping(t);
  for (unsigned idxTile = 0; idxTile < ttm.size(); ++idxTile) {
    std::size_t chunkSize = 0;
    const auto &ttmTile = ttm[idxTile];
    for (unsigned j = 0; j < ttmTile.size(); ++j) {
      chunkSize += ttmTile[j].end() - ttmTile[j].begin();
    }
    if (chunkSize > 0) {
      os << idxTile << " : " << chunkSize << std::endl;
    }
  }
}

size_t getTile(poplar::Graph &graph, const poplar::Tensor &t) {
  const auto &mapping = graph.getTileMapping(t);
  size_t res, count = 0;
  for (size_t i = 0; i < mapping.size(); ++i) {
    if (!mapping[i].empty()) {
      ++count;
      res = i;
      assert(mapping[i].size() == 1);
    }
  }
  if (count != 1) {
    dumpTileMapping(graph, t, std::cerr);
  }
  assert(count == 1);
  return res;
}

poplar::VertexRef connectVertex(
    poplar::Graph &graph, poplar::ComputeSet &cs, const std::string &vertexName,
    const std::unordered_map<std::string, poplar::Tensor> &vars,
    const std::unordered_map<std::string, std::vector<poplar::Tensor>> &vectors,
    size_t tile) {
  poplar::VertexRef vtx = graph.addVertex(cs, vertexName);
  for (const auto &p : vars) {
    graph.connect(vtx[p.first], p.second);
  }
  for (const auto &p : vectors) {
    graph.connect(vtx[p.first], p.second);
  }
  graph.setPerfEstimate(vtx, 1);
  graph.setTileMapping(vtx, tile);
  return vtx;
}
