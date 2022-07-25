// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once
#include <iostream>

#include "utils.hpp"

#include <poplar/CSRFunctions.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <poputil/VertexTemplates.hpp>

#include <type_traits>
#include <unordered_map>

using namespace poplar;

Device getDevice();
Device getDevice(uint32_t tilesPerIPU);
uint32_t getNumTiles(uint32_t size, uint32_t totalTiles, uint32_t numWorkers,
                     uint32_t loadWorker);
std::vector<std::vector<poplar::Interval>>
createTileMapping(uint32_t size, uint32_t numTiles, uint32_t totalTiles,
                  uint32_t firstTile = 0);
std::vector<uint32_t> tileSize(const std::vector<uint32_t> &size,
                               uint32_t totalTiles, uint32_t numWorkers,
                               uint32_t loadWorker);
std::vector<uint16_t> to_half(const Target &target,
                              const std::vector<float> &vec);
std::vector<float> to_float(const Target &target,
                            const std::vector<uint16_t> &vec);
void printMapping(const std::vector<std::vector<poplar::Interval>> &mapping);
void dumpTileMapping(const Graph &graph, const Tensor &t,
                     std::ostream &os = std::cout);
size_t getTile(poplar::Graph &graph, const poplar::Tensor &t);
poplar::VertexRef connectVertex(
    poplar::Graph &graph, poplar::ComputeSet &cs, const std::string &vertexName,
    const std::unordered_map<std::string, poplar::Tensor> &vars,
    const std::unordered_map<std::string, std::vector<poplar::Tensor>> &vectors,
    size_t tile = 0);

template <typename T> class ReadHandle {
  std::vector<T> data_;
  std::string name_;

public:
  ReadHandle(const poplar::Tensor &t, const std::string &name,
             poplar::Graph &graph)
      : data_(t.numElements(), 0.0), name_{name} {
    graph.createHostRead(name_, t);
  }
  const std::vector<T> &read(poplar::Engine &engine) {
    engine.readTensor(name_, (char *)data_.data(),
                      (char *)data_.data() + sizeof(T) * data_.size());
    return data_;
  }
};

class ReadStream {
  std::vector<float> data_;
  poplar::DataStream stream_;
  std::string name_;
  bool connected_ = false;

public:
  ReadStream(const poplar::Tensor &t, const std::string &name,
             poplar::Graph &graph)
      : data_(t.numElements(), 0.0), name_{name} {
    stream_ = graph.addDeviceToHostFIFO(name_, poplar::FLOAT, t.numElements());
  }
  void connect(poplar::Engine &engine) {
    engine.connectStream(name_, (char *)data_.data());
    connected_ = true;
  }
  const std::vector<float> &data() const {
    assert(connected_);
    return data_;
  }
  poplar::DataStream &stream() { return stream_; }
};
