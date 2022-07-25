// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <vector>

template <typename T> std::string to_string(const T *data, size_t size) {
  std::string res = "[";
  for (size_t i = 0; i < size; ++i) {
    res += std::to_string(data[i]);
    if (i < size - 1)
      res += ",";
  }
  return res += "]";
}
template <typename T> std::string to_string(const std::vector<T> &input) {
  std::string res = "[";
  for (size_t i = 0; i < input.size(); ++i) {
    res += std::to_string(input[i]);
    if (i < input.size() - 1)
      res += ",";
  }
  return res += "]";
}

class Shape {
private:
  std::vector<size_t> dims_;
  size_t size_;
  std::vector<size_t> accum_;

  void print_indent(size_t dim, std::ostream &out) const {
    for (size_t i = 0; i < dim; ++i)
      out << " ";
  }

  template <typename T>
  void print_dim(const std::vector<T> &data, std::ostream &out, size_t dim,
                 size_t &offset) const {
    print_indent(dim, out);
    out << "[";
    if (dim == dims_.size() - 1) {
      for (size_t i = 0; i < dims_[dim]; ++i) {
        out << std::to_string(data[i + offset]);
        if (i < dims_[dim] - 1)
          out << ",";
      }
      offset += dims_[dim];
    } else {
      out << "\n";
      for (size_t i = 0; i < dims_[dim]; ++i)
        print_dim(data, out, dim + 1, offset);
      print_indent(dim, out);
    }
    out << "]\n";
  }

public:
  template <typename It> Shape(It start, It end) {
    for (auto it = start; it != end; ++it) {
      dims_.push_back(*it);
    }
    accum_.resize(dims_.size());
    size_ = 1;
    for (int i = dims_.size() - 1; i >= 0; --i) {
      accum_[i] = size_;
      size_ *= dims_[i];
    }
  }
  Shape(const std::initializer_list<size_t> &dims)
      : Shape(dims.begin(), dims.end()) {}

  Shape permute(const std::vector<size_t> &dims) const {
    assert(dims.size() == dims_.size());
    std::vector<size_t> indices(dims_.size());
    size_t i = 0;
    for (size_t d : dims) {
      indices[i] = dims_[d];
      ++i;
    }
    return Shape(indices.begin(), indices.end());
  }

  size_t rank() const { return dims_.size(); }

  size_t size() const { return size_; }
  std::vector<size_t> dims() const { return dims_; }
  bool same_dims(const std::vector<size_t> &d) const {
    if (d.size() != dims_.size())
      return false;
    for (size_t i = 0; i < dims_.size(); ++i) {
      if (dims_[i] != d[i])
        return false;
    }
    return true;
  }

  std::string to_string() const { return ::to_string(dims_); }

  size_t index(const std::vector<size_t> &indices) const {
    size_t res = 0;
    size_t i = 0;
    for (size_t d : indices) {
      assert(d < dims_[i]);
      res += d * accum_[i];
      ++i;
    }
    return res;
  }

  void indices(size_t index, std::vector<size_t> &indices) const {
    assert(indices.size() == dims_.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      indices[i] = (index / accum_[i]) % dims_[i];
    }
  }

  template <typename T>
  void print_tensor(const std::vector<T> &data, std::ostream &out) const {
    assert(data.size() == size_);
    size_t offset = 0;
    print_dim(data, out, 0, offset);
  }
};

template <typename T>
std::vector<T> transpose(const std::vector<T> &data, const Shape &from,
                         const std::vector<size_t> &dims) {
  assert(from.rank() == dims.size());
  assert(data.size() == from.size());
  std::vector<size_t> indicesFrom(from.rank());
  std::vector<size_t> indicesTo(from.rank());
  std::vector<T> res(data.size());
  Shape to = from.permute(dims);
  for (size_t i = 0; i < data.size(); ++i) {
    from.indices(i, indicesFrom);
    size_t j = 0;
    for (size_t d : dims) {
      indicesTo[j] = indicesFrom[d];
      ++j;
    }
    res[to.index(indicesTo)] = data[i];
  }
  return res;
}

template <typename T> struct NDArray {
private:
  Shape shape_;
  std::vector<T> data_;

public:
  NDArray(const Shape &s) : shape_{s} { data_ = std::vector<T>(s.size(), T{}); }
  NDArray(const Shape &s, const std::vector<T> &data)
      : shape_{s}, data_{data} {}
  NDArray(const Shape &s, const T &value) : shape_(s), data_(s.size(), value) {}

  T &operator[](const std::vector<size_t> &indices) {
    return data_[shape_.index(indices)];
  }
  const T &operator[](const std::vector<size_t> &indices) const {
    return data_[shape_.index(indices)];
  }
  const T *operator()(const std::vector<size_t> &indices) const {
    return &data_[shape_.index(indices)];
  }
  T *operator()(const std::vector<size_t> &indices) {
    return &data_[shape_.index(indices)];
  }

  const std::vector<T> &data() const { return data_; }
  std::vector<T> copy_data() const { return data_; }
  void print(std::ostream &out) const { shape_.print_tensor(data_, out); }
  const Shape &shape() const { return shape_; }
  std::vector<size_t> dims() const { return shape_.dims(); }
};

template <typename T> void logSoftmax(T *data, uint32_t size) {
  T max = -std::numeric_limits<T>::max();
  for (size_t i = 0; i < size; ++i) {
    if (data[i] > max)
      max = data[i];
  }
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += std::exp(double(data[i]) - max);
  }
  sum = std::log(sum);
  for (size_t i = 0; i < size; ++i) {
    data[i] = T(double(data[i]) - max - sum);
  }
}

template <typename T>
void logSoftmax(std::vector<T> &data, const Shape &shape) {
  assert(data.size() == shape.size());
  const auto dims = shape.dims();
  size_t last_dim = dims.back();
  for (size_t i = 0; i < data.size(); i += last_dim) {
    logSoftmax(&data[i], last_dim);
  }
}

class DataGenerator {
  std::mt19937 mt_;
  std::uniform_real_distribution<float> dist_;    // range
  std::uniform_int_distribution<int32_t> choice_; // range
public:
  DataGenerator(float scale, uint32_t alphabet, uint32_t seed = 42)
      : mt_(seed), dist_{-scale, scale}, choice_{0, int32_t(alphabet)} {}

  float generate() { return dist_(mt_); }
  std::vector<float> generate(uint32_t size) {
    std::vector<float> data(size, 0.0);
    for (uint32_t i = 0; i < size; ++i) {
      data[i] = dist_(mt_);
    }
    return data;
  }
  std::vector<uint32_t> labels(uint32_t size) {
    std::vector<uint32_t> data(size, 0);
    for (uint32_t i = 0; i < size; ++i) {
      data[i] = choice_(mt_);
    }
    return data;
  }
  std::mt19937 &mt() { return mt_; }
};
