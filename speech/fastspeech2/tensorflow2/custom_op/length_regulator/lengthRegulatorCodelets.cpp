// Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;


template <class T>
class duration2IndexVertex : public Vertex {
  public:
    Input<Vector<T>> duration;
    Output<Vector<int>> index;
    int hmel_size;
    int i;

    bool compute() {
      int prefix_sum = 0;
      index[0] = hmel_size;   // flag for out of range
      for (auto j = 0; j < hmel_size; ++j) {
        prefix_sum += int(duration[j]);
        if (prefix_sum > i) {
          index[0] = j;
          break;
        }
      }

      return true;
    }
};

template class duration2IndexVertex<half>;
template class duration2IndexVertex<float>;


template <class T>
class grad2DurationVertex : public Vertex {
  public:
    Input<Vector<T>> gradient;
    Input<Vector<T>> duration;
    Output<Vector<T>> grad2Dur;
    int index;

    bool compute() {
      if (duration[index] == 0) {
        grad2Dur[0] = 0;
      } else {
        int prefix_sum = 0;
        for (auto i = 0; i < index; i++) {
          prefix_sum += duration[i];
        }
        grad2Dur[0] = duration[index] * gradient[prefix_sum];
      }

      return true;
    }
};

template class grad2DurationVertex<half>;
template class grad2DurationVertex<float>;


template <class T>
class lengthRegulatorVertex : public Vertex {

public:
  Input<Vector<T>> hmel;
  Input<Vector<int>> index; 
  Output<Vector<T>> mel_out;
  int mel_count;
  int size;

  bool compute() {

    for (auto i = 0u; i < size; ++i) {
        mel_out[i] = hmel[index[i]];
    }

    return true;
  }
};

template class lengthRegulatorVertex<half>;
template class lengthRegulatorVertex<float>;
