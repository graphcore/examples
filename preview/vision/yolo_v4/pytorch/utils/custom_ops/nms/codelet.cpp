// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef __IPU__
#include <ipu_vector_math>
#endif

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <print.h>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template <typename T> class UpdateBestVertex : public poplar::Vertex {
public:
  UpdateBestVertex();
  poplar::Input<uint32_t> best;
  poplar::InOut<poplar::Vector<T>> scores;

  const uint32_t offset;
  bool compute() {
    if (best >= offset && best < offset + scores.size()) {
      scores[best - offset] = -std::numeric_limits<T>::max();
    }
    return true;
  }
};

template class UpdateBestVertex<float>;
template class UpdateBestVertex<half>;

template <typename T> class UpdateBestMultiVertex : public poplar::Vertex {
public:
  UpdateBestMultiVertex();
  poplar::Input<uint32_t> best;
  poplar::InOut<poplar::Vector<T>> scores;

  const uint32_t C;
  const uint32_t offset;
  bool compute() {
    const uint32_t base_offset = C * offset;
    if (best >= base_offset && best < base_offset + scores.size()) {
      scores[best - base_offset] = -std::numeric_limits<T>::max();
    }
    return true;
  }
};

template class UpdateBestMultiVertex<float>;
template class UpdateBestMultiVertex<half>;

template <typename T, typename C>
class UpdateAnswerVertex : public poplar::Vertex {
public:
  UpdateAnswerVertex();
  poplar::Input<uint32_t> best_indices;                 // [bs]
  poplar::Input<T> best_scores;                         // [bs]
  poplar::Input<poplar::Vector<T, ONE_PTR>> best_boxes; // [bs, 4]
  poplar::Input<C> best_classes;                        // [bs]

  poplar::InOut<poplar::Vector<int32_t, ONE_PTR>> lengths; // [K, bs]

  poplar::InOut<poplar::Vector<int32_t>> top_indices;    // [K, bs]
  poplar::InOut<poplar::Vector<T, ONE_PTR>> top_scores;  // [K, bs]
  poplar::InOut<poplar::Vector<T, ONE_PTR>> top_boxes;   // [K, bs, 4]
  poplar::InOut<poplar::Vector<C, ONE_PTR>> top_classes; // [K, bs]
  poplar::Input<uint32_t> i;

  const float score_threshold;
  const int32_t K;
  const uint32_t offset;
  bool compute() {
    const uint32_t size = top_indices.size();
    if (i >= offset && i < offset + size) {
      const uint32_t j = i - offset;
      if (float(*best_scores) > score_threshold) { // let's copy the values
        top_indices[j] = best_indices;
        top_scores[j] = best_scores;
        top_classes[j] = best_classes;
        std::memcpy(&top_boxes[j], &best_boxes[0], 4 * sizeof(T));
      } else { // we discard the value, so we need to update lengths
        if (lengths[j] == K) {
          lengths[j] = int32_t(i);
        }
      }
    }
    return true;
  }
};

template class UpdateAnswerVertex<float, int>;
template class UpdateAnswerVertex<float, unsigned int>;
template class UpdateAnswerVertex<half, int>;
template class UpdateAnswerVertex<half, unsigned int>;

template <typename T> class UpdateAnswerMultiVertex : public poplar::Vertex {
public:
  UpdateAnswerMultiVertex();
  poplar::Input<uint32_t> best_indices;                 // [bs]
  poplar::Input<T> best_scores;                         // [bs]
  poplar::Input<poplar::Vector<T, ONE_PTR>> best_boxes; // [bs, 4]

  poplar::InOut<poplar::Vector<int32_t, ONE_PTR>> lengths; // [K, bs]

  poplar::InOut<poplar::Vector<int32_t>> top_indices;          // [K, bs]
  poplar::InOut<poplar::Vector<T, ONE_PTR>> top_scores;        // [K, bs]
  poplar::InOut<poplar::Vector<T, ONE_PTR>> top_boxes;         // [K, bs, 4]
  poplar::InOut<poplar::Vector<int32_t, ONE_PTR>> top_classes; // [K, bs]
  poplar::Input<uint32_t> i;

  const float score_threshold;
  const int32_t K;
  const int32_t C;
  const uint32_t offset;
  bool compute() {
    const uint32_t size = top_indices.size();
    if (i >= offset && i < offset + size) {
      const uint32_t j = i - offset;
      if (float(*best_scores) > score_threshold) { // let's copy the values
        top_indices[j] = best_indices / C;
        top_scores[j] = best_scores;
        top_classes[j] = best_indices % C;
        std::memcpy(&top_boxes[j], &best_boxes[0], 4 * sizeof(T));
      } else { // we discard the value, so we need to update lengths
        if (lengths[j] == K) {
          lengths[j] = int32_t(i);
        }
      }
    }
    return true;
  }
};

template class UpdateAnswerMultiVertex<float>;
template class UpdateAnswerMultiVertex<half>;

template <typename T> class SliceVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<T, ONE_PTR>> input;
  poplar::Input<uint32_t> index;
  poplar::Output<T> output;
  bool compute() {
    *output = input[index];
    return true;
  }
};

template class SliceVertex<float>;
template class SliceVertex<half>;
template class SliceVertex<unsigned int>;
template class SliceVertex<int>;

template <typename T> class SliceMultiVertex : public poplar::Vertex {
public:
  SliceMultiVertex();
  poplar::Input<poplar::Vector<T, ONE_PTR>> input;
  poplar::Input<uint32_t> index;
  poplar::Output<T> output;
  const uint32_t C;
  bool compute() {
    *output = input[index / C];
    return true;
  }
};

template class SliceMultiVertex<float>;
template class SliceMultiVertex<half>;
template class SliceMultiVertex<unsigned int>;
template class SliceMultiVertex<int>;

template <typename T, typename C> class GatherVertex : public poplar::Vertex {
public:
  GatherVertex();
  poplar::Input<uint32_t> index;
  poplar::Input<poplar::Vector<T, ONE_PTR>> boxes;
  poplar::Input<poplar::Vector<C>> classes;
  poplar::InOut<poplar::Vector<T, ONE_PTR>> output;
  poplar::InOut<C> outputClass;

  const uint32_t offset;
  bool compute() {
    if (index >= offset && index < offset + classes.size()) {
      const uint32_t i = index - offset;
      std::memcpy(&output[0], &boxes[i * 4], 4 * sizeof(T));
      *outputClass = classes[i];
    }
    return true;
  }
};

template class GatherVertex<float, int>;
template class GatherVertex<half, int>;
template class GatherVertex<float, unsigned int>;
template class GatherVertex<half, unsigned int>;

template <typename T> class GatherMultiVertex : public poplar::Vertex {
public:
  GatherMultiVertex();
  poplar::Input<uint32_t> index;

  poplar::Input<poplar::Vector<T>> boxes;
  poplar::InOut<poplar::Vector<T, ONE_PTR>> output;

  const uint32_t C;
  const uint32_t offset;
  bool compute() {
    const size_t N = boxes.size() / 4;
    const uint32_t base_offset = offset * C;
    if (index >= base_offset && index < base_offset + N * C) {
      const uint32_t i = index / C - offset;
      std::memcpy(&output[0], &boxes[i * 4], 4 * sizeof(T));
    }
    return true;
  }
};

template class GatherMultiVertex<float>;
template class GatherMultiVertex<half>;

#ifdef __IPU__
template <typename T>
inline float computeIOU(const T *boxes, const T *bestBox, const T &bestArea) {
  T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2], by2 = bestBox[3];
  T x1 = boxes[0], y1 = boxes[1], x2 = boxes[2], y2 = boxes[3];
  T area = (x2 - x1) * (y2 - y1);
  T xx1 = std::max(x1, bx1);
  T yy1 = std::max(y1, by1);
  T xx2 = std::min(x2, bx2);
  T yy2 = std::min(y2, by2);
  float w = std::max(0.0f, float(xx2 - xx1));
  float h = std::max(0.0f, float(yy2 - yy1));
  float inter = w * h;

  float iou = float(inter) / (float(area) + float(bestArea) - inter);
  return iou;
}
inline float computeIOU(const half *boxes, const half *bestBox,
                        const half &bestArea) {
  half2 *boxes_ = (half2 *)boxes, *bestBox_ = (half2 *)bestBox;
  half2 x1y1 = boxes_[0], x2y2 = boxes_[1];
  half2 bx1by1 = bestBox_[0], bx2by2 = bestBox_[1];
  half2 diff = x2y2 - x1y1;
  half area = diff[0] * diff[1];
  half2 xx1yy1 = half2_fmax(x1y1, bx1by1);
  half2 xx2yy2 = half2_fmin(x2y2, bx2by2);
  half2 zero = {0.0, 0.0};
  half2 diff2 = xx2yy2 - xx1yy1;
  half2 wh = half2_fmax(zero, diff2);
  float inter = float(wh[0] * wh[1]);
  float iou = float(inter) / (float(area) + float(bestArea) - inter);
  return iou;
}

#else
template <typename T>
inline float computeIOU(const T *boxes, const T *bestBox, const T &bestArea) {
  T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2], by2 = bestBox[3];
  T x1 = boxes[0], y1 = boxes[1], x2 = boxes[2], y2 = boxes[3];
  T area = (x2 - x1) * (y2 - y1);
  T xx1 = std::max(x1, bx1);
  T yy1 = std::max(y1, by1);
  T xx2 = std::min(x2, bx2);
  T yy2 = std::min(y2, by2);
  float w = std::max(0.0f, float(xx2 - xx1));
  float h = std::max(0.0f, float(yy2 - yy1));
  float inter = w * h;

  float iou = float(inter) / (float(area) + float(bestArea) - inter);
  return iou;
}
#endif

template <typename T, typename C> class NmsVertex : public poplar::MultiVertex {
public:
  NmsVertex();
  poplar::InOut<poplar::Vector<T>> scores; // [bs, N]
  poplar::Input<poplar::Vector<C, ONE_PTR>> classes;
  poplar::Input<poplar::Vector<T, ONE_PTR>> boxes; // [bs, N, 4]
  poplar::Input<C> bestClass;
  poplar::Input<poplar::Vector<T, ONE_PTR>> bestBox; // [4]

  const float sigma;
  const float threshold;
  const float score_threshold;

  bool compute(unsigned int workerId) {
    T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2], by2 = bestBox[3];
    T barea = (bx2 - bx1) * (by2 - by1);
    for (size_t i = workerId; i < scores.size();
         i += MultiVertex::numWorkers()) {
      if (scores[i] > T(score_threshold) && classes[i] == bestClass) {
        float iou = computeIOU(&boxes[i * 4], &bestBox[0], barea);
        if (sigma == 0.0f) {
          if (iou > threshold) {
            scores[i] = -std::numeric_limits<T>::max();
          }
        } else {
          float weight = std::exp(-(iou * iou) / sigma); // paper version
          scores[i] *= weight;
        }
      }
    }
    return true;
  }
};

template class NmsVertex<float, int>;
template class NmsVertex<float, unsigned int>;
template class NmsVertex<half, int>;
template class NmsVertex<half, unsigned int>;

template <typename T> class NmsMultiVertex : public poplar::MultiVertex {
public:
  NmsMultiVertex();
  poplar::InOut<poplar::Vector<T>> scores;         // [bs, N, C]
  poplar::Input<poplar::Vector<T, ONE_PTR>> boxes; // [bs, N, 4]
  poplar::Input<uint32_t> bestIndices;
  poplar::Input<poplar::Vector<T>> bestBox; // [4]

  const float sigma;
  const float threshold;
  const float score_threshold;
  const uint32_t C;

  bool compute(unsigned int workerId) {
    T bx1 = bestBox[0], by1 = bestBox[1], bx2 = bestBox[2], by2 = bestBox[3];
    T barea = (bx2 - bx1) * (by2 - by1);
    for (size_t i = workerId; i < scores.size() / C;
         i += MultiVertex::numWorkers()) {
      const int32_t class_i = bestIndices % C;
      const int32_t score_i = i * C + class_i;
      if (scores[score_i] > T(score_threshold)) {
        float iou = computeIOU(&boxes[i * 4], &bestBox[0], barea);
        if (sigma == 0.0f) {
          if (iou > threshold) {
            scores[score_i] = -std::numeric_limits<T>::max();
          }
        } else {
          float weight = std::exp(-(iou * iou) / sigma); // paper version
          scores[score_i] *= weight;
        }
      }
    }
    return true;
  }
};

template class NmsMultiVertex<float>;
template class NmsMultiVertex<half>;

template <typename T> class ConditionVertex : public poplar::Vertex {
public:
  ConditionVertex();
  poplar::Input<poplar::Vector<T>> bestScores; // [bs]
  poplar::Input<uint32_t> i;
  poplar::Input<uint32_t> numIter;
  poplar::Output<bool> condition;
  const float score_threshold;

  bool compute() {
    if (i == 0) {
      *condition = true;
      return true;
    }
    bool res = i < numIter;
    uint32_t s_i = 0;
    while (res && s_i < bestScores.size()) {
      res = res && (float(bestScores[s_i]) > score_threshold);
      ++s_i;
    }
    *condition = res;
    return true;
  }
};

template class ConditionVertex<float>;
template class ConditionVertex<half>;
