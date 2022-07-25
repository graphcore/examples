// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

class CompactInPlaceVertex : public poplar::Vertex {
public:
  // Fields
  poplar::InOut<poplar::Vector<unsigned int>> in;
  poplar::Output<unsigned int> outLen;

  // Compute function
  bool compute() {
    unsigned int len = 0;
    unsigned int previous = std::numeric_limits<unsigned int>::max();
    for (const auto &v : in) {
      if (v != previous) {
        previous = v;
        if (v != 0) {
          in[len] = v;
          ++len;
        }
      }
    }
    for (auto i = len; i < in.size(); ++i) {
      in[i] = 0;
    }
    *outLen = len;
    return true;
  }
};

template <typename T, typename I> class InitVertex : public poplar::Vertex {
public:
  // Fields
  poplar::Input<poplar::Vector<I, ONE_PTR>> in;
  poplar::Input<I> len;
  poplar::Output<poplar::Vector<I, ONE_PTR>> seq;
  poplar::Output<poplar::Vector<bool, ONE_PTR>> diff;
  poplar::Output<poplar::Vector<bool, ONE_PTR>> revDiff;

  // Compute function
  bool compute() {
    unsigned int j = len - 1;
    seq[0] = 0;
    diff[0] = false;
    revDiff[2 * len] = false;
    for (unsigned int i = 0; i < len; ++i) {
      seq[2 * i + 1] = in[i];
      seq[2 * i + 2] = 0;
      if (i > 0 && in[i] != in[i - 1]) {
        diff[2 * i + 1] = true;
      } else {
        diff[2 * i + 1] = false;
      }
      diff[2 * i + 2] = false;
      if (j < len - 1 && in[j] != in[j + 1]) {
        revDiff[2 * j + 1] = true;
      } else {
        revDiff[2 * j + 1] = false;
      }
      revDiff[2 * j] = false;
      --j;
    }
    return true;
  }
};

template class InitVertex<float, int>;
template class InitVertex<float, unsigned int>;
template class InitVertex<half, int>;
template class InitVertex<half, unsigned int>;

template <typename T> class InitInfinityVertex : public poplar::Vertex {
public:
  // Fields
  poplar::Output<poplar::Vector<T>> out;

  // Compute function
  bool compute() {
    for (auto i = 0; i < out.size(); ++i)
      out[i] = -std::numeric_limits<T>::max();
    return true;
  }
};

template class InitInfinityVertex<float>;
template class InitInfinityVertex<half>;

bool isNegInf(float v) {
  return v == -std::numeric_limits<float>::max() ||
         v == -std::numeric_limits<float>::infinity();
}

bool isNegInf(half v) {
  return v == -std::numeric_limits<half>::max() ||
         v == -std::numeric_limits<half>::infinity() ||
         v == half(-std::numeric_limits<float>::infinity());
}

template <typename T> inline void assignInf(T &left, const T &right) {
  T inf = -std::numeric_limits<T>::max();
  if (isNegInf(right) || isNegInf(left)) {
    left = inf;
  } else {
    left += right;
  }
}

template <typename T> class AddInfinityVertex : public poplar::Vertex {
public:
  // Fields
  poplar::InOut<poplar::Vector<T>> alpha;
  poplar::Input<poplar::Vector<T, ONE_PTR>> beta;

  // Compute function
  bool compute() {
    for (auto i = 0; i < alpha.size(); ++i)
      assignInf(alpha[i], beta[i]);
    return true;
  }
};

template class AddInfinityVertex<float>;
template class AddInfinityVertex<half>;

class UpdateTVertex : public poplar::Vertex {
public:
  poplar::InOut<poplar::Vector<unsigned int, ONE_PTR>> t;
  bool compute() {
    ++t[0];
    --t[1];
    return true;
  }
};

template <typename T, typename T2, typename I>
class GradVertex : public poplar::Vertex {
public:
  // Fields
  poplar::InOut<poplar::Vector<float>> grad;     // [numLabels]
  poplar::Input<poplar::Vector<I, ONE_PTR>> seq; // [seqLen]
  poplar::Input<I> targetLen;
  poplar::Input<I> inputLen;
  poplar::Input<unsigned int> t;
  poplar::Input<poplar::Vector<T, ONE_PTR>> alphaBeta; // [seqLen]
  poplar::Input<T> nll;
  poplar::Input<T> gr;
  poplar::Input<poplar::Vector<T2, ONE_PTR>> lpp;

  // Compute function
  bool compute() {
    if (t < inputLen) {
      unsigned seqLen = 2 * targetLen + 1;
      for (auto i = 0; i < seqLen; ++i) {
        bool v_inf = isNegInf(grad[seq[i]]);
        float v = grad[seq[i]];
        if (v_inf) {
          grad[seq[i]] = float(alphaBeta[i]);
        } else {
          bool alphaBeta_inf = isNegInf(alphaBeta[i]);
          float alphaBeta_i = float(alphaBeta[i]);
          float m = std::max(v, alphaBeta_i);
          float inter = 0.0f;
          if (!alphaBeta_inf)
            inter = std::exp(alphaBeta_i - m);
          inter = std::exp(v - m) + inter;
          if (inter != 0.0f)
            grad[seq[i]] = float(std::log(inter) + m);
          else
            grad[seq[i]] = -std::numeric_limits<float>::max();
        }
      }
      for (auto i = 0; i < grad.size(); ++i) {
        // Hack to work around fp16 -inf problems.
        bool lpInf = isNegInf(lpp[i]);
        float lp = float(lpp[i]);
        float res = grad[i];
        float inter = 0.0f;
        if (!isNegInf(grad[i])) {
          inter = res + float(*nll);
          if (!lpInf) {
            inter = float(*gr) * (std::exp(lp) - std::exp(inter - lp));
          }
        } else {
          if (!lpInf) {
            inter = float(*gr) * std::exp(lp);
          }
        }
        grad[i] = float(inter);
        // grad[i] = *gr * (std::exp(lp) - std::exp(res + (*nll) - lp));
      }
    }
    for (auto i = 0; i < grad.size(); ++i) {
      float v = grad[i];
      if (isNegInf(v)) {
        grad[i] = 0.0f;
      }
    }
    return true;
  }
};

template class GradVertex<float, half, int>;
template class GradVertex<float, half, unsigned int>;
template class GradVertex<float, float, int>;
template class GradVertex<float, float, unsigned int>;
template class GradVertex<half, half, int>;
template class GradVertex<half, half, unsigned int>;

template <typename T> class GatherLossVertex : public poplar::Vertex {
public:
  poplar::Input<unsigned int> indice;
  poplar::Input<poplar::Vector<T, ONE_PTR>> loss;
  poplar::Output<T> out;

  // Compute function
  bool compute() {
    *out = loss[indice];
    return true;
  }
};

template class GatherLossVertex<float>;
template class GatherLossVertex<half>;

template <typename T> class NLLLossVertex : public poplar::Vertex {
public:
  // Fields
  poplar::Input<poplar::Vector<T, ONE_PTR>> loss;
  poplar::Output<T> out;

  // Compute function
  bool compute() {
    T l1 = loss[0];
    T l2 = loss[1];
    float l1_ = float(l1);
    float l2_ = float(l2);
    float m = float(std::max(l1, l2));
    if (isNegInf(m)) {
      m = 0.0f;
    }
    float inter = 0.0f;
    if (!isNegInf(l1))
      inter = std::exp(l1_ - m);
    if (!isNegInf(l2))
      inter += std::exp(l2_ - m);
    if (inter != 0.0f)
      *out = T(-(std::log(inter) + m));
    else { // Should be infinity, but not in fp16.
      *out = T(0.0);
    }
    return true;
  }
};

template class NLLLossVertex<float>;
template class NLLLossVertex<half>;

template <typename T>
static T max3(const T &t1, const T &t1_s1, const T &t1_s2, const bool diff) {
  T m = std::max(t1, t1_s1);
  if (diff) {
    m = std::max(m, t1_s2);
  }
  if (isNegInf(m)) {
    return T(0.0);
  }
  return m;
}

template <typename T, typename T2, typename I>
class CTCForwardVertex : public poplar::Vertex {
public:
  // Fields
  poplar::Input<poplar::Vector<T>> alpha_t1;
  poplar::Input<poplar::Vector<T, ONE_PTR>> alpha_t1_s1;
  poplar::Input<poplar::Vector<T, ONE_PTR>> alpha_t1_s2;
  poplar::Input<poplar::Vector<I, ONE_PTR>> seq;
  poplar::Input<poplar::Vector<unsigned short, ONE_PTR>> batchIndices;
  poplar::Input<poplar::Vector<unsigned short, ONE_PTR>> sliceIndices;
  poplar::Input<poplar::Vector<I, ONE_PTR>> inputLengths;
  poplar::Input<poplar::Vector<I, ONE_PTR>> targetLengths;
  poplar::InOut<unsigned int> t;
  poplar::Vector<poplar::Input<poplar::Vector<T2, ONE_PTR>>, ONE_PTR> lpp_t;
  poplar::Input<poplar::Vector<bool, ONE_PTR>> diff;
  poplar::Output<poplar::Vector<T, ONE_PTR>> out;
  poplar::Vector<poplar::InOut<poplar::Vector<T, ONE_PTR>>> alpha;
  poplar::Vector<poplar::InOut<poplar::Vector<T, ONE_PTR>>, ONE_PTR> loss;

  // Compute function
  bool compute() {
    unsigned int i = 0;
    unsigned int offset = t < alpha.size() ? 0 : alpha.size();
    for (const auto &v : alpha_t1) {
      T res = -std::numeric_limits<T>::max();
      if (t == 0) {
        if (sliceIndices[i] < 2) {
          res = T(lpp_t[seq[i]][batchIndices[i]]);
        }
      } else {
        unsigned seqLen = 2 * targetLengths[batchIndices[i]] + 1;
        unsigned start = 0;
        if (seqLen > 2 * (inputLengths[batchIndices[i]] - t)) {
          start = seqLen - 2 * (inputLengths[batchIndices[i]] - t);
        }
        unsigned end = std::min(2 * t + 2, seqLen);
        unsigned inputLen_i = inputLengths[batchIndices[i]];
        if (sliceIndices[i] < seqLen && sliceIndices[i] >= start &&
            sliceIndices[i] <= end && t < inputLen_i) {
          float v_ = float(v);
          float t1_s1_ = float(alpha_t1_s1[i]);
          float t1_s2_ = float(alpha_t1_s2[i]);
          float max = max3(v_, t1_s1_, t1_s2_, diff[i]);
          T2 lpp_t_i = lpp_t[seq[i]][batchIndices[i]];
          float inter = 0.0;
          if (diff[i])
            inter = std::exp(t1_s2_ - max);
          float tmp = std::exp(v_ - max) + std::exp(t1_s1_ - max) + inter;
          if (tmp != 0)
            res = T(std::log(tmp) + max + float(lpp_t_i));
          // extract the inputs for the loss
          if (t == inputLen_i - 1) {
            if (sliceIndices[i] == seqLen - 1) {
              loss[0][batchIndices[i]] = res;
            } else {
              if (sliceIndices[i] == seqLen - 2) {
                loss[1][batchIndices[i]] = res;
              }
            }
          }
        }
      }
      out[i] = res;
      assignInf(alpha[t - offset][i], res);
      ++i;
    }
    ++t;
    return true;
  }
};

template class CTCForwardVertex<float, float, int>;
template class CTCForwardVertex<float, float, unsigned int>;
template class CTCForwardVertex<float, half, int>;
template class CTCForwardVertex<float, half, unsigned int>;
template class CTCForwardVertex<half, half, int>;
template class CTCForwardVertex<half, half, unsigned int>;

template <typename T, typename T2, typename I>
class CTCBackwardVertex : public poplar::Vertex {
public:
  // Fields
  poplar::Input<poplar::Vector<T>> beta_t1;
  poplar::Input<poplar::Vector<T, ONE_PTR>> beta_t1_s1;
  poplar::Input<poplar::Vector<T, ONE_PTR>> beta_t1_s2;
  poplar::Input<poplar::Vector<I, ONE_PTR>> seq;
  poplar::Input<poplar::Vector<unsigned short, ONE_PTR>> batchIndices;
  poplar::Input<poplar::Vector<unsigned short, ONE_PTR>> sliceIndices;
  poplar::Input<poplar::Vector<I, ONE_PTR>> inputLengths;
  poplar::Input<poplar::Vector<I, ONE_PTR>> targetLengths;
  poplar::InOut<unsigned int> t;
  poplar::Vector<poplar::Input<poplar::Vector<T2, ONE_PTR>>, ONE_PTR> lpp_t;
  poplar::Input<poplar::Vector<bool, ONE_PTR>> diff;
  poplar::Vector<poplar::InOut<poplar::Vector<T, ONE_PTR>>> alpha;
  poplar::InOut<poplar::Vector<T, ONE_PTR>> out;

  // Compute function
  bool compute() {
    const float inf = -std::numeric_limits<float>::max();
    unsigned int i = 0;
    unsigned int offset = t < alpha.size() ? 0 : alpha.size();
    for (const auto &v : beta_t1) {
      T res = -std::numeric_limits<T>::max();
      const unsigned inputLen_i = inputLengths[batchIndices[i]];
      const unsigned seqLen = 2 * targetLengths[batchIndices[i]] + 1;
      unsigned start = 0;
      if (seqLen > 2 * (inputLen_i - t)) {
        start = seqLen - 2 * (inputLen_i - t);
      }
      const unsigned end = std::min(2 * t + 2, seqLen);
      const unsigned short s_i = sliceIndices[i];
      if (t == inputLen_i - 1 && (s_i == seqLen - 1 || s_i == seqLen - 2)) {
        res = T(lpp_t[seq[i]][batchIndices[i]]);
      } else {
        if (s_i < seqLen && s_i >= start && s_i <= end && t < inputLen_i - 1) {
          const float max =
              float(max3(v, beta_t1_s1[i], beta_t1_s2[i], diff[i]));
          float inter = 0.0f;
          if (diff[i])
            inter = std::exp(float(beta_t1_s2[i]) - max);
          inter = std::exp(float(v) - max) +
                  std::exp(float(beta_t1_s1[i]) - max) + inter;
          if (inter == 0.0f)
            inter = inf;
          else {
            inter = std::log(inter);
            res = inter + max + float(lpp_t[seq[i]][batchIndices[i]]);
          }
        }
      }
      out[i] = T(res);
      assignInf(alpha[t - offset][i], T(res));
      ++i;
    }
    --t;
    return true;
  }
};

template class CTCBackwardVertex<float, half, int>;
template class CTCBackwardVertex<float, half, unsigned int>;
template class CTCBackwardVertex<float, float, int>;
template class CTCBackwardVertex<float, float, unsigned int>;
template class CTCBackwardVertex<half, half, int>;
template class CTCBackwardVertex<half, half, unsigned int>;
