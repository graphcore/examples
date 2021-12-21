// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef __IPU__
#include <ipu_vector_math>
#endif

//#include <print.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

inline float log_sum_exp(float a, float b) {
  if (std::isinf(a) && a < 0)
    return b;
  if (std::isinf(b) && b < 0)
    return a;
  if (a > b)
    return std::log1p(std::exp(b - a)) + a;
  else
    return std::log1p(std::exp(a - b)) + b;
}

template <typename T> class AlphaZeroVertex : public poplar::Vertex {
public:
  poplar::InOut<T> out;
  bool compute() {
    *out = T(0.0);
    return true;
  }
};

template class AlphaZeroVertex<float>;
template class AlphaZeroVertex<half>;

template <typename T> class AlphaUVertex : public poplar::Vertex {
public:
  AlphaUVertex();
  poplar::Input<float> alpha;
  poplar::Input<T> prob;
  poplar::Input<int> label_len;
  poplar::InOut<float> out;
  const unsigned int u;
  bool compute() {
    if (u <= label_len)
      *out = alpha + float(*prob);
    else
      *out = -std::numeric_limits<float>::infinity();
    return true;
  }
};

template class AlphaUVertex<float>;
template class AlphaUVertex<half>;

template <typename T> class AlphaT0Vertex : public poplar::Vertex {
public:
  AlphaT0Vertex();
  poplar::Input<float> alpha;
  poplar::Input<T> prob;
  poplar::Input<int> input_len;
  poplar::Output<float> out;
  const unsigned int t;
  bool compute() {
    if (t < input_len)
      *out = alpha + float(*prob);
    else
      *out = -std::numeric_limits<float>::infinity();
    return true;
  }
};

template class AlphaT0Vertex<float>;
template class AlphaT0Vertex<half>;

class AlphaVertex : public poplar::Vertex {
public:
  poplar::Input<int> label;
  poplar::Input<float> alpha_1;
  poplar::Input<float> alpha_2;
  poplar::Input<float> prob_1;
  poplar::Input<poplar::Vector<float>> probs_t;
  poplar::InOut<float> out;
  bool compute() {
    float no_emit = alpha_1 + prob_1;
    float emit = alpha_2 + probs_t[label];
    *out = log_sum_exp(emit, no_emit);

    return true;
  }
};

template <typename T> class AlphaVertexCompact : public poplar::Vertex {
public:
  poplar::Input<float> alpha_1;
  poplar::Input<float> alpha_2;
  poplar::Input<T> prob_1;
  poplar::Input<T> prob_2;
  poplar::InOut<float> out;
  bool compute() {
    float no_emit = alpha_1 + float(*prob_1);
    float emit = alpha_2 + float(*prob_2);
    *out = log_sum_exp(emit, no_emit);

    return true;
  }
};

template class AlphaVertexCompact<float>;
template class AlphaVertexCompact<half>;

template <typename T> class BetaTUVertex : public poplar::Vertex {
public:
  BetaTUVertex();
  poplar::Input<T> prob_0;
  poplar::Input<int> input_len;
  poplar::Input<int> label_len;
  poplar::Output<float> out;
  const unsigned int t;
  const unsigned int u;
  bool compute() {
    if ((t == input_len - 1) && (u == label_len)) {
      *out = float(*prob_0);
    }

    return true;
  }
};

template class BetaTUVertex<float>;
template class BetaTUVertex<half>;

template <typename T> class BetaTVertexCompact : public poplar::Vertex {
public:
  BetaTVertexCompact();
  poplar::Input<poplar::Vector<T>> probs_T_U;
  poplar::Input<float> beta_T_U1;
  poplar::Input<int> input_len;
  poplar::Input<int> label_len;
  // U = label_len + 1
  poplar::InOut<float> out;
  const unsigned int t;
  const unsigned int u;
  bool compute() {
    if (t == input_len - 1) {
      if (u == label_len) {
        *out = float(probs_T_U[0]);
      } else { // u < U - 1
        *out = beta_T_U1 + float(probs_T_U[1]);
      }
    }

    return true;
  }
};

template class BetaTVertexCompact<float>;
template class BetaTVertexCompact<half>;

template <typename T> class BetaUVertex : public poplar::Vertex {
public:
  BetaUVertex();
  poplar::Input<float> beta_T1_U;
  poplar::Input<T> prob_0;
  poplar::Input<int> input_len;
  poplar::Input<int> label_len;
  // U = label_len + 1
  poplar::InOut<float> out;
  const unsigned int t;
  const unsigned int u;
  bool compute() {
    if (t == input_len - 1) {
      if (u == label_len) {
        *out = float(*prob_0);
      }
    } else { // t < T - 1
      if (u == label_len) {
        *out = beta_T1_U + float(*prob_0);
      }
    }

    return true;
  }
};

template class BetaUVertex<float>;
template class BetaUVertex<half>;

float sum(const size_t limit, float max, float *probs) {
  float res = 0.0f;
  for (unsigned int j = 0; j < limit; ++j) {
    res += std::exp(probs[j] - max);
  }
  return res;
}

#ifdef __IPU__
float sum4(const size_t limit, half max, half *probs) {
  float res = 0.0f;
  half4 res4 = {0.0f, 0.0f, 0.0f, 0.0f};
  half4 max4 = half4{max, max, max, max};
  half4 *in = reinterpret_cast<half4 *>(probs);
  size_t end = limit / 4;
  for (unsigned int j = 0; j < end; ++j) {
    half4 tmp = half4_exp(in[j] - max4);
    res4 += tmp;
  }
  res = float(res4[0]) + float(res4[1]) + float(res4[2]) + float(res4[3]);
  unsigned int remain = limit % 4;
  for (unsigned int i = 0; i < remain; ++i) {
    res += float(half_exp(probs[limit - 1 - i] - max));
  }
  return res;
}

float sum(const size_t limit, half max, half *probs) {
  if ((uint32_t(probs) % 8) == 0)
    return sum4(limit, max, probs);
  float res = 0.0f;
  for (unsigned int j = 0; j < limit; ++j) {
    res += std::exp(float(probs[j] - max));
  }
  return res;
}
#else
float sum(const size_t limit, half max, half *probs) {
  float res = 0.0f;
  for (unsigned int j = 0; j < limit; ++j) {
    res += std::exp(float(probs[j] - max));
  }
  return res;
}
#endif

template <typename T> class SoftmaxMapVertex : public poplar::Vertex {
public:
  SoftmaxMapVertex();

  poplar::Input<poplar::Vector<T>> probs; // size == alphabet
  poplar::Input<poplar::Vector<T>> max;
  poplar::Output<poplar::Vector<T>> out;
  const uint32_t alphabet;

  bool compute() {
    for (unsigned int i = 0; i < max.size(); ++i) {
      out[i] = T(std::log(sum(alphabet, max[i], const_cast <T *>(&probs[i * alphabet]))));
    }
    return true;
  }
};

template class SoftmaxMapVertex<float>;
template class SoftmaxMapVertex<half>;

template <typename T> class BetaVertexCompact : public poplar::Vertex {
public:
  BetaVertexCompact();
  poplar::Input<poplar::Vector<T>> probs_T_U;

  poplar::Input<float> beta_T_U1;
  poplar::Input<float> beta_T1_U;

  poplar::Input<int> input_len;
  poplar::Input<int> label_len;

  // U = label_len + 1
  poplar::InOut<float> out;
  const unsigned int t;
  const unsigned int u;
  bool compute() {
    if (t == input_len - 1) {
      if (u == label_len) {
        *out = float(probs_T_U[0]);
      } else { // u < U - 1
        *out = beta_T_U1 + float(probs_T_U[1]);
      }
    } else { // t < T - 1
      if (u == label_len) {
        *out = beta_T1_U + float(probs_T_U[0]);
      } else { // u < U - 1
        float no_emit = beta_T1_U + float(probs_T_U[0]);
        float emit = beta_T_U1 + float(probs_T_U[1]);
        *out = log_sum_exp(emit, no_emit);
      }
    }

    return true;
  }
};

template class BetaVertexCompact<float>;
template class BetaVertexCompact<half>;

template <typename T> class MapLossVertex : public poplar::Vertex {
public:
  poplar::Input<int> input_len;
  poplar::Input<int> label_len;
  poplar::Input<int> u;
  poplar::Input<poplar::Vector<float>> alphas;
  poplar::Input<poplar::Vector<T>> probs;
  poplar::Output<float> out;
  bool compute() {
    if (u == label_len) {
      *out = -(alphas[input_len - 1] + float(probs[input_len - 1]));
    } else {
      *out = -std::numeric_limits<float>::infinity();
    }
    return true;
  }
};

template class MapLossVertex<float>;
template class MapLossVertex<half>;

template <typename T> class GradientsTVertex : public poplar::Vertex {
public:
  GradientsTVertex();
  poplar::InOut<float> out;
  poplar::Input<int> input_len;
  poplar::Input<int> label_len;
  poplar::Input<float> alpha_T_U;
  poplar::Input<float> beta_T1_U;
  poplar::Input<float> loss;
  poplar::Input<T> prob;
  const unsigned int t;
  const unsigned int u;

  bool compute() {
    if (t < input_len - 1 || u == label_len) {
      float g;
      if (t < input_len - 1) {
        g = alpha_T_U + beta_T1_U;
      } else {
        g = alpha_T_U;
      }
      if (t < input_len && u <= label_len)
        *out = -std::exp(loss + g + float(*prob));
    }
    return true;
  }
};

template class GradientsTVertex<float>;
template class GradientsTVertex<half>;

template <typename T> class Gradients0Vertex : public poplar::Vertex {
public:
  Gradients0Vertex();
  poplar::InOut<float> out;
  poplar::Input<int> input_len;
  poplar::Input<int> label_len;
  poplar::Input<float> alpha_T_U;
  poplar::Input<float> loss;
  poplar::Input<T> prob;
  const unsigned int t;
  const unsigned int u;

  bool compute() {
    if (u == label_len && t == input_len - 1) {
      *out = -std::exp(loss + alpha_T_U + float(*prob));
    }
    return true;
  }
};

template class Gradients0Vertex<float>;
template class Gradients0Vertex<half>;

template <typename T> class GradientsUVertexCompact : public poplar::Vertex {
public:
  GradientsUVertexCompact();
  poplar::InOut<float> out;
  poplar::Input<int> input_len;
  poplar::Input<int> label_len;
  poplar::Input<float> alpha_T_U;
  poplar::Input<float> beta_T_U1;
  poplar::Input<float> loss;
  poplar::Input<T> prob;
  const unsigned int t;
  const unsigned int u;

  bool compute() {
    if (u < label_len && t < input_len) {
      float g = alpha_T_U + beta_T_U1;
      *out = -std::exp(loss + g + float(*prob));
    }
    return true;
  }
};

template class GradientsUVertexCompact<float>;
template class GradientsUVertexCompact<half>;

template <typename T, typename T2> class CopyVertex : public poplar::Vertex {
public:
  poplar::Input<T> in;
  poplar::InOut<T2> out;

  bool compute() {
    *out = T2(in);
    return true;
  }
};

template class CopyVertex<float, float>;
template class CopyVertex<float, half>;
template class CopyVertex<half, half>;

template <typename T, typename T2> class AddVertex : public poplar::Vertex {
public:
  poplar::Input<T> in;
  poplar::InOut<T2> out;

  bool compute() {
    *out += T2(in);
    return true;
  }
};

template class AddVertex<float, float>;
template class AddVertex<float, half>;
template class AddVertex<half, half>;

template <typename T> class CopyIndexVertex : public poplar::Vertex {
public:
  CopyIndexVertex();
  poplar::Input<poplar::Vector<T>> in;
  poplar::InOut<T> out;
  poplar::Input<int> label_len;
  poplar::Input<int> label;
  const unsigned int u;

  bool compute() {
    if (u < label_len) {
      *out = in[label];
    }
    return true;
  }
};

template class CopyIndexVertex<float>;
template class CopyIndexVertex<half>;

template <typename I, typename O>
class CopyExpandVertex : public poplar::Vertex {
public:
  CopyExpandVertex();
  poplar::Input<I> in;
  poplar::InOut<poplar::Vector<O>> out;
  poplar::Input<int> label_len;
  poplar::Input<int> label;
  const unsigned int u;

  bool compute() {
    if (u < label_len) {
      out[label - 1] = O(in);
    }
    return true;
  }
};

template class CopyExpandVertex<float, float>;
template class CopyExpandVertex<float, half>;
template class CopyExpandVertex<half, half>;

template <typename I, typename O>
class AddExpandVertex : public poplar::Vertex {
public:
  AddExpandVertex();
  poplar::Input<I> in;
  poplar::InOut<poplar::Vector<O>> out;
  poplar::Input<int> label_len;
  poplar::Input<int> label;
  const unsigned int u;

  bool compute() {
    if (u < label_len) {
      out[label - 1] += O(in);
    }
    return true;
  }
};

template class AddExpandVertex<float, float>;
template class AddExpandVertex<float, half>;
template class AddExpandVertex<half, half>;

class CompactLogProbs : public poplar::Vertex {
public:
  poplar::Vector<poplar::Input<poplar::Vector<float, ONE_PTR>>> logProbs_BTU_A;
  poplar::Input<poplar::Vector<unsigned int, ONE_PTR>> labels_BTU;
  poplar::Output<poplar::Vector<float, ONE_PTR>> logProbsCompacted_2_BTU$0;
  poplar::Output<poplar::Vector<float, ONE_PTR>> logProbsCompacted_2_BTU$1;

  bool compute() {
    for (unsigned idxBTU = 0; idxBTU < logProbs_BTU_A.size(); ++idxBTU) {
      logProbsCompacted_2_BTU$0[idxBTU] = logProbs_BTU_A[idxBTU][0];
      logProbsCompacted_2_BTU$1[idxBTU] =
          logProbs_BTU_A[idxBTU][labels_BTU[idxBTU]];
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Dynamic version codelets
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T> class CompactLogProbsVertex : public poplar::Vertex {
public:
  poplar::Vector<poplar::Input<poplar::Vector<T, ONE_PTR>>> logProbs_BTU_A;
  poplar::Input<poplar::Vector<int, ONE_PTR>> labels_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> labelLengths_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> uMap_BTU;
  poplar::Output<poplar::Vector<float, ONE_PTR>> logProbsCompacted0_BTU;
  poplar::Output<poplar::Vector<float, ONE_PTR>> logProbsCompacted1_BTU;

  bool compute() {
    for (unsigned idxBTU = 0; idxBTU < logProbs_BTU_A.size(); ++idxBTU) {
      int u = uMap_BTU[idxBTU];
      int ll = labelLengths_BTU[idxBTU];
      logProbsCompacted0_BTU[idxBTU] =
          static_cast<float>(logProbs_BTU_A[idxBTU][0]);
      if (u < ll) {
        logProbsCompacted1_BTU[idxBTU] =
            static_cast<float>(logProbs_BTU_A[idxBTU][labels_BTU[idxBTU]]);
      } else {
        logProbsCompacted1_BTU[idxBTU] =
            -std::numeric_limits<float>::infinity();
      }
    }
    return true;
  }
};

template class CompactLogProbsVertex<float>;
template class CompactLogProbsVertex<half>;

class IncrementNVertex : public poplar::Vertex {
public:
  poplar::InOut<poplar::Vector<unsigned, ONE_PTR>> counterN;
  bool compute() {
    ++counterN[0];
    return true;
  }
};

class DecrementNVertex : public poplar::Vertex {
public:
  poplar::InOut<poplar::Vector<unsigned, ONE_PTR>> counterN;
  bool compute() {
    --counterN[0];
    return true;
  }
};

template <typename T> class AlphaDynamicVertex : public poplar::Vertex {
public:
  poplar::Input<unsigned> counterN;
  poplar::Vector<poplar::Input<poplar::Vector<T, ONE_PTR>>>
      logProbsNSlice_UB_A2;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logAlphaSliceIn_Um1B;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logAlphaSliceIn_UB;
  poplar::Input<poplar::Vector<int, ONE_PTR>> inputLengths_UB;
  poplar::Input<poplar::Vector<int, ONE_PTR>> uMap_UB;
  poplar::InOut<poplar::Vector<float>> logAlphaSliceOut_UB;

  bool compute() {
    int n = static_cast<int>(counterN);
    for (std::size_t i = 0; i < logAlphaSliceOut_UB.size(); ++i) {
      int u = uMap_UB[i];
      int t = n - u;
      // t + 1 is the target layer
      int tLen = inputLengths_UB[i];
      if (t + 1 < tLen) {
        float noEmit = logAlphaSliceIn_UB[i] +
                       static_cast<float>(logProbsNSlice_UB_A2[i][0]);
        float emit = logAlphaSliceIn_Um1B[i] +
                     static_cast<float>(logProbsNSlice_UB_A2[i][1]);
        logAlphaSliceOut_UB[i] = log_sum_exp(emit, noEmit);
      } else {
        logAlphaSliceOut_UB[i] = -std::numeric_limits<float>::infinity();
      }
    }
    return true;
  }
};

template class AlphaDynamicVertex<float>;
template class AlphaDynamicVertex<half>;

template <typename T> class BetaDynamicVertex : public poplar::Vertex {
public:
  poplar::Input<unsigned> counterN;
  poplar::Vector<poplar::Input<poplar::Vector<T, ONE_PTR>>>
      logProbsNSlice_UB_A2;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logBetaSliceIn_Up1B;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logBetaSliceIn_UB;
  poplar::Input<poplar::Vector<int, ONE_PTR>> inputLengths_UB;
  poplar::Input<poplar::Vector<int, ONE_PTR>> labelLengths_UB;
  poplar::Input<poplar::Vector<int, ONE_PTR>> uMap_UB;
  poplar::InOut<poplar::Vector<float>> logBetaSliceOut_UB;

  bool compute() {
    int n = static_cast<int>(counterN);
    for (std::size_t i = 0; i < logBetaSliceOut_UB.size(); ++i) {
      int u = uMap_UB[i];
      int t = n - u;
      int tLen = inputLengths_UB[i];
      int uLen = labelLengths_UB[i];
      if (t < 0 || t >= tLen || u > uLen) {
        logBetaSliceOut_UB[i] = -std::numeric_limits<float>::infinity();
      } else if (t == tLen - 1 && u == uLen) {
        logBetaSliceOut_UB[i] = static_cast<float>(logProbsNSlice_UB_A2[i][0]);
      } else {
        float noEmit = logBetaSliceIn_UB[i] +
                       static_cast<float>(logProbsNSlice_UB_A2[i][0]);
        float emit = logBetaSliceIn_Up1B[i] +
                     static_cast<float>(logProbsNSlice_UB_A2[i][1]);
        logBetaSliceOut_UB[i] = log_sum_exp(emit, noEmit);
        // logBetaSliceOut_UB[i] = emit + noEmit;
      }
    }
    return true;
  }
};

template class BetaDynamicVertex<float>;
template class BetaDynamicVertex<half>;

template <typename T> class Grads0DynamicVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<T>> logProbs_0_BTU;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logAlpha_BTU;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logBeta_BTs1U;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logLoss_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> inputLengths_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> labelLengths_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> tMap_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> uMap_BTU;
  poplar::Output<poplar::Vector<float, ONE_PTR>> grads_0;

  bool compute() {
    for (std::size_t i = 0; i < logProbs_0_BTU.size(); ++i) {
      int tLen = inputLengths_BTU[i];
      int uLen = labelLengths_BTU[i];
      int t = tMap_BTU[i];
      int u = uMap_BTU[i];
      if (t < tLen - 1) {
        float g = logAlpha_BTU[i] + logBeta_BTs1U[i];
        grads_0[i] = -std::exp(logLoss_BTU[i] + g +
                               static_cast<float>(logProbs_0_BTU[i]));
      } else if (t == tLen - 1 && u == uLen) {
        float g = logAlpha_BTU[i];
        grads_0[i] = -std::exp(logLoss_BTU[i] + g +
                               static_cast<float>(logProbs_0_BTU[i]));
      } else {
        grads_0[i] = 0.0f;
      }
    }
    return true;
  }
};

template class Grads0DynamicVertex<float>;
template class Grads0DynamicVertex<half>;

template <typename T> class Grads1DynamicVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<T>> logProbs_1_BTU;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logAlpha_BTU;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logBeta_BTUs1;
  poplar::Input<poplar::Vector<float, ONE_PTR>> logLoss_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> inputLengths_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> labelLengths_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> tMap_BTU;
  poplar::Input<poplar::Vector<int, ONE_PTR>> uMap_BTU;
  poplar::Output<poplar::Vector<float, ONE_PTR>> grads_1;

  bool compute() {
    for (std::size_t i = 0; i < logProbs_1_BTU.size(); ++i) {
      int tLen = inputLengths_BTU[i];
      int uLen = labelLengths_BTU[i];
      int t = tMap_BTU[i];
      int u = uMap_BTU[i];
      if (t < tLen && u < uLen) {
        float g = logAlpha_BTU[i] + logBeta_BTUs1[i];
        grads_1[i] = -std::exp(logLoss_BTU[i] + g +
                               static_cast<float>(logProbs_1_BTU[i]));
      } else {
        grads_1[i] = 0.0f;
      }
    }
    return true;
  }
};

template class Grads1DynamicVertex<float>;
template class Grads1DynamicVertex<half>;

template <typename TP> class expandAndSubVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float>> compactedT0;
  poplar::Input<poplar::Vector<float, ONE_PTR>> compactedT1;
  poplar::Vector<poplar::InOut<poplar::Vector<TP, ONE_PTR>>> subT;
  poplar::Input<poplar::Vector<int, ONE_PTR>> labels;
  poplar::Input<poplar::Vector<int, ONE_PTR>> labelLengths;
  poplar::Input<unsigned> intervalStart;
  poplar::Input<poplar::Vector<unsigned, ONE_PTR>> dimsT;

  bool compute() {
    unsigned B = dimsT[0];
    unsigned T = dimsT[1];
    unsigned U = dimsT[2];
    unsigned A = dimsT[3];
    for (unsigned idxBtu = 0, start = intervalStart;
         idxBtu < compactedT0.size(); ++idxBtu, ++start) {
      unsigned u = start % U;
      unsigned res = start / U;
      unsigned t = res % T;
      unsigned b = res / T;

      for (unsigned a = 0; a < A; ++a) {
        subT[idxBtu][a] = -subT[idxBtu][a];
      }
      subT[idxBtu][0] += static_cast<TP>(compactedT0[idxBtu]);
      int ll = labelLengths[b];
      if (u < ll) {
        int label = labels[idxBtu];
        subT[idxBtu][label] += static_cast<TP>(compactedT1[idxBtu]);
      }
    }
    return true;
  }
};

template class expandAndSubVertex<float>;
template class expandAndSubVertex<half>;
