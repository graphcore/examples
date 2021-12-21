// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Based on "Sequence Transduction with Recurrent Neural Network" by Graves

#include "rnnt.hpp"
#include <cmath>
#include <tuple>
#include <unordered_map>

namespace {

inline float log_sum_exp(float a, float b) {
  if (std::isinf(a) && a < 0)
    return b;
  if (std::isinf(b) && b < 0)
    return a;
  if (a > b)
    return std::log1p(exp(b - a)) + a;
  else
    return std::log1p(exp(a - b)) + b;
}

float alpha(const NDArray<float> &log_probs, size_t input_length,
            const NDArray<int32_t> &labels, size_t label_length, size_t batch,
            const NDArray<float> &alphas, size_t t, size_t u) {
  float res = -std::numeric_limits<float>::infinity();
  if (t == 0) {
    if (u == 0) {
      return 0.0f;
    } else { // u > 0, t == 0
      if (u - 1 < label_length) {
        size_t label = labels[{batch, u - 1}];
        res = alphas[{batch, 0, u - 1}] + log_probs[{batch, 0, u - 1, label}];
      }
    }
  } else { // t > 0
    if (u == 0) {
      res = alphas[{batch, t - 1, 0}] + log_probs[{batch, t - 1, 0, 0
                                                   /*blank*/}];
    } else { // t > 0, u > 0
      float no_emit =
          alphas[{batch, t - 1, u}] + log_probs[{batch, t - 1, u, 0 /*blank*/}];
      size_t label = labels[{batch, u - 1}];
      float emit =
          alphas[{batch, t, u - 1}] + log_probs[{batch, t, u - 1, label}];
      res = log_sum_exp(emit, no_emit);
    }
  }
  return res;
}

void alpha(const NDArray<float> &log_probs, size_t input_length,
           const NDArray<int32_t> &labels, size_t label_length, size_t batch,
           const NDArray<float> &alphas, size_t t, size_t u, float &out) {
  if (t == 0) {
    if (u == 0) {
      out = 0.0f;
    } else { // u > 0, t == 0
      if (u - 1 < label_length) {
        size_t label = labels[{batch, u - 1}];
        out = alphas[{batch, 0, u - 1}] + log_probs[{batch, 0, u - 1, label}];
      }
    }
  } else { // t > 0
    if (t < input_length) {
      if (u == 0) {
        out = alphas[{batch, t - 1, 0}] + log_probs[{batch, t - 1, 0, 0
                                                     /*blank*/}];
      } else { // t > 0, u > 0
        float no_emit = alphas[{batch, t - 1, u}] +
                        log_probs[{batch, t - 1, u, 0 /*blank*/}];
        size_t label = labels[{batch, u - 1}];
        float emit =
            alphas[{batch, t, u - 1}] + log_probs[{batch, t, u - 1, label}];
        out = log_sum_exp(emit, no_emit);
      }
    }
  }
}

float alpha(const NDArray<float> &log_probs, size_t input_length,
            const NDArray<int32_t> &labels, size_t label_length, size_t batch,
            NDArray<float> &alphas) {
  auto dims = log_probs.dims();
  size_t U = label_length + 1;
  for (size_t t = 0; t < input_length; t++) {
    for (size_t u = 0; u < U; u++) {
      alphas[{batch, t, u}] = alpha(log_probs, input_length, labels,
                                    label_length, batch, alphas, t, u);
    }
  }
  float forward_ll = alphas[{batch, input_length - 1, U - 1}] +
                     log_probs[{batch, input_length - 1, U - 1, 0 /*blank*/}];
  return -forward_ll;
}

float beta(const NDArray<float> &log_probs, size_t input_length,
           const NDArray<int32_t> &labels, size_t label_length, size_t batch,
           const NDArray<float> &betas, size_t t, size_t u) {
  size_t U = label_length + 1;
  float res;
  if (t == input_length - 1) {
    if (u == U - 1) {
      res = log_probs[{batch, input_length - 1, U - 1, 0 /*blank*/}];
    } else { // u < U - 1
      size_t label = labels[{batch, size_t(u)}];
      res = betas[{batch, input_length - 1, size_t(u + 1)}] +
            log_probs[{batch, input_length - 1, size_t(u), label}];
    }
  } else { // t < T - 1
    if (u == U - 1) {
      res = betas[{batch, size_t(t + 1), U - 1}] +
            log_probs[{batch, size_t(t), U - 1, 0 /*blank*/}];
    } else { // u < U - 1
      float no_emit = betas[{batch, size_t(t + 1), size_t(u)}] +
                      log_probs[{batch, size_t(t), size_t(u), 0 /*blank*/}];
      size_t label = labels[{batch, size_t(u)}];
      float emit = betas[{batch, size_t(t), size_t(u + 1)}] +
                   log_probs[{batch, size_t(t), size_t(u), label}];
      res = log_sum_exp(emit, no_emit);
    }
  }
  return res;
}

float beta(const NDArray<float> &log_probs, size_t input_length,
           const NDArray<int32_t> &labels, size_t label_length, size_t batch,
           NDArray<float> &betas) {
  auto dims = log_probs.dims();
  size_t U = label_length + 1;
  for (int t = input_length - 1; t >= 0; --t) {
    for (int u = U - 1; u >= 0; --u) {
      betas[{batch, size_t(t), size_t(u)}] = beta(
          log_probs, input_length, labels, label_length, batch, betas, t, u);
    }
  }

  float backward_ll = betas[{batch, 0, 0}];
  return backward_ll;
}

}; // namespace

std::pair<std::vector<float>, NDArray<float>>
alpha(const std::vector<float> &log_probs, const Shape &shape,
      const std::vector<size_t> &input_lengths,
      const std::vector<int32_t> &labels,
      const std::vector<size_t> &label_lengths) {
  assert(shape.size() == log_probs.size());
  auto dims = shape.dims();
  assert(dims.size() == 4);
  size_t batch_size = dims[0];
  size_t maxT = dims[1];
  size_t maxU = dims[2];
  // size_t alphabet = dims[3];
  Shape labels_shape{batch_size, maxU - 1};
  Shape alphas_shape{batch_size, maxT, maxU};
  NDArray<float> alphas(alphas_shape, -std::numeric_limits<float>::infinity());
  NDArray<float> probs{shape, log_probs};
  NDArray<int32_t> labels_{labels_shape, labels};
  std::vector<float> losses(batch_size);
  assert(input_lengths.size() == batch_size);
  assert(label_lengths.size() == batch_size);
  assert(labels.size() == batch_size * (maxU - 1));
  for (size_t i = 0; i < batch_size; ++i) {
    losses[i] =
        alpha(probs, input_lengths[i], labels_, label_lengths[i], i, alphas);
  }
  return {losses, alphas};
}

NDArray<float> compact_cpu(const std::vector<float> &logProbs,
                           const Shape &shape,
                           const std::vector<int32_t> &labels,
                           const std::vector<int32_t> &labelLengths) {
  assert(shape.size() == logProbs.size());
  auto dims = shape.dims();
  assert(dims.size() == 4);
  size_t batch_size = dims[0];
  size_t T = dims[1];
  size_t U = dims[2];
  assert(labelLengths.size() == batch_size);
  assert(labels.size() == batch_size * (U - 1));
  unsigned labelLen = U - 1;

  NDArray<float> logProbsS(shape, logProbs);
  Shape labelsShape{batch_size, labelLen};
  NDArray<int> labelsS{labelsShape, labels};
  Shape compactedShape({batch_size, T, U, 2});
  NDArray<float> logProbsCompacted(compactedShape,
                                   -std::numeric_limits<float>::infinity());
  for (unsigned b = 0; b < batch_size; ++b) {
    for (uint32_t t = 0; t < T; ++t) {
      for (uint32_t u = 0; u < U; ++u) {
        logProbsCompacted[{b, t, u, 0}] = logProbsS[{b, t, u, 0}];
        if (u < uint32_t(labelLengths[b])) {
          uint32_t label = 0;
          if (u != U - 1) {
            label = labelsS[{b, u}];
            logProbsCompacted[{b, t, u, 1}] = logProbsS[{b, t, u, label}];
          }
        }
      }
    }
  }
  return logProbsCompacted;
}

NDArray<float> expand_cpu(const std::vector<float> &compactGrads,
                          const Shape &shape, uint32_t alphabet,
                          const std::vector<int32_t> &labels,
                          const std::vector<int32_t> &labelLengths) {
  auto dims = shape.dims();
  assert(dims.size() == 4);
  size_t batch_size = dims[0];
  size_t T = dims[1];
  size_t U = dims[2];
  assert(labelLengths.size() == batch_size);
  assert(labels.size() == batch_size * (U - 1));
  unsigned labelLen = U - 1;

  NDArray<float> compactedS(shape, compactGrads);
  Shape labelsShape{batch_size, labelLen};
  NDArray<int> labelsS{labelsShape, labels};

  Shape gradsShape({batch_size, T, U, alphabet});
  NDArray<float> grads(gradsShape);
  for (uint32_t b = 0; b < batch_size; ++b) {
    for (uint32_t t = 0; t < T; ++t) {
      for (uint32_t u = 0; u < U; ++u) {
        grads[{b, t, u, 0}] = compactedS[{b, t, u, 0}];
        if (u < uint32_t(labelLengths[b])) {
          size_t label = 0;
          if (u != U - 1) {
            label = labelsS[{b, u}];
          }
          grads[{b, t, u, label}] = compactedS[{b, t, u, 1}];
        }
      }
    }
  }
  return grads;
}

std::pair<std::vector<float>, NDArray<float>>
alpha_static(const std::vector<float> &log_probs, const Shape &shape,
             const std::vector<size_t> &input_lengths,
             const std::vector<int32_t> &labels,
             const std::vector<size_t> &label_lengths) {
  assert(shape.size() == log_probs.size());
  auto dims = shape.dims();
  assert(dims.size() == 4);
  size_t batch_size = dims[0];
  size_t maxT = dims[1];
  size_t maxU = dims[2];
  // size_t alphabet = dims[3];
  Shape labels_shape{batch_size, maxU - 1};
  Shape alphas_shape{batch_size, maxT, maxU};
  NDArray<float> alphas(alphas_shape, -std::numeric_limits<float>::infinity());
  NDArray<float> probs{shape, log_probs};
  NDArray<int32_t> labels_{labels_shape, labels};
  std::vector<float> losses(batch_size);
  assert(input_lengths.size() == batch_size);
  assert(label_lengths.size() == batch_size);
  assert(labels.size() == batch_size * (maxU - 1));
  size_t U = maxU;
  size_t T = maxT;
  for (size_t tp = 0; tp < T + U - 1; tp++) {
    size_t start = size_t(std::max(int(tp) - int(T) + 1, 0));
    size_t end = std::min(U - 1, tp);       // inclusive end
    for (size_t z = start; z <= end; z++) { // this loop is parallel
      size_t u = z;
      size_t t = tp - z;
      for (size_t i = 0; i < batch_size; ++i) {
        alpha(probs, input_lengths[i], labels_, label_lengths[i], i, alphas, t,
              u, alphas[{i, t, u}]);
      }
    }
  }
  for (size_t i = 0; i < batch_size; ++i) {
    float forward_ll =
        alphas[{i, input_lengths[i] - 1, label_lengths[i]}] +
        probs[{i, input_lengths[i] - 1, label_lengths[i], 0 /*blank*/}];
    losses[i] = -forward_ll;
  }
  return {losses, alphas};
}

std::pair<std::vector<float>, NDArray<float>>
beta(const std::vector<float> &log_probs, const Shape &shape,
     const std::vector<size_t> &input_lengths,
     const std::vector<int32_t> &labels,
     const std::vector<size_t> &label_lengths) {
  assert(shape.size() == log_probs.size());
  auto dims = shape.dims();
  assert(dims.size() == 4);
  size_t batch_size = dims[0];
  size_t maxT = dims[1];
  size_t maxU = dims[2];
  // size_t alphabet = dims[3];
  Shape labels_shape{batch_size, maxU - 1};
  Shape betas_shape{batch_size, maxT, maxU};
  NDArray<float> betas(betas_shape, -std::numeric_limits<float>::infinity());
  NDArray<float> probs{shape, log_probs};
  NDArray<int32_t> labels_{labels_shape, labels};
  std::vector<float> losses(batch_size);
  assert(input_lengths.size() == batch_size);
  assert(label_lengths.size() == batch_size);
  assert(labels.size() == batch_size * (maxU - 1));
  for (size_t i = 0; i < batch_size; ++i) {
    losses[i] =
        -beta(probs, input_lengths[i], labels_, label_lengths[i], i, betas);
  }
  return {losses, betas};
}

std::pair<std::vector<float>, NDArray<float>>
beta_static(const std::vector<float> &log_probs, const Shape &shape,
            const std::vector<size_t> &input_lengths,
            const std::vector<int32_t> &labels,
            const std::vector<size_t> &label_lengths) {
  assert(shape.size() == log_probs.size());
  auto dims = shape.dims();
  assert(dims.size() == 4);
  size_t batch_size = dims[0];
  size_t maxT = dims[1];
  size_t maxU = dims[2];
  // size_t alphabet = dims[3];
  Shape labels_shape{batch_size, maxU - 1};
  Shape betas_shape{batch_size, maxT, maxU};
  NDArray<float> betas(betas_shape, -std::numeric_limits<float>::infinity());
  NDArray<float> probs{shape, log_probs};
  NDArray<int32_t> labels_{labels_shape, labels};
  std::vector<float> losses(batch_size);
  assert(input_lengths.size() == batch_size);
  assert(label_lengths.size() == batch_size);
  assert(labels.size() == batch_size * (maxU - 1));

  size_t U = maxU;
  size_t T = maxT;
  for (int tp = T + U - 2; tp >= 0; --tp) {
    size_t start = size_t(std::max(tp - int(T) + 1, 0));
    size_t end = std::min(U - 1, size_t(tp)); // inclusive end
    for (size_t z = start; z <= end; z++) {   // this loop is parallel
      size_t u = z;
      size_t t = tp - z;
      for (size_t i = 0; i < batch_size; ++i) {
        betas[{i, size_t(t), size_t(u)}] = beta(
            probs, input_lengths[i], labels_, label_lengths[i], i, betas, t, u);
      }
    }
  }

  for (size_t i = 0; i < batch_size; ++i) {
    losses[i] = -betas[{i, 0, 0}];
  }
  return {losses, betas};
}

void grads(const NDArray<float> &log_probs, size_t input_length,
           const NDArray<int32_t> &labels, size_t label_length, size_t batch,
           const NDArray<float> &alphas, const NDArray<float> &betas,
           float loss, NDArray<float> &grads) {
  auto dims = log_probs.dims();
  size_t alphabet = dims[3];
  size_t U = label_length + 1;
  grads[{batch, input_length - 1, U - 1, 0 /*blank*/}] =
      alphas[{batch, input_length - 1, U - 1}];
  for (size_t t = 0; t < input_length - 1; t++) {
    for (size_t u = 0; u < U; u++) {
      grads[{batch, t, u, 0 /*blank*/}] =
          alphas[{batch, t, u}] + betas[{batch, t + 1, u}];
    }
  }
  for (size_t t = 0; t < input_length; t++) {
    for (size_t u = 0; u < U - 1; u++) {
      size_t label = labels[{batch, u}];
      grads[{batch, t, u, label}] =
          alphas[{batch, t, u}] + betas[{batch, t, u + 1}];
    }
  }
  for (size_t t = 0; t < input_length; t++) {
    for (size_t u = 0; u < U; u++) {
      for (size_t v = 0; v < alphabet; v++) {
        float g = grads[{batch, t, u, v}];
        if (g != 0) {
          grads[{batch, t, u, v}] =
              -std::exp(loss + g + log_probs[{batch, t, u, v}]);
        }
      }
    }
  }
}

// From (20)
void grads_sparse(const NDArray<float> &log_probs, size_t input_length,
                  const NDArray<int32_t> &labels, size_t label_length,
                  size_t batch, const NDArray<float> &alphas,
                  const NDArray<float> &betas, float loss,
                  NDArray<float> &grads) {
  auto dims = log_probs.dims();
  size_t T = input_length;
  size_t U = label_length + 1;
  for (size_t t = 0; t < T; t++) {
    for (size_t u = 0; u < U; u++) {
      if (t < T - 1) {
        float g = alphas[{batch, t, u}] + betas[{batch, t + 1, u}];
        float lp = log_probs[{batch, t, u, 0}];
        grads[{batch, t, u, 0}] = -std::exp(loss + g + lp);
      } else if (t == T - 1 && u == U - 1) {
        float g = alphas[{batch, t, u}];
        float lp = log_probs[{batch, t, u, 0}];
        grads[{batch, t, u, 0}] = -std::exp(loss + g + lp);
      } else {
        grads[{batch, t, u, 0}] = 0.0f;
      }
      if (u < U - 1) {
        float g = alphas[{batch, t, u}] + betas[{batch, t, u + 1}];
        size_t label = labels[{batch, u}];
        float lp = log_probs[{batch, t, u, label}];
        grads[{batch, t, u, label}] = -std::exp(loss + g + lp);
      }
    }
  }
}

float grads(const NDArray<float> &log_probs, size_t input_length,
            const NDArray<int32_t> &labels, size_t label_length, size_t batch,
            const NDArray<float> &alphas, const NDArray<float> &betas,
            float loss, const NDArray<float> &grads, size_t t, size_t u) {
  if ((t == input_length - 1) && (u == label_length)) {
    return alphas[{batch, t, u}];
  }
  return 0.0;
}

NDArray<float> grads_static(const std::vector<float> &log_probs,
                            const Shape &shape,
                            const std::vector<size_t> &input_lengths,
                            const std::vector<int32_t> &labels,
                            const std::vector<size_t> &label_lengths,
                            const NDArray<float> &alphas,
                            const NDArray<float> &betas,
                            const std::vector<float> &losses) {
  assert(shape.size() == log_probs.size());
  auto dims = shape.dims();
  assert(dims.size() == 4);
  size_t batch_size = dims[0];
  size_t maxT = dims[1];
  size_t maxU = dims[2];
  // size_t alphabet = dims[3];
  Shape labels_shape{batch_size, maxU - 1};
  NDArray<float> probs{shape, log_probs};
  NDArray<float> grads_{shape};
  NDArray<int32_t> labels_{labels_shape, labels};
  assert(input_lengths.size() == batch_size);
  assert(label_lengths.size() == batch_size);
  assert(labels.size() == batch_size * (maxU - 1));
  for (size_t t = 0; t < maxT; t++) {
    for (size_t u = 0; u < maxU; u++) {
      for (size_t i = 0; i < batch_size; ++i) {
        if (t < input_lengths[i] - 1 || u == label_lengths[i]) {
          float g;
          if (t < input_lengths[i] - 1) {
            g = alphas[{i, t, u}] + betas[{i, t + 1, u}];
          } else {
            g = alphas[{i, t, u}];
          }
          grads_[{i, t, u, 0}] = -std::exp(losses[i] + g + probs[{i, t, u, 0}]);
        }
        if (u < label_lengths[i]) {
          float g = alphas[{i, t, u}] + betas[{i, t, u + 1}];
          size_t label = labels_[{i, u}];
          grads_[{i, t, u, label}] =
              -std::exp(losses[i] + g + probs[{i, t, u, label}]);
        }
      }
    }
  }
  return grads_;
}

NDArray<float> grads(const std::vector<float> &log_probs, const Shape &shape,
                     const std::vector<size_t> &input_lengths,
                     const std::vector<int32_t> &labels,
                     const std::vector<size_t> &label_lengths,
                     const NDArray<float> &alphas, const NDArray<float> &betas,
                     const std::vector<float> &losses, bool sparse) {

  assert(shape.size() == log_probs.size());
  auto dims = shape.dims();
  assert(dims.size() == 4);
  size_t batch_size = dims[0];
  size_t maxU = dims[2];
  // size_t alphabet = dims[3];
  Shape labels_shape{batch_size, maxU - 1};
  NDArray<float> probs{shape, log_probs};
  NDArray<float> grads_{shape};
  NDArray<int32_t> labels_{labels_shape, labels};
  assert(input_lengths.size() == batch_size);
  assert(label_lengths.size() == batch_size);
  assert(labels.size() == batch_size * (maxU - 1));
  for (size_t i = 0; i < batch_size; ++i) {
    if (sparse) {
      grads_sparse(probs, input_lengths[i], labels_, label_lengths[i], i,
                   alphas, betas, losses[i], grads_);
    } else {
      grads(probs, input_lengths[i], labels_, label_lengths[i], i, alphas,
            betas, losses[i], grads_);
    }
  }
  return grads_;
}
