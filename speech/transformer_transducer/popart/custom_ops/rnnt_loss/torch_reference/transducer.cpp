
#if !defined(APPLE)
#include <omp.h>
#endif

#include <torch/extension.h>

// torch extension adds NDEBUG compiler option first
// and I don't see a way to undefine it
// Standard assert() is disabled, using custom ASSERT
#ifdef DEBUG
  #define ASSERT(cond) if (false == (cond)) abort();
#else
  #define ASSERT(ignore) ((void)0)
#endif

inline float log_sum_exp(float a, float b) {
  if (!isfinite(a))
    return b;
  if (!isfinite(b))
    return a;
  if (a > b)
    return log1p(exp(b - a)) + a;
  else
    return log1p(exp(a - b)) + b;
}

struct Idx {
  Idx(size_t len_)
    : len(len_)
  {}

  size_t len;
};

struct Idx3 : public Idx {
  Idx3(size_t len_)
    : Idx(len_)
  {}

  inline int operator() (int t, int u, int v, int U, int V) {
    size_t idx = t * (U * V) + u * V + v;
    ASSERT(idx < len);
    return idx; 
  }
};

struct Idx2 : public Idx {
  Idx2(size_t len_)
    : Idx(len_)
  {}

  inline int operator() (int t, int u, int U) {
    size_t idx = t * U + u;
    ASSERT(idx < len);
    return idx; 
  }
};

int cumsum(int *lens, int num) {
  int sum = 0;
  for (int i = 0; i < num; i++)
    sum += lens[i];
  return sum;
}

float cost_and_grad_single(float *log_probs, float *grads, size_t logs_len, int *labels,
                           int blank, int T, int U, int V, int s) {
  Idx2 idx2(T * U);
  Idx3 idx3(logs_len);
  // Forward pass
  float *alphas = (float *)malloc(T * U * sizeof(float));
  alphas[0] = 0;
  for (int t = 1; t < T; t++) {
    alphas[idx2(t, 0, U)] =
        alphas[idx2(t - 1, 0, U)] + log_probs[idx3(t - 1, 0, blank, s, V)];
  }

  for (int u = 1; u < U; u++) {
    alphas[idx2(0, u, U)] = alphas[idx2(0, u - 1, U)] +
                            log_probs[idx3(0, u - 1, labels[u - 1], s, V)];
  }

  for (int t = 1; t < T; t++) {
    for (int u = 1; u < U; u++) {
      float no_emit =
          alphas[idx2(t - 1, u, U)] + log_probs[idx3(t - 1, u, blank, s, V)];
      float emit = alphas[idx2(t, u - 1, U)] +
                   log_probs[idx3(t, u - 1, labels[u - 1], s, V)];
      alphas[idx2(t, u, U)] = log_sum_exp(emit, no_emit);
    }
  }
  float forward_ll = alphas[idx2(T - 1, U - 1, U)] +
                     log_probs[idx3(T - 1, U - 1, blank, s, V)];

  // Backward pass
  float *betas = (float *)malloc(T * U * sizeof(float));
  betas[idx2(T - 1, U - 1, U)] = log_probs[idx3(T - 1, U - 1, blank, s, V)];
  for (int t = T - 2; t >= 0; t--) {
    betas[idx2(t, U - 1, U)] =
        betas[idx2(t + 1, U - 1, U)] + log_probs[idx3(t, U - 1, blank, s, V)];
  }
  for (int u = U - 2; u >= 0; u--) {
    betas[idx2(T - 1, u, U)] = betas[idx2(T - 1, u + 1, U)] +
                               log_probs[idx3(T - 1, u, labels[u], s, V)];
  }

  for (int t = T - 2; t >= 0; t--) {
    for (int u = U - 2; u >= 0; u--) {
      float no_emit =
          betas[idx2(t + 1, u, U)] + log_probs[idx3(t, u, blank, s, V)];
      float emit =
          betas[idx2(t, u + 1, U)] + log_probs[idx3(t, u, labels[u], s, V)];
      betas[idx2(t, u, U)] = log_sum_exp(emit, no_emit);
    }
  }
  float backward_ll = betas[0];

  float diff = fabs(backward_ll - forward_ll);
  float diff_tol = fmax(1e-6 * fabs(forward_ll), 1e-8);
  if (diff > diff_tol) {
    printf("WARNING: Forward backward likelihood mismatch %f\n", diff);
  }

  // Gradients w.r.t. log probabilities
  grads[idx3(T - 1, U - 1, blank, s, V)] = alphas[idx2(T - 1, U - 1, U)];
  for (int t = 0; t < T - 1; t++) {
    for (int u = 0; u < U; u++) {
      grads[idx3(t, u, blank, s, V)] =
          alphas[idx2(t, u, U)] + betas[idx2(t + 1, u, U)];
    }
  }
  for (int t = 0; t < T; t++) {
    for (int u = 0; u < U - 1; u++) {
      int l = labels[u];
      grads[idx3(t, u, l, s, V)] =
          alphas[idx2(t, u, U)] + betas[idx2(t, u + 1, U)];
    }
  }
  for (int t = 0; t < T; t++) {
    for (int u = 0; u < U; u++) {
      for (int v = 0; v < V; v++) {
        float g = grads[idx3(t, u, v, s, V)];
        if (g != 0) {
          grads[idx3(t, u, v, s, V)] =
              -exp(-forward_ll + g + log_probs[idx3(t, u, v, s, V)]);
        }
      }
    }
  }

  // Cleanup
  free(alphas);
  free(betas);

  return -forward_ll;
}

void cost_and_grad(float *log_probs, float *grads, size_t logs_len, float *costs,
                   int *flat_labels, int *label_lengths, int *input_lengths,
                   int batch_size, int max_t, int max_u, int alphabet_size,
                   int blank) {

#ifndef DEBUG
#pragma omp parallel for
#endif
  for (int mb = 0; mb < batch_size; ++mb) {
    int T = input_lengths[mb];     // Length of utterance (time)
    int U = label_lengths[mb] + 1; // Length of transcription
    size_t mb_offset = mb * max_t * max_u * alphabet_size;
    int label_offset = (max_u - 1) * mb;
    ASSERT(mb_offset < logs_len);
    costs[mb] = cost_and_grad_single(log_probs + mb_offset, grads + mb_offset, logs_len - mb_offset,
                                     flat_labels + label_offset, blank, T, U,
                                     alphabet_size, max_u);
  }
}

void transduce(torch::Tensor th_log_probs, torch::Tensor th_labels,
               torch::Tensor th_input_lengths, torch::Tensor th_label_lengths,
               torch::Tensor th_costs, torch::Tensor th_grads, int blank) {
  int batch_size = th_log_probs.size(0);
  int max_t = th_log_probs.size(1);
  int max_u = th_log_probs.size(2);
  int alphabet_size = th_log_probs.size(3);

  auto log_probs = th_log_probs.data_ptr<float>();
  auto input_lengths = th_input_lengths.data_ptr<int>();
  int *labels = th_labels.data_ptr<int>();
  int *label_lengths = th_label_lengths.data_ptr<int>();

  float *costs = th_costs.data_ptr<float>();
  float *grads = th_grads.data_ptr<float>();

  size_t gr_dim = static_cast<size_t>(th_grads.dim());
  auto gr_shape = th_grads.sizes();
#ifdef DEBUG
  size_t lp_dim = static_cast<size_t>(th_log_probs.dim());
  auto lp_shape = th_log_probs.sizes();
  ASSERT(gr_dim == lp_dim);
#endif
  size_t logs_len = 1;
  for (size_t i = 0; i < gr_dim; ++i) {
    ASSERT(gr_shape[i] == lp_shape[i]);
    logs_len *= gr_shape[i];
  }

  cost_and_grad(log_probs, grads, logs_len, costs, labels, label_lengths, input_lengths,
                batch_size, max_t, max_u, alphabet_size, blank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transduce", &transduce, "Transduce");
}
