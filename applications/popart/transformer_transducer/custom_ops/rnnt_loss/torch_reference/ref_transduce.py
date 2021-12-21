"""
Python reference implementation of the
sequence transducer.
Author: Awni Hannun

Based on the papers:

 - "Sequence Transduction with Recurrent Neural Networks"
    Graves, 2012
    https://arxiv.org/abs/1211.3711

 - "Speech Recognition with Deep Recurrent Neural Networks"
    Graves, et al., 2013
    https://arxiv.org/abs/1303.5778
"""

import math
import numpy as np

NEG_INF = -float("inf")

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                    for a in args))
    return a_max + lsp

def log_softmax(acts, axis):
    """
    Log softmax over the last axis of the 3D array.
    """
    acts = acts - np.max(acts, axis=axis, keepdims=True)
    probs = np.sum(np.exp(acts), axis=axis, keepdims=True)
    log_probs = acts - np.log(probs)
    return log_probs

def forward_pass(log_probs, labels, blank):

    T, U, _ = log_probs.shape
    alphas = np.zeros((T, U))

    for t in range(1, T):
        alphas[t, 0] = alphas[t-1, 0] + log_probs[t-1, 0, blank]

    for u in range(1, U):
        alphas[0, u] = alphas[0, u-1] + log_probs[0, u-1, labels[u-1]]
    for t in range(1, T):
        for u in range(1, U):
            no_emit = alphas[t-1, u] + log_probs[t-1, u, blank]
            emit = alphas[t, u-1] + log_probs[t, u-1, labels[u-1]]
            alphas[t, u] = logsumexp(emit, no_emit)

    loglike = alphas[T-1, U-1] + log_probs[T-1, U-1, blank]
    return alphas, loglike

def backward_pass(log_probs, labels, blank):

    T, U, _ = log_probs.shape
    betas = np.zeros((T, U))
    betas[T-1, U-1] = log_probs[T-1, U-1, blank]

    for t in reversed(range(T-1)):
        betas[t, U-1] = betas[t+1, U-1] + log_probs[t, U-1, blank]

    for u in reversed(range(U-1)):
        betas[T-1, u] = betas[T-1, u+1] + log_probs[T-1, u, labels[u]]

    for t in reversed(range(T-1)):
        for u in reversed(range(U-1)):
            no_emit = betas[t+1, u] + log_probs[t, u, blank]
            emit = betas[t, u+1] + log_probs[t, u, labels[u]]
            betas[t, u] = logsumexp(emit, no_emit)

    return betas, betas[0, 0]

def compute_gradient(log_probs, alphas, betas, labels, blank):
    T, U, _ = log_probs.shape
    grads = np.full(log_probs.shape, -float("inf"))
    log_like = betas[0, 0]

    grads[T-1, U-1, blank] = alphas[T-1, U-1]

    grads[:T-1, :, blank] = alphas[:T-1, :] + betas[1:, :]
    for u, l in enumerate(labels):
        grads[:, u, l] = alphas[:, u] + betas[:, u+1]

    grads = grads + log_probs - log_like
    grads = np.exp(grads)

    grads = -grads
    return grads

def transduce(log_probs, labels, blank=0):
    """
    Args:
        acts: 3D array with shape
              [input len, output len + 1, vocab size]
        labels: 1D array with shape [output time steps]
    Returns:
        float: The negative log-likelihood
        3D array: Gradients with respect to the
                    unnormalized input actications
    """
    alphas, ll_forward = forward_pass(log_probs, labels, blank)
    betas, ll_backward = backward_pass(log_probs, labels, blank)
    grads = compute_gradient(log_probs, alphas, betas, labels, blank)
    return -ll_forward, grads

def transduce_batch(log_probs, labels, blank=0):
    grads = np.zeros_like(log_probs)
    costs = []
    for b in range(log_probs.shape[0]):
        ll, g = transduce(log_probs[b, ...], labels[b], blank)
        grads[b, ...] = g
        costs.append(ll)
    return costs, grads

def test():
    blank = 0
    vocab_size = 10
    input_len = 8
    output_len = 3
    inputs = np.random.rand(input_len, output_len + 1, vocab_size)
    labels = np.random.randint(1, vocab_size, output_len)

    log_probs = log_softmax(inputs, axis=2)
    alphas, ll_forward = forward_pass(log_probs, labels, blank)
    betas, ll_backward = backward_pass(log_probs, labels, blank)

    assert np.allclose(ll_forward, ll_backward,
                       atol=1e-12, rtol=1e-12), \
      "Loglikelihood from forward and backward pass mismatch."

    grads = compute_gradient(log_probs, alphas, betas, labels, blank)
    neg_loglike = -ll_forward
    num_grads = numerical_gradient(log_probs, labels, neg_loglike, blank)
    assert np.allclose(grads, num_grads,
                       atol=1e-6, rtol=1e-6), \
            "Gradient / numerical gradient mismatch."

def numerical_gradient(log_probs, labels, neg_loglike, blank):
    epsilon = 1e-5
    T, U, V = log_probs.shape
    grads = np.zeros(log_probs.shape)
    for t in range(T):
        for u in range(U):
            for v in range(V):
                log_probs[t, u, v] += epsilon
                alphas, ll_forward = forward_pass(log_probs, labels, blank)
                grads[t, u, v] = (-ll_forward - neg_loglike) / epsilon
                log_probs[t, u, v] -= epsilon
    return grads

def small_test():
    acts = np.array([[[0.1, 0.6, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.6, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.8, 0.1]],
                     [[0.1, 0.6, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.1, 0.1],
                      [0.7, 0.1, 0.2, 0.1, 0.1]]])
    labels = [1, 2]
    blank = 0
    log_probs = log_softmax(acts, axis=2)
    ll, grads = transduce(log_probs, labels, blank)

def big_test():

    blank = 0
    acts = [
        [[[0.06535690384862791, 0.7875301411923206, 0.08159176605666074],
          [0.5297155426466327, 0.7506749639230854, 0.7541348379087998],
          [0.6097641124736383, 0.8681404965673826, 0.6225318186056529]],

         [[0.6685222872103057, 0.8580392805336061, 0.16453892311765583],
          [0.989779515236694, 0.944298460961015, 0.6031678586829663],
          [0.9467833543605416, 0.666202507295747, 0.28688179752461884]],

         [[0.09418426230195986, 0.3666735970751962, 0.736168049462793],
          [0.1666804425271342, 0.7141542198635192, 0.3993997272216727],
          [0.5359823524146038, 0.29182076440286386, 0.6126422611507932]],

         [[0.3242405528768486, 0.8007644367291621, 0.5241057606558068],
          [0.779194617063042, 0.18331417220174862, 0.113745182072432],
          [0.24022162381327106, 0.3394695622533106, 0.1341595066017014]]],


        [[[0.5055615569388828, 0.051597282072282646, 0.6402903936686337],
          [0.43073311517251, 0.8294731834714112, 0.1774668847323424],
          [0.3207001991262245, 0.04288308912457006, 0.30280282975568984]],

         [[0.6751777088333762, 0.569537369330242, 0.5584738347504452],
          [0.08313242153985256, 0.06016544344162322, 0.10795752845152584],
          [0.7486153608562472, 0.943918041459349, 0.4863558118797222]],

         [[0.4181986264486809, 0.6524078485043804, 0.024242983423721887],
          [0.13458171554507403, 0.3663418070512402, 0.2958297395361563],
          [0.9236695822497084, 0.6899291482654177, 0.7418981733448822]],

         [[0.25000547599982104, 0.6034295486281007, 0.9872887878887768],
          [0.5926057265215715, 0.8846724004467684, 0.5434495396894328],
          [0.6607698886038497, 0.3771277082495921, 0.3580209022231813]]]]
    labels = [[1, 2],
              [1, 1]]

    log_probs = log_softmax(acts, axis=3)
    costs, grads = transduce_batch(log_probs, labels, blank)

if __name__ == "__main__":
    test()
    small_test()
    big_test()
