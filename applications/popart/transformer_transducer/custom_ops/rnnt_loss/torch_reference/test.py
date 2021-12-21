"""
Tests for the C implementation of the sequence transducer.

From outside the package directory, run
`python -m transducer.test.`
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch
import torch.nn as nn

from transducer import Transducer, TransducerLoss

def wrap_and_call(fn, acts, labels):
    acts = torch.tensor(acts.astype(np.float32), requires_grad=True)
    if use_cuda:
        acts = acts.cuda()

    lengths = [acts.shape[1]] * acts.shape[0]
    print("Lengths", lengths)
    label_lengths = [len(l) for l in labels]
    print("label lengths", label_lengths)
    labels = [l for label in labels for l in label]
    print("labels", labels)
    labels = torch.IntTensor(labels)
    lengths = torch.IntTensor(lengths)
    label_lengths = torch.IntTensor(label_lengths)

    log_probs = nn.functional.log_softmax(acts, dim=3)
    print("log_probs", log_probs)
    def grad_hook(grad):
        log_probs.saved_grad = grad.clone()
    log_probs.register_hook(grad_hook)

    costs = fn.apply(log_probs, labels, lengths, label_lengths)
    cost = torch.sum(costs)
    cost.backward()
    grads = log_probs.saved_grad
    if use_cuda:
        costs = costs.cpu()
        grads = grads.cpu()

    return costs.data.numpy(), grads.data.numpy()


def small_test():
    acts = np.array([[[0.1, 0.6, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.6, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.8, 0.1]],
                     [[0.1, 0.6, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.1, 0.1],
                      [0.7, 0.1, 0.2, 0.1, 0.1]]])
    labels = [[1, 2]]
    print("Acts.shape", acts.shape)
    acts = acts[None, ...]
    print("Acts.shape", acts.shape)

    tfn = Transducer(blank_label=0)
    cost, grads = wrap_and_call(tfn, acts, labels)
    expected_cost = 4.495666
    expected_grads = np.array([[[-0.308198071906, -0.6918019280939998, 0.0, 0.0, 0.0],
                                [-0.308198071906, 0.0, -0.3836038561880001, 0.0, 0.0],
                                [-0.3836038561880001, 0.0, 0.0, 0.0, 0.0]],
                               [[0.0, -0.308198071906, 0.0, 0.0, 0.0],
                                [0.0, 0.0, -0.6163961438119995, 0.0, 0.0],
                                [-0.9999999999999991, 0.0, 0.0, 0.0, 0.0]]])
    assert np.allclose(cost, expected_cost, rtol=1e-6), \
        "small_test costs mismatch."
    assert np.allclose(grads, expected_grads), \
        "small_test gradient mismatch."

def big_test():

    # minibatch x T x U x alphabet_size
    activations = [
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
    print("Acts2", len(activations), len(activations[0]), len(activations[0][0]), len(activations[0][0][0]))

    expected_costs = [4.2806528590890736, 3.9384369822503591]
    expected_grads = [
            [[[-0.4322264564338117, -0.5677735435661883, 0.0],
              [-0.36565009313836844, 0.0, -0.20212345042782007],
              [-0.20212345042782007, 0.0, 0.0]],

             [[-0.16521672442463506, -0.2670097320091765, 0.0],
              [-0.3943653886107811, 0.0, -0.2382944365367636],
              [-0.44041788696458367, 0.0, 0.0]],

             [[-0.052129794015740985, -0.11308693040889405, 0.0],
              [-0.18313786985332664, 0.0, -0.3243144491663483],
              [-0.7647323361309323, 0.0, 0.0]],

             [[0.0, -0.052129794015740985, 0.0],
              [0.0, 0.0, -0.23526766386906767],
              [-1.0, 0.0, 0.0]]],

            [[[-0.7161424128232795, -0.2838575871767207, 0.0],
              [-0.18382932237365335, -0.10002826480306751, 0.0],
              [-0.10002826480306751, 0.0, 0.0]],

             [[-0.41121794618117213, -0.3049244666421072, 0.0],
              [-0.3295759402552584, -0.15917784876050195, 0.0],
              [-0.2592061135635692, 0.0, 0.0]],

             [[-0.11607642141651396, -0.29514152476465827, 0.0],
              [-0.2865333615432337, -0.3381841034766833, 0.0],
              [-0.5973902170402529, 0.0, 0.0]],

             [[0.0, -0.11607642141651396, 0.0],
              [0.0, -0.4026097829597475, 0.0],
              [-1.0, 0.0, 0.0]]]]

    activations = np.array(activations)
    labels = [[1, 2],
              [1, 1]]

    tfn = Transducer(blank_label=0)
    costs, grads = wrap_and_call(tfn, activations, labels)

    assert np.allclose(costs, expected_costs), \
        "big_test average costs mismatch."

    assert np.allclose(grads, expected_grads), \
        "big_test grads for average cost mismatch."

def time_test():
    blank = 0
    batch_size = 32
    vocab_size = 30
    input_len = 400
    output_len = 80
    acts = np.random.rand(batch_size, input_len, output_len + 1, vocab_size)
    labels = np.random.randint(1, vocab_size, (batch_size, output_len))

    acts = torch.FloatTensor(acts)
    lengths = [acts.shape[1]] * acts.shape[0]
    label_lengths = [len(l) for l in labels]
    labels = np.array([l for label in labels for l in label])
    labels = torch.IntTensor(labels)
    lengths = torch.IntTensor(lengths)
    label_lengths = torch.IntTensor(label_lengths)
    log_probs = nn.functional.log_softmax(acts, dim=3)

    start = time.time()
    iters = 10
    for _ in range(iters):
        tfn = Transducer(blank_label=0)
        costs = tfn.apply(log_probs, labels, lengths, label_lengths)
    end = time.time()

    print("Time per iteration: {:.3f}(s)".format((end-start)/iters))


if __name__ == "__main__":
    use_cuda = False
    small_test()
    big_test()
    print("CPU Tests passed!")
    if torch.cuda.is_available():
        use_cuda = True
        small_test()
        print("GPU Tests passed!")
    time_test()
