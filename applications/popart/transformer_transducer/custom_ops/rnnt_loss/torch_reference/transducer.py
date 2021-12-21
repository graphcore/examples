from __future__ import division

import torch
from torch.autograd import Function
import transducer_cpp


class Transducer(Function):
    def __init__(self, blank_label=None):
        """
        Constructor for Transducer cost.

        Arguments:
            blank_label (optional) (Int): Integer representing the index
                of the blank, defaults to `alphabet_size - 1`.
        """
        super(Transducer, self).__init__()
        self.blank_label = blank_label

    @staticmethod
    def forward(ctx, log_probs, labels, lengths, label_lengths):
        """
        Computes the Transducer cost for a minibatch of examples.

        Arguments:
            log_probs (FloatTensor): The log probabilities should
                be of shape
                (minibatch, input len, output len, vocab size).
            labels (IntTensor): 1D tensor of labels for each example
                consecutively.
            lengths (IntTensor): 1D tensor of number actviation time-steps
                for each example.
            label_lengths (IntTensor): 1D tensor of label lengths for
                each example.

        Returns:
            costs (FloatTensor): .
        """
        is_cuda = log_probs.is_cuda

        certify_inputs(log_probs, labels, lengths, label_lengths)

        log_probs = log_probs.cpu()
        costs = torch.zeros(log_probs.shape[0])
        grads = log_probs.new(log_probs.shape).zero_()

        blank_label = 0  # self.blank_label
        if blank_label is None:
            blank_label = log_probs.shape[-1] - 1

        transducer_cpp.transduce(
            log_probs, labels, lengths, label_lengths, costs, grads, blank_label
        )
        if is_cuda:
            costs = costs.cuda()
            grads = grads.cuda()
        ctx.save_for_backward(grads)

        return costs

    @staticmethod
    def backward(ctx, cost):
        return ctx.saved_tensors[0], None, None, None


class TransducerLoss(Transducer):
    def __init__(self, size_average=True, blank_label=None):
        super(TransducerLoss, self).__init__(blank_label)
        self.size_average = size_average

    def forward(self, *args):
        parent = super(TransducerLoss, self)
        costs = parent.forward(*args)
        cost = torch.sum(costs)
        if self.size_average:
            cost = cost / costs.shape[0]
        return costs.new((cost,))

    def backward(self, *args):
        parent = super(TransducerLoss, self)
        grads = parent.backward(*args)[0]
        if self.size_average:
            grads = grads / grads.shape[0]
        return grads, None, None, None


def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def certify_inputs(log_probs, labels, lengths, label_lengths):
    check_type(log_probs, torch.float32, "log_probs")
    check_type(labels, torch.int32, "labels")
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a label length per example.")

    check_dim(log_probs, 4, "log_probs")
    check_dim(labels, 1, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
