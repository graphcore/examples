# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import torch.nn.functional as F

import logging_util

from rnnt_reference.model import label_collate

# set up logging
logger = logging_util.get_basic_logger('TransducerGreedyDecoder')


class TransducerGreedyDecoder:
    """A greedy transducer decoder.

    Args:
        blank_idx: inded of blank symbol
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
    """
    def __init__(self, blank_idx, max_symbols_per_step=30, max_symbol_per_sample=None, shift_labels_by_one=True):
        self.blank_idx = blank_idx
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step
        assert max_symbol_per_sample is None or max_symbol_per_sample > 0
        self.max_symbol_per_sample = max_symbol_per_sample
        self._SOS = -1   # start of sequence
        self.shift_labels_by_one = shift_labels_by_one

    def _pred_step(self, model, label, hidden, device):
        if label == self._SOS:
            return model.predict(None, hidden, add_sos=False)

        label = label_collate([[label]]).to(device)
        return model.predict(label, hidden, add_sos=False)

    def _joint_step(self, model, enc, pred, log_normalize=False):
        # logits = model.joint(enc, pred)[:, 0, 0, :]
        # Follows model.joint() logic, but with transcription FC  layer excluded
        pred = model.joint_pred(pred)

        enc = enc.unsqueeze(dim=2)   # (B, T, 1, H)
        pred = pred.unsqueeze(dim=1)   # (B, 1, U + 1, H)

        logits = model.joint_net(enc + pred)
        logits = logits[:, 0, 0, :]

        del enc, pred

        if log_normalize:
            probs = F.log_softmax(logits, dim=len(logits.shape) - 1)
            return probs
        else:
            return logits

    def decode(self, model, x, out_lens):
        """Returns a list of sentences given an input batch.

        Args:
            x: Output of the transcription network (followed by joint-transcription-fc).
            out_lens: list of int representing the length of each output sequence

        Returns:
            list containing batch number of sentences (strings).
        """
        model = getattr(model, 'module', model)
        with torch.no_grad():

            # logits, out_lens = model.encode(x, out_lens)
            # Excluding transcription network
            logits = x

            output = []
            for batch_idx in range(logits.size(0)):
                inseq = logits[batch_idx, :, :].unsqueeze(1)
                logitlen = out_lens[batch_idx]
                sentence = self._greedy_decode(model, inseq, logitlen)
                output.append(sentence)

        return output

    def _greedy_decode(self, model, x, out_len):
        device = x.device

        hidden = None
        label = []
        for time_idx in range(out_len):
            if self.max_symbol_per_sample is not None \
                    and len(label) > self.max_symbol_per_sample:
                break
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (
                    self.max_symbols is None or
                    symbols_added < self.max_symbols):
                g, hidden_prime = self._pred_step(
                    model,
                    self._SOS if label == [] else label[-1],
                    hidden,
                    device
                )

                joint = self._joint_step(model, f, g, log_normalize=False)
                logp = joint[0, :]


                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self.blank_idx:
                    not_blank = False
                else:
                    # This flag means we need to offset labels by - 1 when extracting labels from trained logits
                    # The reason for offset is that we treat logits "A" dimension as [<blank>, valid characters... A-1]
                    # Thus, blank-symbol has idx 0 and real symbols must have indices [1:A-1]
                    # RNN-T Loss uses labels as indices of logits (in A dimension)
                    # The opposite logic must be applied when logits are trained - see transducer_builder.py
                    if self.shift_labels_by_one:
                        k = k - 1
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        return label
