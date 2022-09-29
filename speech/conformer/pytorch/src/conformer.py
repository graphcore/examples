# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch
import poptorch
from src.layers.ctc_loss import CtcLoss
from src.utils.mask import make_pad_mask_decoder
from src.utils.mask import mask_finished_preds
from src.utils.mask import mask_finished_scores
from src.utils.mask import subsequent_mask
from src.utils.common import IGNORE_ID
from src.utils.common import add_sos_eos
from src.utils.common import log_add
from src.utils.common import remove_duplicates_and_blank
from src.utils.ipu_pipeline import BasePipelineModel

from typing import List, Optional, Tuple
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


class Conformer(BasePipelineModel):

    def __init__(self, normalizer, encoder, decoder, loss_fn, args, dtype):
        super().__init__()
        self.normalize = normalizer
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.args = args
        vocab_size = self.args['decoder']['vocab_size']
        output_size = self.args['encoder']['output_size']
        self.out = torch.nn.Linear(output_size, vocab_size)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.ignore_id = IGNORE_ID
        self.ctc_loss_fn = CtcLoss()
        self.ctc_weight = self.args['loss_weight']['ctc_weight']
        self.dtype = dtype


    def forward(self, feature, feature_length, target_in, target_out, target_length):
        '''
        Args:
            feature: (Batch, Length, FeatureSize)
            feature_length: (Batch, )
            target_in: (Batch, Length)
            target_out: (Batch, Length)
            target_length: (Batch,)
        '''
        feature, feature_length = self.normalize(feature, feature_length)
        if self.dtype == torch.float16: 
            feature = feature.type(torch.float16)
        feature, feature_length, encoder_masks = self.encoder(feature, feature_length)
        feature_encoder = self.out(feature)
        log_probs = torch.nn.functional.log_softmax(feature_encoder, -1)
        logits, target_mask = self.decoder(
            hs_pad=feature,
            hlens=feature_length,
            ys_in_pad=target_in,
            ys_in_lens=target_length
        )
        loss = self.loss_fn(logits, target_out, target_mask)

        loss_ctc = self.ctc_loss_fn(log_probs.transpose(0, 1), target_out, feature_length, target_length, self.dtype)
        return poptorch.identity_loss((1-self.ctc_weight)*loss+self.ctc_weight*loss_ctc, reduction='mean'), loss, loss_ctc


    def _forward_encoder(self, feature, feature_length):
        '''
        Args:
            speech: (Batch, Length, FeatureSize)
            speech_lengths: (Batch, )
            text_in: (Batch, Length)
            text_out: (Batch, Length)
            text_length: (Batch,)
        '''
        feature, feature_length = self.normalize(feature, feature_length)
        feature, feature_length, encoder_masks = self.encoder(feature, feature_length)

        return feature, encoder_masks, feature_length


    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 3,
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask, feature_length = self._forward_encoder(
            speech, speech_lengths)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long).fill_(self.sos)  # (B*N, 1)

        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.repeat([batch_size]).unsqueeze(1)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen+1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1)  # (B*N, i, i)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = torch.div(best_k_index, beam_size, rounding_mode='floor')  # best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps, best_scores


    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
      ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
        Returns:
            List[List[int]]: best path result
        """
        '''assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0'''

        batch_size = speech.shape[0]
        encoder_out, encoder_mask, feature_length = self._forward_encoder(
            speech, speech_lengths)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        feature_encoder = self.out(encoder_out)
        ctc_probs = torch.nn.functional.log_softmax(feature_encoder, -1)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask_decoder(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores


    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:

        hyps, tmp, l, en = self._ctc_prefix_beam_search(speech, speech_lengths, beam_size, decoding_chunk_size, num_decoding_left_chunks, simulate_streaming)
        return hyps[0]

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
    ) -> Tuple[List[List[int]], torch.Tensor]:


        encoder_out, encoder_mask, feature_length = self._forward_encoder(speech, speech_lengths)  # (B, maxlen, encoder_dim)
        encoder_out = encoder_out[0]

        maxlen = encoder_out.size(1)
        feature_encoder = self.out(encoder_out)

        ctc_probs = torch.nn.functional.log_softmax(feature_encoder, -1)

        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out, speech_lengths


    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        ctc_weight: float = 0.5
    ) -> List[int]:

        device = speech.device
        hyps, encoder_out, feature_length = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size)
        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, 0)
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps], device=device, dtype=torch.long)
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        feature_length = feature_length.repeat(beam_size)
        decoder_out, target_mask = self.decoder(
            hs_pad=encoder_out,
            hlens=feature_length,
            ys_in_pad=hyps_pad,
            ys_in_lens=hyps_lens,
        )

        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.detach().numpy()

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[0][j][w]
            score += decoder_out[0][len(hyp[0])][self.eos]

            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0], best_score
