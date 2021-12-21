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


class Conformer(torch.nn.Module):
    def __init__(self, normalizer, encoder, decoder, loss_fn):
        super().__init__()
        self.normalize = normalizer
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, feature, feature_length, target_in, target_out, target_length):
        '''
        Args:
            speech: (Batch, Length, FeatureSize)
            speech_lengths: (Batch, )
            text_in: (Batch, Length)
            text_out: (Batch, Length)
            text_length: (Batch,)
        '''
        feature, feature_length = self.normalize(feature, feature_length)
        feature, feature_length = self.encoder(feature, feature_length)
        logits, target_mask = self.decoder(
            hs_pad=feature,
            hlens=feature_length,
            ys_in_pad=target_in,
            ys_in_lens=target_length,
        )
        loss = self.loss_fn(logits, target_out, target_mask)

        outputs = torch.argmax(logits, dim=-1)
        return loss, outputs
