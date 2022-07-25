# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tests.unit_tester import BatchGenerator, Handler, Wrapper
from fastpitch.model import TemporalPredictor, FastPitch
from fastpitch.transformer import TransformerLayer
import torch
import loss_functions
from pathlib import Path
import pytest


class TestLayers:

    def test_temporal_predictor(self):
        pytorch_model = TemporalPredictor(input_size=384, filter_size=256, kernel_size=3, dropout=0)
        pytorch_model = Wrapper(pytorch_model, 0)
        batch_proto_list = [
            dict(column='feature', size=(2, 189, 384), data_type='float32'),
            dict(column='mask', size=(2, 189, 1), data_type='float32')
        ]
        batch = BatchGenerator().build_datas(batch_proto_list)
        print(batch)
        handler = Handler(batch, pytorch_model, atol=1e-4)
        handler.run_one_step()

    def test_transformer_layer(self):
        pytorch_model = TransformerLayer(n_head=1, d_model=192, d_head=32, d_inner=768, kernel_size=3, dropout=0, **{'dropatt': 0, 'pre_lnorm': False})
        pytorch_model = Wrapper(pytorch_model, 0)
        batch_proto_list = [
            dict(column='feature', size=(1, 189, 192), data_type='float32'),
            dict(column='mask', size=(1, 189, 1), data_type='float32')
        ]
        batch = BatchGenerator().build_datas(batch_proto_list)
        handler = Handler(batch, pytorch_model, atol=1e-4)
        handler.run_one_step()

    def test_model(self):
        model = FastPitch(
            n_mel_channels=80, max_seq_len=2048, n_symbols=148, padding_idx=0, symbols_embedding_dim=32, in_fft_n_layers=1, in_fft_n_heads=1,
            in_fft_d_head=64, in_fft_conv1d_kernel_size=3, in_fft_conv1d_filter_size=768, in_fft_output_size=32, p_in_fft_dropout=0, p_in_fft_dropatt=0,
            p_in_fft_dropemb=0, out_fft_n_layers=1, out_fft_n_heads=1, out_fft_d_head=64, out_fft_conv1d_kernel_size=3, out_fft_conv1d_filter_size=768,
            out_fft_output_size=32, p_out_fft_dropout=0, p_out_fft_dropatt=0, p_out_fft_dropemb=0, dur_predictor_kernel_size=3, dur_predictor_filter_size=64,
            p_dur_predictor_dropout=0, dur_predictor_n_layers=2, pitch_predictor_kernel_size=3, pitch_predictor_filter_size=64, p_pitch_predictor_dropout=0,
            pitch_predictor_n_layers=2, pitch_embedding_kernel_size=3, n_speakers=1, speaker_emb_weight=1)
        stats = {'mean': 218.5497281640811, 'std': 64.82629532867513}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.model.pitch_mean[0] = stats['mean']
                self.model.pitch_std[0] = stats['std']
                self.criterion = loss_functions.get_loss_function(
                    'FastPitch', dur_predictor_loss_scale=0.1,
                    pitch_predictor_loss_scale=0.1)

            def forward(self, batch):
                text_padded, mel_padded, dur_padded, pitch_padded, dur_lens, output_leng = batch
                y_pred = self.model([text_padded, mel_padded, dur_padded, pitch_padded], use_gt_durations=True)
                loss = self.criterion(y_pred, [mel_padded, dur_padded, dur_lens, pitch_padded])
                return loss

        model = Model()
        batch_proto_list = [
            dict(column='token', size=(1, 189), data_type='int', max_value=50),
            dict(column='feature', size=(1, 80, 870), data_type='float32'),
            dict(column='token', size=(1, 189), data_type='int', max_value=10),
            dict(column='feature', size=(1, 189), data_type='float32'),
            dict(column='length', size=[1], data_type='int', max_value=150),
            dict(column='length', size=[1], data_type='int', max_value=800),
        ]
        batch = BatchGenerator().build_datas(batch_proto_list)


        handler = Handler(batch, model, atol=1e-7)
        handler.run_one_step()

if __name__ == '__main__':
    app_path = str(Path(__file__).parent.parent.resolve())
    config_name = str(Path(app_path, 'tests', 'test_layers.py'))
    pytest.main(['-s', config_name])
