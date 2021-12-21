# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Vocab:
    def __init__(self, use_generate, vocab_path, blank='<blank>', pad='<blank>', unk="<unk>", sos='<sos/eos>', eos='<sos/eos>'):
        '''
        Args:
            use_generate(bool): whether use generated data mode
            vocab_path(str): vocab path
            blank(str): blank token string
            pad(str): pad token string, there is blank
            unk(str): unknown token string
            sos(str): sos token string
            eos(str): eos token string
        '''
        self._blank = blank
        self._pad = pad
        self._unk = unk
        self._sos = sos
        self._eos = eos
        if use_generate:
            self.vocab_list = ['<blank>', '<unk>', '的', '一', '在', '十', '中', '是', '人', '有', '俭', '靛', '脍', '<sos/eos>']
        else:
            with open(vocab_path, 'r') as reader:
                self.vocab_list = [i.strip() for i in reader.readlines()]
        self.map_token_id = {token: index for index, token in enumerate(self.vocab_list)}
        self.map_id_token = {index: token for index, token in enumerate(self.vocab_list)}

    def tokenize(self, str):
        return list(str)

    def token2id(self, token):
        return self.map_token_id[token]

    def id2token(self, id):
        return self.map_id_token[id]

    def str2id(self, str):
        return [self.map_token_id.get(i, self.unk_id) for i in self.tokenize(str)]

    @property
    def sos_id(self):
        return self.map_token_id[self._sos]

    @property
    def eos_id(self):
        return self.map_token_id[self._eos]

    @property
    def unk_id(self):
        return self.map_token_id[self._unk]

    @property
    def blank_id(self):
        return self.map_token_id[self._blank]

    @property
    def pad_id(self):
        return self.map_token_id[self._pad]
