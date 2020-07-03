# Copyright 2020 Graphcore Ltd.
import numpy as np
from random import random
import re
import nltk

_pad = '_'
_eos = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ!,-.:;? '

_arpabet_symbols = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
    'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
    'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

_arpabet_dict = nltk.corpus.cmudict.dict()

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + ['@' + s for s in _arpabet_symbols]

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_braces_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def try_pronunciation(word, replace_prob):
    """ returns phonetic representation of word (probabilistically) """
    try:
        # arpabet dict keys are all lower case
        phonemes = _arpabet_dict[word.lower()][0]
        phonemes = " ".join(['@' + pho for pho in phonemes])
    except KeyError:
        return word

    return '{%s}' % phonemes if random() < replace_prob else word


def replace_with_pronunciation(text, replace_prob):
    """ randomly replaces words in text with pronunciation """
    text = ' '.join(try_pronunciation(word, replace_prob) for word in text.split(' '))
    return text


def text_to_sequence(text, replace_prob=0.5):
    """ Converts a string of text to a sequence of IDs corresponding to the symbols in the text """

    text = text.upper()

    if replace_prob > 0.0:
        text = replace_with_pronunciation(text, replace_prob)

    sequence = []
    while len(text):
        match_result = _curly_braces_re.match(text)
        if not match_result:
            # no curly-brace match
            sequence += _symbols_to_sequence(text)
            break
        sequence += _symbols_to_sequence(match_result.group(1))
        # split into list of phonemes before converting to sequence
        sequence += _symbols_to_sequence(match_result.group(2).split(' '))
        text = match_result.group(3)

    # end with EOS symbol
    sequence.append(_symbol_to_id[_eos])
    return sequence


def _symbols_to_sequence(symbols):
    """ converts sequence of symbols to sequence of IDs """
    # putting space for unknown punctuations
    space_id = _symbol_to_id[' ']
    return [_symbol_to_id.get(s, space_id) for s in symbols]


def pad_text_sequence(text_sequence, max_text_sequence_length):
    """ pad text sequence if required """
    pad = max_text_sequence_length - len(text_sequence)
    if pad <= 0:
        return text_sequence[0:max_text_sequence_length]
    return text_sequence + [_symbol_to_id[_pad]] * pad


def convert_numpy_text_array_to_numpy_sequence_array(text_array, max_text_sequence_length):
    """ converts an numpy array of strings to a numpy array of text sequences """
    batch_shape = list(text_array.shape)
    sequences = []
    for text in text_array.flat:
        sequences.append(pad_text_sequence(text_to_sequence(text), max_text_sequence_length))
    out_array = np.array(sequences)
    out_array = out_array.reshape(batch_shape + [max_text_sequence_length]).astype('int32')
    return out_array
