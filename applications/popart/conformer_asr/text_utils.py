# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

_blank_symbol_ctc = '_'
_pad_symbol = '#'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ\' '

# Export all symbols:
symbols = [_blank_symbol_ctc] + list(_characters) + [_pad_symbol]

# Mappings from symbol to numeric ID and vice versa
# (remember - ID zero is reserved for blank symbol of CTC loss)
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text):
    """ converts text to sequence of numeric IDs """
    # putting space for punctuations
    space_id = _symbol_to_id[' ']
    return [_symbol_to_id.get(s, space_id) if s not in [_blank_symbol_ctc, _pad_symbol] else space_id
            for s in text]


def sequence_to_text(sequence, seq_length):
    """ converts numeric sequence to text """
    pad_id = _symbol_to_id[_pad_symbol]
    return "".join([_id_to_symbol[id] for id in sequence[0:seq_length] if id != pad_id])


def pad_text_sequence(text_sequence, max_text_sequence_length):
    """ pad numeric text sequence if required """
    pad = max_text_sequence_length - len(text_sequence)
    if pad <= 0:
        return text_sequence[0:max_text_sequence_length]
    return text_sequence + [_symbol_to_id[_pad_symbol]] * pad
