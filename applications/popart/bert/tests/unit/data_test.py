# Copyright 2019 Graphcore Ltd.
import os
import tempfile
import numpy as np
import struct
import pytest
from functools import reduce
from itertools import chain

from bert_data.dataset import DataSet
from bert_data.pretraining_dataset import (
    BinaryDataLoader,
    SyntheticDataLoader,
    BertDataTransform,
    data_file_format as pretraining_format,
    data_ranges as pretraining_ranges
)
from bert_data.squad_dataset import (
    SquadDataLoader,
    generate_synthetic_features,
    no_drop_batches_per_step
)


def test_dataloader():
    sequence_length = 128
    mask_tokens = 20
    batch_size = 2

    ind = []
    pos = []
    lbl = []

    with tempfile.TemporaryDirectory() as pwd:
        input_path = os.path.join(pwd, "input.bin")
        with open(input_path, "wb") as f:
            for _ in range(batch_size):
                i = np.random.randint(0, sequence_length, (sequence_length))
                p = np.random.randint(0, sequence_length, (sequence_length))
                l = np.random.randint(0, sequence_length, (mask_tokens))

                line = reduce(lambda accl, i: accl + struct.pack('<I', i),
                              chain(i, p, l), b'')
                f.write(line)

                ind.append(i)
                pos.append(p)
                lbl.append(l)

        ind = np.stack(ind)
        pos = np.stack(pos)
        lbl = np.stack(lbl)

        dl = BinaryDataLoader([input_path],
                              [sequence_length, sequence_length, mask_tokens],
                              batch_size,
                              shuffle=False)
        dl_itr = iter(dl)

        ind_, pos_, lbl_ = next(dl_itr)

        assert(np.all(ind_ == ind))
        assert(np.all(pos_ == pos))
        assert(np.all(lbl_ == lbl))


def test_synthetic_pretraining():
    sequence_length = 128
    mask_tokens = 20
    batch_size = 2
    vocab_length = 4864

    sizes = pretraining_format(sequence_length, mask_tokens)
    ranges = pretraining_ranges(sequence_length, mask_tokens, vocab_length)

    dl = SyntheticDataLoader([],
                             sizes,
                             batch_size,
                             shuffle=False,
                             synthetic_ranges=ranges)

    assert(len(dl) == 1)

    dl_itr = iter(dl)

    for data, size, max_value in zip(next(dl_itr), sizes, ranges):
        assert(np.all(data < max_value))
        assert(data.shape == (batch_size, size))


def test_synthetic_squad():
    sequence_length = 128
    batch_size = 2
    vocab_length = 4864

    features = generate_synthetic_features(
        sequence_length, vocab_length, batch_size)

    dl = SquadDataLoader(
        features,
        batch_size=batch_size)

    assert(len(dl) == 1)

    sizes = [sequence_length, sequence_length, sequence_length, 1, 1, 1]
    ranges = [vocab_length, sequence_length + 1, 2, sequence_length + 1, sequence_length, sequence_length]

    dl_itr = iter(dl)

    for data, size, max_value in zip(next(dl_itr), sizes, ranges):
        assert(np.all(data < max_value))
        assert(data.shape == (batch_size, size))


def test_transform():
    sequence_length = 128
    mask_tokens = 20
    batch_size = 2
    vocab_length = 9728

    data = [
        # indicies
        np.random.randint(0, 2 * vocab_length, (batch_size, sequence_length)).astype(np.uint32),
        # position
        np.random.randint(0, sequence_length, (batch_size, sequence_length)).astype(np.uint32),
        # segments
        np.random.randint(0, 2, (batch_size, sequence_length)).astype(np.uint32),
        # masks
        np.random.randint(0, sequence_length, (batch_size)).astype(np.uint32),
        np.random.randint(0, sequence_length, (batch_size)).astype(np.uint32),
        # labels
        np.random.randint(0, 2 * vocab_length, (batch_size, mask_tokens)).astype(np.uint32),
        np.random.randint(0, 2, (batch_size)).astype(np.uint32)
    ]

    # Simulate the BinaryDataLoader
    def dl():
        while True:
            yield data

    class TestDataLoader(object):
        def __iter__(self):
            return dl()

    dt = BertDataTransform(TestDataLoader(), vocab_length, mask_tokens)
    dt_iter = iter(dt)
    ind, pos, seg, msk_1, msk_2, lbl, lbl_2 = next(dt_iter)
    assert(np.all(ind < vocab_length))
    OutOfBounds = data[0] > vocab_length
    assert(np.all(ind[OutOfBounds] == 100))

    assert(np.all(lbl < vocab_length))
    OutOfBounds = data[5] > vocab_length
    assert(np.all(lbl[OutOfBounds] == 0))


def test_dataset():
    sequence_length = 128
    mask_tokens = 20
    batch_size = 2
    vocab_length = 9728
    batches_per_step = 100
    replication_factor = 2
    accumulation_factor = 3
    samples_per_step = batches_per_step * batch_size * replication_factor * accumulation_factor

    data = [
        # indicies
        np.random.randint(0, 2 * vocab_length, (samples_per_step, sequence_length)).astype(np.uint32),
        # position
        np.random.randint(0, sequence_length, (samples_per_step, sequence_length)).astype(np.uint32),
        # masks
        np.random.randint(0, sequence_length, (samples_per_step,)).astype(np.uint32),
        np.random.randint(0, sequence_length, (samples_per_step,)).astype(np.uint32),
        # labels
        np.random.randint(0, 2 * vocab_length, (samples_per_step, mask_tokens)).astype(np.uint32),
        np.random.randint(0, 2, (samples_per_step,)).astype(np.uint32)
    ]

    # Simulate the BertDataTransform
    def dl():
        while True:
            yield data

    class TestDataLoader(object):
        def __len__(self):
            return 1

        def __iter__(self):
            return dl()

    ds = DataSet(TestDataLoader(),
                 [
                    ("indices", (sequence_length * batch_size,)),
                    ("positions", (sequence_length * batch_size,)),
                    ("msk_mask", (batch_size,)),
                    ("seq_mask", (batch_size,)),
                    ("labels", (mask_tokens * batch_size,)),
                    ("nsp_labels", (batch_size,))
                 ],
                 batches_per_step,
                 replication_factor,
                 accumulation_factor)

    ds_iter = iter(ds)
    sample = next(ds)

    assert(np.all(sample["indices"].shape == np.array(
        [batches_per_step, accumulation_factor, replication_factor, batch_size * sequence_length])))
    assert(np.all(sample["positions"].shape == np.array(
        [batches_per_step, accumulation_factor, replication_factor, batch_size * sequence_length])))
    assert(np.all(sample["labels"].shape == np.array(
        [batches_per_step, accumulation_factor, replication_factor, batch_size * mask_tokens])))

    assert(np.all(sample["indices"].flatten() == data[0].flatten()))
    assert(np.all(sample["positions"].flatten() == data[1].flatten()))
    assert(np.all(sample["labels"].flatten() == data[4].flatten()))


def test_no_drop_remainder():
    pipeline_stages = 14
    # Prime number dataset
    prime_dataset = 113
    bps = no_drop_batches_per_step(
        2,
        prime_dataset,
        1,
        1,
        1,
        pipeline_stages,
        True
    )
    assert prime_dataset == bps

    # gradient accumulation
    gradient_accumulation = 16
    bps = no_drop_batches_per_step(
        2,
        prime_dataset * gradient_accumulation,
        1,
        1,
        gradient_accumulation,
        pipeline_stages,
        True
    )
    assert prime_dataset == bps

    # batch_size
    batch_size = 2
    bps = no_drop_batches_per_step(
        2,
        prime_dataset * batch_size,
        batch_size,
        1,
        1,
        pipeline_stages,
        True
    )
    assert prime_dataset == bps

    # replication factor
    replication_factor = 2
    bps = no_drop_batches_per_step(
        2,
        prime_dataset * replication_factor,
        1,
        replication_factor,
        1,
        pipeline_stages,
        True
    )
    assert prime_dataset == bps

    bps = no_drop_batches_per_step(
        2,
        prime_dataset * replication_factor * batch_size * gradient_accumulation,
        batch_size,
        replication_factor,
        gradient_accumulation,
        pipeline_stages,
        True
    )
    assert prime_dataset == bps

    # Inference
    bps = no_drop_batches_per_step(
        6,
        3*7,
        1,
        1,
        1,
        3,
        False
    )
    assert 3 == bps

    # Failure
    with pytest.raises(RuntimeError) as excinfo:
        no_drop_batches_per_step(
            6,
            prime_dataset,
            2,
            2,
            2,
            3,
            True
        )
    assert "no_drop_remainder cannot adjust batches_per_step" in str(excinfo.value)
