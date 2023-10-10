# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Callable, Iterable, List, Optional
import torch


def batch_inference(
    dataset: Iterable[torch.Tensor],
    next_token_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    sequence_length: int,
    eos_token_id: int,
    pad_token_id: int = 0,
    output_length: Optional[int] = None,
    micro_batch_size: int = 1,
) -> List[torch.Tensor]:
    """Runs inference with a fixed micro_batch_size and the generation loop on the host.
        Results are returned in the same order as the dataset.

    Args:
        dataset (Iterable[torch.Tensor]): data
        next_token_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Heuristic for batched next token generation.
                                Inputs of the callable are logits tensor for the whole sequence and an indices tensor that identify
                                the lengths of each batch. Returned tensor is the next token id for each batch.
                                Example: 3 batches
                                tok tok tok tok PAD PAD PAD
                                tok tok tok PAD PAD PAD PAD
                                tok tok tok tok tok tok PAD
                                the indices tensor should be [4, 3, 6]
        sequence_length (int): length of the sequences. Each batch has the same sequence length, but different padding.
        eos_token_id (int): end of text token
        pad_token_id (int, optional): token index used for padding. Defaults to 0.
        output_length: Optional[int] = Maximum number of tokens to generate.
        micro_batch_size (int, optional): size of batches. Defaults to 1.

    Returns:
        List[torch.Tensor]: new generated tokens for each batch, in the same order as the dataset.
    """
    # Create buffers
    enc_batch = torch.full((micro_batch_size, sequence_length), pad_token_id).long()
    enc_masks = torch.ones((micro_batch_size, sequence_length), dtype=torch.float16)
    dec_batch = torch.full((micro_batch_size, sequence_length), pad_token_id).long()
    dec_masks = torch.zeros((micro_batch_size, sequence_length), dtype=torch.float16)
    # batch_lens starts from one because initially we have the starting <pad> token
    batch_lens = torch.ones((micro_batch_size,)).long()
    batch_ids = torch.zeros((micro_batch_size,)).long()
    sample_id = 0
    samples = 0

    def should_stop(i, token):
        return token == eos_token_id or (output_length and batch_lens[i] >= output_length)

    # Initialise buffers before starting to generate
    dummy_data = (torch.zeros((sequence_length)).long(), torch.zeros((sequence_length), dtype=torch.float16))
    rampdown = False
    dl = iter(dataset)
    dummy_is = set()
    for i in range(micro_batch_size):
        try:
            data = next(dl)
        # If there is less than one batch worth of data we need to prepare with dummy data.
        except StopIteration as _:
            data = dummy_data
            rampdown = True
            samples = sample_id

        if rampdown:
            dummy_is.add(i)

        enc_input, enc_mask = data
        enc_batch[i, :].copy_(enc_input[:sequence_length])
        enc_masks[i, :].copy_(enc_mask[:sequence_length])
        # Initially the model can only attend to the first token which is the starting token (<pad>)
        dec_masks[i, 0] = 1
        batch_ids[i] = sample_id
        sample_id += 1

    # Used for index_put_
    axis_0 = torch.arange(0, micro_batch_size).long()

    results = {}
    while True:
        new_token = next_token_fn(enc_batch, enc_masks, dec_batch, dec_masks, batch_lens)
        # Update the decoder batch
        torch.index_put_(dec_batch, (axis_0, batch_lens), new_token)
        # Extend the decoder masks
        torch.index_put_(dec_masks, (axis_0, batch_lens), torch.ones((micro_batch_size,), dtype=torch.float16))
        batch_lens += 1
        for i, t in enumerate(new_token):
            # if a sample in the batch should be terminated (based on EOS or output length)
            # the generated text is put in the results and a new sample is drawn for generation
            if i not in dummy_is and should_stop(i, t):
                # We save the generated tokens, minus the initial <pad> token
                results[int(batch_ids[i])] = dec_batch[i, 1 : batch_lens[i]].clone()
                try:
                    data = next(dl)
                # When we run out of samples to generate, use dummy data.
                except StopIteration as _:
                    data = dummy_data
                    rampdown = True
                    # Update the number of samples processed only the first time
                    # we hit the StopIteration
                    if samples == 0:
                        samples = sample_id

                if rampdown:
                    dummy_is.add(i)
                    if len(dummy_is) == micro_batch_size:
                        # If the number of elements in the dataset is < than the micro batch size,
                        # then only return the valid samples
                        out = []
                        for i in range(samples):
                            if i in results:
                                out.append(results[i])
                        return out

                # Put the new sample in the batch
                enc_input, enc_mask = data
                enc_batch[i, :].copy_(enc_input[:sequence_length])
                enc_masks[i, :].copy_(enc_mask[:sequence_length])
                # Reset the decoder tokens and mask for this sample
                dec_batch[i, :] = pad_token_id
                dec_masks[i, :] = 0
                dec_masks[i, 0] = 1
                batch_ids[i] = sample_id
                sample_id += 1
                batch_lens[i] = 1


if __name__ == "__main__":
    import random

    dataset = [torch.randint(1, 100, (random.randint(4, 20),)) for _ in range(100)]

    batch_size = 8
    out_tokens = 9
    max_len = 40

    def random_next_token(*_, **__):
        return torch.randint(0, 10, (batch_size,)).long()

    results = batch_inference(
        dataset,
        random_next_token,
        max_len + out_tokens,
        eos_token_id=1000,
        pad_token_id=0,
        output_length=out_tokens,
        micro_batch_size=batch_size,
    )
    print(*results, sep="\n")
