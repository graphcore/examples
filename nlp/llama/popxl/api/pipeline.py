# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Union, List, Optional


from scipy.special import softmax

import popxl
from inference import inference
from modelling.embedding import LlamaEmbeddingsTP
from modelling.hf_mapping import hf_mapping_lm_tp

from popxl_addons import timer
from popxl_addons.array_munging import tensor_parallel_input, repeat

from config import LlamaConfig
from textwrap import dedent

from transformers.models.llama import LlamaForCausalLM
from transformers import AutoTokenizer

from tqdm import tqdm
import logging
import popart
import numpy as np
import time

# Prompt format code from https://huggingface.co/blog/llama2#how-to-prompt-llama-2
# Licensed under Llama 2 Community License for distribution: https://huggingface.co/meta-llama/Llama-2-7b/blob/main/LICENSE.txt


def default_llama_prompt():
    prompt_template = """<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{prompt} [/INST]""".format(
        prompt="{prompt}"
    )

    return prompt_template


def format_prompts(prompt: Union[str, List[str]], format_str: str):
    if isinstance(prompt, str):
        prompt = [prompt]

    # iterate over prompts and apply prompt template
    return [format_str.format(prompt=p) for p in prompt]


def tokenize_initial(prompt: List[str], tokenizer: AutoTokenizer, config: LlamaConfig):
    tokenizer.padding_side = "right"

    tokenizer_result = tokenizer(prompt, return_length=True)
    tokenized_prompt = tokenizer_result.input_ids

    # we want to obtain the real unpadded length from the tokenizer, hence we tokenize without padding, then pad later.
    tokenized_length = np.asarray(tokenizer_result.length, dtype=np.int32)

    padded_prompt = np.full(
        (
            len(prompt),
            config.model.sequence_length,
        ),
        tokenizer.pad_token_id,
        dtype=np.int32,
    )

    # length can vary, hence we iterate over each prompt.
    for i in range(len(prompt)):
        padded_prompt[i, : tokenized_length[i]] = tokenized_prompt[i]

    return padded_prompt, tokenized_prompt, tokenized_length


class LlamaPipeline:
    def __init__(
        self,
        config: LlamaConfig,
        *args,
        hf_llama_checkpoint: Union[str, LlamaForCausalLM] = "meta-llama/Llama-2-7b-chat-hf",
        sequence_length: int = None,
        micro_batch_size: int = None,
        tokenizer: Optional[AutoTokenizer] = None,
        prompt_format: Optional[str] = None,
        **kwargs,
    ) -> None:

        # Setup for model
        if sequence_length is not None:
            config.model.sequence_length = sequence_length
        if micro_batch_size is not None:
            config.execution.micro_batch_size = micro_batch_size

        logging.info(f"Creating session")
        session: popxl.Session = inference(config)

        if isinstance(hf_llama_checkpoint, str):
            logging.info(f"Downloading '{hf_llama_checkpoint}' pretrained weights")
            hf_model = LlamaForCausalLM.from_pretrained(hf_llama_checkpoint)
            logging.info("Completed pretrained weights download.")

            if tokenizer is None:
                logging.info(f"Downloading '{hf_llama_checkpoint}' tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(hf_llama_checkpoint, use_fast=False, add_eos_token=True)
                logging.info("Completed tokenizer download.")

        else:
            logging.info("hf_model already specified, skipping download.")
            hf_model = hf_llama_checkpoint

        if tokenizer is None:
            raise ValueError(
                "A tokenizer needs to be passed to the pipeline if a custom checkpoint is being provided."
                "Use: AutoTokenizer.from_pretrained(model-name) to create the tokenizer."
            )

        with timer("Loading HF pretrained model to IPU"):
            weights = hf_mapping_lm_tp(config, session, hf_model)
            session.write_variables_data(weights)

        logging.info("IPU pretrained weights loading complete.")

        self.prompt_format = prompt_format
        if self.prompt_format is None:
            self.prompt_format = default_llama_prompt()

        tokenizer.pad_token_id = 0

        self.original_eos_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.pretrained = hf_model
        self.config = config
        self.session = session
        self.decoded_result = None
        self.last_instruction_prompt = None
        self.use_kv_cache = config.execution.use_cache

        logging.info("Finished initialising pipeline")

    def next_token(self, inputs, lengths, temperature=None, k=None, N=None, prompt_mode=False):
        shards = self.config.execution.tensor_parallel * self.config.execution.data_parallel

        # (bs, tp, slen)
        parallel_inputs = tensor_parallel_input(
            inputs, shards, shards, lambda t, i: LlamaEmbeddingsTP.offset_input(t, i, self.config)
        )

        if self.use_kv_cache:
            if len(parallel_inputs.shape) == 1:
                parallel_inputs = np.expand_dims(parallel_inputs, axis=0)
            parallel_inputs = parallel_inputs.transpose(1, 0)
        else:
            if inputs.shape[0] == 1:
                parallel_inputs = np.expand_dims(parallel_inputs, axis=0)
            parallel_inputs = parallel_inputs.transpose(1, 0, 2)

        next_token_logits = self.session.run(
            {
                self.session.inputs.words: parallel_inputs,
                self.session.inputs.last_token_indices: repeat(np.array(lengths - 1), shards, axis=0),
            }
        )

        # Used when caching the KV values for the prompt - skip search for next token
        if prompt_mode:
            return

        # Extract logits and perform topk search when generating new tokens
        next_token_logits = next_token_logits[self.session.outputs.next_token_logits][
            0
        ]  # extract 0th replica as all are identical
        next_token_logits = next_token_logits[:N]

        if k:
            topk_shape = (next_token_logits.shape[0], k)
            topk_logits = np.empty(topk_shape)
            topk_idx = np.empty(topk_shape, dtype=np.int32)
            for i in range(next_token_logits.shape[0]):
                topk_idx[i] = np.argpartition(next_token_logits[i], -k)[-k:]
                topk_logits[i] = next_token_logits[i, topk_idx[i]]
            next_token_logits = topk_logits

        assert temperature is not None, "Temperature value not passed to pipeline."
        assert k is not None, "top-k `k` value not passed to pipeline."
        assert temperature >= 0.0, "Temperature must be at least 0."
        if temperature > 0:
            next_token_prob = softmax(next_token_logits.astype(np.float32) / temperature, axis=-1)
            next_token_id = np.asarray(
                [
                    np.random.choice(next_token_logits.shape[-1], p=next_token_prob[i])
                    for i in range(next_token_prob.shape[0])
                ]
            )
        else:
            # mathematically equivalent to temperature = 0
            next_token_id = next_token_logits.argmax(axis=-1)

        if k:
            # retrieve real token ids from top_k subset.
            next_token_id = topk_idx[range(next_token_logits.shape[0]), next_token_id]

        next_token_id = np.concatenate(
            (
                next_token_id,
                np.asarray(
                    [self.tokenizer.eos_token_id] * (self.config.execution.micro_batch_size - N), dtype=np.int32
                ),
            )
        )
        return next_token_id

    """
        Run Llama 2.0 inference loop on a `str` prompt, or a list of prompts.

        prompt: Union[str, List[str]], prompt or list of prompts to run inference on.
        temperature: float, control sampling temperature by dividing logits by this value. For temperature = 0 where argmax sampling is used instead
        k: int, limits random sampling to top `k` most probably tokens. For `k=0` equivalent to `k=vocab_size`.
        output_length: Optional[int], maximum number of tokens to sample. Cannot exceed `sequence_length - output_length`. Defaults to maximum possible value.
        prompt_format: Optional[str], if specified override prompt format specified during pipeline init.
        end_key: Optional[str], if specified override end key specified during pipeline init.
        print_live: Optional[bool], whether to print the tokens one-by-one as they are decoded. `None` results in automatic behaviour depending on batch size.
        print_final: bool, whether to print the total time taken and throughput.
    """

    def __call__(
        self,
        prompt: Union[str, List[str]],
        *args,
        temperature: float = 1.0,
        k: int = 5,
        output_length: Optional[int] = None,
        prompt_format: Optional[str] = None,
        print_live: Optional[bool] = None,
        print_final: bool = True,
    ):
        assert 0.0 <= temperature, "Temperature must be at least 0"
        assert (
            0 <= k <= self.config.model.embedding.vocab_size
        ), f"top k value must be in the range [0, vocab_size] (maximum = {self.config.model.embedding.vocab_size})"
        original_prompt = prompt if isinstance(prompt, list) else [prompt]

        prompt_format = prompt_format if prompt_format is not None else self.prompt_format
        prompt = format_prompts(prompt, prompt_format)
        N = len(prompt)  # record original number of prompts so we can remove padding later

        # if print_live is not provided, set to True for single example case, else False. If it is provided set to that.
        print_live = len(prompt) == 1 if print_live is None else print_live

        # Preprocess the data including batching it
        micro_batch = self.config.execution.micro_batch_size

        assert (
            len(prompt) <= micro_batch
        ), f"Number of prompts greater than session batch size! Got {len(prompt)} but expected no more than {self.config.execution.micro_batch_size}"

        # Create a mask to show when a specific batch entry has finished sampling.
        # Padding elements begin already complete.
        complete_mask = np.asarray([False] * len(prompt) + [True] * (micro_batch - len(prompt)), dtype=bool)

        # Apply padding to batch.
        prompt = prompt + [""] * (micro_batch - len(prompt))

        logging.info("Attach to IPUs")
        self.session.__enter__()
        logging.info("Start inference")

        padded_prompt, tokenized_prompt, tokenized_length = tokenize_initial(prompt, self.tokenizer, self.config)

        self.last_instruction_prompt = prompt
        num_generated = 0
        result = [[] for _ in range(len(prompt))]

        if output_length is None:
            output_length = self.config.model.sequence_length - max(tokenized_length)

        assert 1 <= output_length <= self.config.model.sequence_length - max(tokenized_length)

        # Llama uses Sentencepiece based tokenizers which have whitespace decoding issues when doing a
        # word-by-word live print: https://github.com/huggingface/transformers/issues/22710
        # Decode full sequence and print difference in sequence using previous length.
        if print_live:
            logging.info(f"Input prompt: {original_prompt[0]}")
            logging.info("Response:")

            # Store initial length of decoded sequence before new tokens are added
            prev_decoded_len = len(self.tokenizer.decode(padded_prompt[0], skip_special_tokens=True))

        latencies = 0
        # Run the prompt tokens through to generate the cache - last prompt token generates next token
        if self.use_kv_cache:
            logging.info("Processing prompt...")
            stopping_max = max(tokenized_length)

            for n in tqdm(range(stopping_max)):
                if n == stopping_max - 1:
                    # The index of the last token is the length of the tokenized sequence - 1
                    # This token is used to generate the first token in the generation phase
                    next_tokens = padded_prompt[range(micro_batch), tokenized_length - 1]
                else:
                    self.next_token(padded_prompt[:, n], n + 1, prompt_mode=True)

        start_time = time.perf_counter()
        for _ in range(output_length):
            # For caching, only pass the last token that was generated
            prev_tokens = next_tokens if self.use_kv_cache else padded_prompt

            lat_st = time.perf_counter()
            next_tokens = self.next_token(prev_tokens, tokenized_length, temperature, k, N)
            latencies += time.perf_counter() - lat_st

            # update mask based on whether EOS was sampled and whether maximum length was exceeded
            complete_mask = complete_mask | (next_tokens == self.tokenizer.eos_token_id)
            complete_mask = complete_mask | (tokenized_length >= self.config.model.sequence_length)

            if complete_mask.all():
                break

            for i, t in enumerate(next_tokens):
                if complete_mask[i]:
                    continue
                result[i].append(t)

            # update final elements in each batch element with next token
            padded_prompt[range(len(prompt)), tokenized_length] = next_tokens

            tokenized_length[~complete_mask] += 1  # update length by one for elements that are not complete
            num_generated += 1

            # Print change in decoded sequence instead of decoding single tokens
            if print_live and not complete_mask[0]:
                # Decode full sequence with new token added
                live_updated = self.tokenizer.decode(padded_prompt[0], skip_special_tokens=True)
                # Get 'change' in decoded sequence
                live_new = live_updated[prev_decoded_len:]
                # Update length of current sequence for next iteration
                prev_decoded_len = len(live_updated)
                print(live_new, end="", flush=True)

        end_time = time.perf_counter()

        self.decoded_result = self.tokenizer.batch_decode(result)[:N]  # unpad result

        print("")
        if print_final:
            logging.info(f"Output in {end_time - start_time:.2f} seconds")
            logging.info(f"Total throughput: {num_generated / (end_time - start_time):.2f} t/s")
            logging.info(f"Batch (size {micro_batch}) latency: {latencies / (num_generated // len(prompt))}s.")

        return self.decoded_result

    def detach(self):
        was_attached_or_device = self.session._was_attached_stack.pop()

        self.session._device.detach()
        self.session._pb_session.setEngineIsLoaded(False)

        # If a DeviceInfo was stored in the stack then restore it.
        if isinstance(was_attached_or_device, popart.DeviceInfo):
            self.session._set_device(was_attached_or_device)


if __name__ == "__main__":
    from utils.setup import llama_config_setup

    config, _, hf_model = llama_config_setup("config/inference.yml", "release", "llama2_7b_pod4", hf_model_setup=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    pipe = LlamaPipeline(config, hf_llama_checkpoint=hf_model, tokenizer=tokenizer)

    print(
        pipe(
            [
                "Make up a question for a school physics exam. Please be serious.",
            ],
            k=5,
            temperature=0.7,
        )
    )
