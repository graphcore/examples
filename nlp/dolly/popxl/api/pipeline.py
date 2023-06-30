# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Union, List, Optional, Any

import logging
from scipy.special import softmax

import popxl
from inference import inference
from modelling.embedding import DollyEmbeddingsTP
from modelling.hf_mapping import hf_mapping_lm_tp
from popxl_addons import timer
from popxl_addons.array_munging import tensor_parallel_input, repeat
from config import DollyConfig
from textwrap import dedent

from transformers.models.gpt_neox import GPTNeoXForCausalLM
from transformers import AutoTokenizer

import popxl
import popart
import numpy as np
import time

# Prompt format code from https://huggingface.co/databricks/dolly-v2-3b/blob/main/instruct_pipeline.py
# Licensed under Apache 2.0 see https://github.com/databrickslabs/dolly/blob/master/LICENSE
# Instruction finetuned models need a specific prompt
DEFAULT_END_KEY = "### End"


def default_dolly_prompt():
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = "The instruction below describes a task. Write a response that appropriately completes the request."

    # This is the prompt that is used for generating responses using an already-trained model. It ends with the response
    # key, where the job of the model is to provide the completion that follows it (which means the response itself).
    format_str = dedent(
        """\
    {intro}
    {instruction_key}
    {instruction}
    {response_key}
    """
    ).format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction="{instruction}",
        response_key=RESPONSE_KEY,
    )
    return format_str


def format_prompts(prompt: Union[str, List[str]], format_str: str):
    if isinstance(prompt, str):
        prompt = [prompt]

    # iterate over prompts and apply prompt template
    return [format_str.format(instruction=p) for p in prompt]


def tokenize_initial(prompt: List[str], tokenizer: AutoTokenizer, config: DollyConfig):
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


class DollyPipeline:
    def __init__(
        self,
        config: DollyConfig,
        *args,
        hf_dolly_checkpoint: Union[str, GPTNeoXForCausalLM] = "databricks/dolly-v2-12b",
        sequence_length: int = None,
        micro_batch_size: int = None,
        tokenizer: Optional[AutoTokenizer] = None,
        prompt_format: Optional[str] = None,
        end_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        if sequence_length is not None:
            config.model.sequence_length = sequence_length
        if micro_batch_size is not None:
            config.execution.micro_batch_size = micro_batch_size

        logging.info(f"Creating session")
        session: popxl.Session = inference(config)
        if isinstance(hf_dolly_checkpoint, str):
            logging.info(f"Downloading '{hf_dolly_checkpoint}' pretrained weights")
            hf_model = GPTNeoXForCausalLM.from_pretrained(hf_dolly_checkpoint)
            logging.info("Completed pretrained weights download.")
            if tokenizer is None:
                logging.info(f"Downloading '{hf_dolly_checkpoint}' tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(hf_dolly_checkpoint)
                logging.info("Completed tokenizer download.")
        else:
            logging.info("hf_model already specified, skipping download.")
            hf_model = hf_dolly_checkpoint
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
            self.prompt_format = default_dolly_prompt()

        self.end_key = end_key
        if self.end_key is None:
            self.end_key = DEFAULT_END_KEY

        self.original_eos_token_id = tokenizer.encode(self.end_key)
        assert len(self.original_eos_token_id) == 1
        self.original_eos_token_id = self.original_eos_token_id[0]
        tokenizer.eos_token_id = self.original_eos_token_id

        self.tokenizer = tokenizer
        self.pretrained = hf_model
        self.config = config
        self.session = session
        self.decoded_result = None
        self.last_instruction_prompt = None
        logging.info("Finished initialising pipeline")

    def next_token(self, inputs, lengths, temperature, k, N):
        shards = self.config.execution.tensor_parallel * self.config.execution.data_parallel

        parallel_inputs = tensor_parallel_input(
            inputs, shards, shards, lambda t, i: DollyEmbeddingsTP.offset_input(t, i, self.config)
        )
        # tensor_parallel_input will squeeze out the batch dim if len(inputs) == 1, so we must expand_dim again.
        if inputs.shape[0] == 1:
            parallel_inputs = np.expand_dims(parallel_inputs, axis=0)
        parallel_inputs = parallel_inputs.transpose(1, 0, 2)

        next_token_logits = self.session.run(
            {
                self.session.inputs.words: parallel_inputs,
                self.session.inputs.last_token_indices: repeat(np.array(lengths - 1), shards, axis=0),
            }
        )[self.session.outputs.next_token_logits][
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

        assert temperature >= 0.0, "Temperature must be at least 0."
        if temperature > 0:
            next_token_prob = softmax(next_token_logits.astype(np.float32) / temperature, axis=-1)
            next_token_id = np.asarray(
                [
                    np.random.choice(next_token_logits.shape[-1], p=next_token_prob[i])
                    for i in range(next_token_prob.shape[0])
                ]
            )
        else:  # mathematically equivalent to temperature = 0
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
        Run Dolly 2.0 inference loop on a `str` prompt, or a list of prompts.

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
        end_key: Optional[str] = None,
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

        if end_key is not None:
            eos_token_id = self.tokenizer.encode(end_key)
            assert len(eos_token_id) == 1
            self.tokenizer.eos_token_id = eos_token_id[0]

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

        padded_prompt, _, tokenized_length = tokenize_initial(prompt, self.tokenizer, self.config)
        self.last_instruction_prompt = prompt
        num_generated = 0
        result = [[] for _ in range(len(prompt))]

        if output_length is None:
            output_length = self.config.model.sequence_length - max(tokenized_length)
        assert 1 <= output_length <= self.config.model.sequence_length - max(tokenized_length)
        if print_live:
            logging.info(f"Input prompt: {original_prompt[0]}")
            logging.info("Response:")

        start_time = time.time()
        for _ in range(output_length):
            next_tokens = self.next_token(padded_prompt, tokenized_length, temperature, k, N)

            # update mask based on whether EOS was sampled and whether maximum length was exceeded
            complete_mask = complete_mask | (next_tokens == self.tokenizer.eos_token_id)
            complete_mask = complete_mask | (tokenized_length >= self.config.model.sequence_length)

            if complete_mask.all():
                break

            for i, t in enumerate(next_tokens):
                if complete_mask[i]:
                    continue
                result[i].append(t)

            padded_prompt[
                range(len(prompt)), tokenized_length
            ] = next_tokens  # update final elements in each batch element with next token
            tokenized_length[~complete_mask] += 1  # update length by one for elements that are not complete
            num_generated += len(prompt)

            if print_live and not complete_mask[0]:
                print(self.tokenizer.decode(next_tokens[0]), end="", flush=True)
        end_time = time.time()

        self.decoded_result = self.tokenizer.batch_decode(result)[:N]  # unpad result

        print("")
        if print_final:
            logging.info(f"Output in {end_time - start_time:.2f} seconds")
            logging.info(f"Throughput: {num_generated / (end_time - start_time):.2f} t/s")

        if end_key is not None:
            self.tokenizer.eos_token_id = self.original_eos_token_id

        return self.decoded_result

    def detach(self):
        was_attached_or_device = self.session._was_attached_stack.pop()

        self.session._device.detach()
        self.session._pb_session.setEngineIsLoaded(False)

        # If a DeviceInfo was stored in the stack then restore it.
        if isinstance(was_attached_or_device, popart.DeviceInfo):
            self.session._set_device(was_attached_or_device)


if __name__ == "__main__":
    from utils.setup import dolly_config_setup

    config, _, hf_model = dolly_config_setup("config/inference.yml", "release", "dolly_pod16", hf_model_setup=True)
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
    pipe = DollyPipeline(config, hf_dolly_checkpoint=hf_model, tokenizer=tokenizer)

    print(
        pipe(
            [
                "Make up a question for a school physics exam. Please be serious.",
            ],
            k=5,
            temperature=0.7,
        )
    )
