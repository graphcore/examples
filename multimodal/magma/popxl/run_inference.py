# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from configs import CONFIG_DIR, MagmaConfig
from inference import inference
from utils.setup import magma_config_setup, set_random_seeds
from utils.sampling import generate

from modelling.gptj.embedding import GPTJEmbeddingsTP
from modelling.magma_mapping import load_magma, magma_mapping

from popxl.utils import to_numpy
from popxl_addons import timer
from popxl_addons.array_munging import tensor_parallel_input, repeat

from magma.image_input import ImageInput
from magma.transforms import clip_preprocess

from functools import partial
import numpy as np
import random
import torch
from typing import Literal


def init_inference_session(
    key: Literal["magma_v1_500", "magma_v1_1024"] = "magma_v1_500", checkpoint_path="./mp_rank_00_model_states.pt"
):
    """
    Create the inference session and loads magma checkpoint to IPU.

    Args:
        - key (Literal['magma_v1_500', 'magma_v1_1024']): string identifying a config defined in configs/inference.yml.

    Returns:
        - session: an open inference session. Weights have already been written.
        - config: the config used to initialise the session
        - tokenizer: the GPT2 tokenizer used by the gpt-j language model.
    """
    # Load config 0 - This also set the random seed
    config, *_ = magma_config_setup(CONFIG_DIR / "inference.yml", "release", key)
    # build session
    session = inference(config)

    # Load pretrained magma
    with timer("Loading magma weights to host"):
        pretrained_magma = load_magma(checkpoint_path, config)

    # Load weights to IPU
    with timer("Loading magma pretrained model to IPU"):
        # in this way the session stays open and RemoteBufferWeightsFromHost is not called on exit
        session.__enter__()
        weights = magma_mapping(config, session, pretrained_magma)
        session.write_variables_data(weights)

    return session, config, pretrained_magma.tokenizer


def run_inference(
    session,
    config,
    tokenizer,
    image_url,
    text_prompt,
    seed: int,
    top_p: float = 0.9,
    top_k: float = 0.0,
    temperature: float = 0.7,
    max_out_tokens: float = 6,
):
    """
    Run magma inference using the given text prompt and image url.
    """
    # Configs
    tp = config.transformer.execution.tensor_parallel
    rf = config.ipus

    # Prepare input
    image = ImageInput(image_url)
    tokenized_text = tokenizer.encode(text_prompt)
    img = image.get_transformed_image(clip_preprocess(config.visual.image_resolution))
    img = img.half()

    # Set seed
    set_random_seeds(seed)

    with timer("Generating answer"):
        with session:
            tokens = 0
            tokenized_text = to_numpy(tokenized_text, session.inputs.words.dtype).flatten()
            L = len(tokenized_text)
            inital_len = L
            words = tokenized_text
            # pad input
            # NOTE config.transformer.sequence_length is modified in inference.py (line 72) to include also image tokens (144)
            words = np.pad(
                words, (0, config.transformer.sequence_length - 144 - L), constant_values=tokenizer.pad_token_id
            )
            words = words.reshape(-1, *session.inputs.words.shape)
            output = []
            while True:
                data_map = {}
                # data map
                data_map[session.inputs.words] = tensor_parallel_input(
                    words, tp, rf, partial(GPTJEmbeddingsTP.offset_input, config=config.transformer)
                ).squeeze()
                data_map[session.inputs.image] = repeat(img, rf, axis=0)
                logits_shards = session.run(data_map)[session.outputs.logits]
                logits = np.concatenate(logits_shards, axis=-1)
                # next token logits - located at the last token index, that is 144+L-1
                logits = torch.Tensor(logits[144 + L - 1, :])
                # next token selection with torch
                next_token_idx = generate(
                    logits.reshape(1, logits.shape[-1]), top_k=top_k, top_p=top_p, temperature=temperature
                )
                output.append(next_token_idx[0].data[0])
                tokens += 1
                if next_token_idx == tokenizer.eos_token_id or tokens == max_out_tokens:
                    break
                # add new token to text input (overwriting padding)
                words[:, L] = next_token_idx
                L += 1
    return tokenizer.decode(output, skip_special_tokens=True)


if __name__ == "__main__":
    session, config, tokenizer = init_inference_session("magma_v1_500")
    img = "https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg"
    txt = "A painting of "
    response = run_inference(session, config, tokenizer, img, txt, config.seed)
    print(response)
