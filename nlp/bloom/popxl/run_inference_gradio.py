# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
import os
from typing import Optional

import gradio as gr
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from scipy.special import softmax
from popxl_addons.array_munging import repeat

from utils.setup import bloom_config_setup
from config import CONFIG_DIR, BloomConfig

from run_inference import setup_session, next_token, tokenize_initial


def run_inference_popxl(
    config: BloomConfig,
    tokenizer: AutoTokenizer,
    hf_model: AutoModel = None,
):
    tp1, tp2 = config.execution.tensor_parallel_1, config.execution.tensor_parallel_2
    session = setup_session(config, hf_model)

    # Called during `onclick` button event
    def inference_fn(prompt: str, tokens: int, temperature: float, k: int):
        logging.info("Received inference request")
        logging.info(
            f"prompt='{prompt}'",
            f"tokens='{tokens}'",
            f"temperature='{temperature}'",
            f"k='{k}'",
        )

        padded_prompt, tokenized_prompt, tokenized_length = tokenize_initial(prompt, tokenizer, config)

        num_generated = 0
        result = tokenized_prompt.tolist()

        # Begin inference loop
        logging.info("Beginning IPU eval loop")
        for _ in tqdm(range(tokens)):
            t = next_token(session, padded_prompt, tokenized_length[0], tp1, tp2, temperature, k)
            result.append(t)
            padded_prompt[tokenized_length[0]] = t
            tokenized_length[0] += 1
            num_generated += 1

            if result[-1] == tokenizer.eos_token_id or tokenized_length[0] >= config.model.sequence_length:
                break

        logging.info("Detokenizing output")
        return tokenizer.decode(result)

    logging.info("Attaching to IPUs")
    with session:
        logging.info("Initialising Gradio interface")
        demo = gr.Blocks()
        with demo:
            intro_text = (
                "Gradio Demo for Bloom on Graphcore IPUs (Bow Pod16). "
                "To use it, simply add your text or click an example below!\n\n"
                "*Tips for usage:*\n"
                "- Do not talk to Bloom as if it were an entity – it is a text completion model not a chatbot.\n"
                "- Bloom does not perform well for zero-shot generation. It is best to prompt it with examples of content you want.\n"
                "- Bloom occasionally produces unexpected, misleading, or offensive outputs.\n"
                "- A lower temperature biases sampling towards greedy sampling, whereas a higher temperature biases towards uniform sampling. "
                "Hence, use lower temperatures for factual predictions, and higher for creative applications.\n"
                "- The top-k parameter limits sampling to tokens within the top `k` highest probabilities. This essentially prevents low probability tokens from ever being sampled."
            )
            gr.Markdown("# Bloom Inference on Graphcore IPUs")
            gr.Markdown(intro_text)
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", value=" ")
            with gr.Row():
                tokens = gr.Slider(1, 100, value=10, step=1, label="Number of tokens")
                temperature = gr.Slider(0.0, 1.5, step=0.1, value=1.0, label="Sampling temperature")
                topk = gr.Slider(0, 20, step=1, value=3, label="Top-k")

            with gr.Row():
                submit = gr.Button("Submit")

            with gr.Row():
                output = gr.Textbox(label="Output")

            # Inbuilt examples for quick demoing
            gr.Examples(
                examples=[
                    ["Aloha, World!", 10, 0.8, 5],
                    ["How many islands are there in Scotland?", 10, 0.6, 3],
                    ["The English Alphabet is as follows: A B C", 10, 0.0, 1],
                    [
                        "ZH: 早上好 EN: Good morning\nZH: 今天天气什么? EN: What is the weather today?\nZH: 圣诞节快乐！EN:",
                        10,
                        0.7,
                        3,
                    ],
                    ["Math exercise - answers: 34+10=44 54+20=", 16, 0.0, 1],
                ],
                inputs=[prompt, tokens, temperature, topk],
                outputs=output,
                fn=inference_fn,
            )

            submit.click(inference_fn, inputs=[prompt, tokens, temperature, topk], outputs=output)

        logging.info("Launching Gradio interface")
        demo.launch()


def main():
    # Configuration
    config, _, pretrained, tokenizer = bloom_config_setup(
        CONFIG_DIR / "inference.yml",
        "release",
        "bloom_560M",
        hf_model_setup=True,
        hf_tokenizer_setup=True,
    )

    if pretrained:
        pretrained = pretrained.eval()

    run_inference_popxl(config, tokenizer, hf_model=pretrained)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e, exc_info=False)  # Log time of exception
        raise
