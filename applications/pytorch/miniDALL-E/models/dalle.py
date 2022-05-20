# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 lucidrains

# This file has been modified by Graphcore


from math import log2, sqrt
import torch
import poptorch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from models import tokenizer
from models.vae import VQGanVAE
from models.transformer import Transformer

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    def inner(*args, **kwargs):
        return val
    return inner


def is_empty(t):
    return t.nelement() == 0


def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


# sampling helpers


def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# main DALL-E class


class DALLE(nn.Module):
    def __init__(self,
                 *,
                 dim,
                 vae,
                 num_text_tokens = 10000,
                 text_seq_len = 256,
                 depth,
                 heads = 8,
                 dim_head = 64,
                 attn_dropout = 0.,
                 ff_dropout = 0,
                 sparse_attn = False,
                 attn_types = None,
                 loss_img_weight = 7,
                 sandwich_norm = False,
                 fp16 = False,
                 byteio = False):
        super().__init__()
        assert isinstance(vae, (VQGanVAE)), 'vae must be an instance of DiscreteVAE'

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2

        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)

        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim)  # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size))

        self.num_text_tokens = num_text_tokens  # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens

        self.vae = vae.eval()
        set_requires_grad(self.vae, False)  # freeze VAE from being trained

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
            sparse_attn = sparse_attn,
            sandwich_norm = sandwich_norm,
            fp16 = fp16,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        self.logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )
        self.logits_mask.requires_grad = False

        self.loss_img_weight = loss_img_weight
        self.fp16 = fp16
        self.byteio = byteio


    @torch.no_grad()
    @eval_decorator
    def generate_texts(self,
                       text=None,
                       *,
                       filter_thres = 0.5,
                       temperature = 1.):
        text_seq_len = self.text_seq_len
        if text is None or text == "":
            text_tokens = torch.tensor([[0]])
        else:
            text_tokens = torch.tensor(tokenizer.tokenizer.encode(text)).unsqueeze(0)

        for _ in range(text_tokens.shape[1], text_seq_len):
            tokens = self.text_emb(text_tokens)
            tokens += self.text_pos_emb(torch.arange(text_tokens.shape[1]))

            seq_len = tokens.shape[1]

            output_transf = self.transformer(tokens)

            logits = self.to_logits(output_transf)

            # mask logits to make sure text predicts text (except last token), and image predicts image

            logits_mask = self.logits_mask[:, :seq_len]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(logits_mask, max_neg_value)
            logits = logits[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)

            text_tokens = torch.cat((text_tokens, sample), dim=-1)

        padding_tokens = set(np.arange(self.text_seq_len) + (self.num_text_tokens - self.text_seq_len))
        texts = [tokenizer.tokenizer.decode(text_token, pad_tokens=padding_tokens) for text_token in text_tokens]
        return text_tokens, texts


    @torch.no_grad()
    @eval_decorator
    def generate_images(self,
                        text,
                        *,
                        clip = None,
                        mask = None,
                        filter_thres = 0.5,
                        temperature = 1.,
                        img = None,
                        num_init_img_tokens = None):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        text = text[:, :text_seq_len]  # make sure text is within bounds
        out = text

        if exists(img):
            image_size = vae.image_size
            assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[3] == image_size, f'input image must have the correct image size {image_size}'

            indices = vae.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
            assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'

            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim = -1)

        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(text, image, mask = mask)[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)

            sample -= (num_text_tokens if is_image else 0)  # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)

            if out.shape[1] <= text_seq_len:
                mask = F.pad(mask, (0, 1), value = True)

        text_seq = out[:, :text_seq_len]

        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)

        if exists(clip):
            scores = clip(text_seq, images)
            return images, scores

        return images

    def forward(self,
                text,
                image = None,
                mask = None):
        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                image_size = self.vae.image_size
                if self.byteio:
                    if self.fp16:
                        image = image.half()
                    else:
                        image = image.float()
                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_emb(image)

            image_emb += self.image_pos_emb(image_emb)

        # make sure padding in text tokens get unique padding token id

        text_range = torch.arange(self.text_seq_len) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # add <bos>

        text = F.pad(text, (1, 0), value = 0)

        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1]))

        seq_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            tokens = torch.cat((tokens, image_emb), dim = 1)
            seq_len += image_len

        # when training, the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if self.training:
            seq_len -= 1
            tokens = tokens[:, :-1]

        out = self.transformer(tokens)

        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image

        logits_mask = self.logits_mask[:, :seq_len]
        if self.fp16:
            max_neg_value = -torch.finfo(torch.float16).max
        else:
            max_neg_value = -torch.finfo(torch.float32).max
        logits.masked_fill_(logits_mask, max_neg_value)

        if not self.training:
            return logits

        assert exists(image), 'when training, image must be supplied'

        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim = 1)

        logits = rearrange(logits, 'b n c -> b c n')

        loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len].permute([0, 2, 1]).reshape([-1, self.total_tokens]), labels[:, :self.text_seq_len].reshape(-1))
        loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:].permute([0, 2, 1]).reshape([-1, self.total_tokens]), labels[:, self.text_seq_len:].reshape(-1))
        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)

        return poptorch.identity_loss(loss, reduction='none')
