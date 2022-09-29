# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 lucidrains

# This file has been modified by Graphcore


from inspect import isfunction
from math import ceil

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# helpers


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(fp16 = False):
    if fp16:
        return -torch.finfo(torch.float16).max
    else:
        return -torch.finfo(torch.float32).max

# classes


class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., fp16 = False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.fp16 = fp16
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        softmax = torch.softmax

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(self.fp16)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device=x.device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

# sparse attention with convolutional pattern, as mentioned in the blog post. customizable kernel size and dilation


class SparseConvCausalAttention(nn.Module):
    def __init__(self, dim, seq_len, image_size = 32, kernel_size = 5, dilation = 1, heads = 8, dim_head = 64, dropout = 0., fp16 = False, **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel size must be odd'

        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.fp16 = False
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h, img_size, kernel_size, dilation, seq_len = *x.shape, self.heads, self.image_size, self.kernel_size, self.dilation, self.seq_len
        softmax = torch.softmax

        img_seq_len = img_size ** 2
        text_len = seq_len + 1 - img_seq_len

        # padding

        padding = seq_len - n + 1
        mask = default(mask, lambda: torch.ones(b, text_len, device=x.device).bool())

        x = F.pad(x, (0, 0, 0, padding), value = 0)
        mask = mask[:, :text_len]

        # derive query / keys / values

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        q = q * self.scale

        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))

        # text attention

        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(self.fp16)

        i, j = dots_text.shape[-2:]
        text_causal_mask = torch.ones(i, j, device=x.device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)

        attn_text = softmax(dots_text, dim = -1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)

        # image attention

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding = effective_kernel_size // 2

        k_img, v_img = map(lambda t: rearrange(t, 'b (h w) c -> b c h w', h = img_size), (k_img, v_img))
        k_img, v_img = map(lambda t: F.unfold(t, kernel_size, padding = padding, dilation = dilation), (k_img, v_img))
        k_img, v_img = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j = kernel_size ** 2), (k_img, v_img))

        # let image attend to all of text

        dots_image = einsum('b i d, b i j d -> b i j', q_img, k_img)
        dots_image_to_text = einsum('b i d, b j d -> b i j', q_img, k_text)

        # calculate causal attention for local convolution

        i, j = dots_image.shape[-2:]
        img_seq = torch.arange(img_seq_len, device=x.device)
        k_img_indices = rearrange(img_seq.float(), '(h w) -> () () h w', h = img_size)
        k_img_indices = F.pad(k_img_indices, (padding,) * 4, value = img_seq_len)  # padding set to be max, so it is never attended to
        k_img_indices = F.unfold(k_img_indices, kernel_size, dilation = dilation)
        k_img_indices = rearrange(k_img_indices, 'b j i -> b i j')

        # mask image attention

        q_img_indices = rearrange(img_seq, 'i -> () i ()')
        causal_mask = q_img_indices < k_img_indices

        # concat text mask with image causal mask

        causal_mask = repeat(causal_mask, '() i j -> b i j', b = b * h)
        mask = repeat(mask, 'b j -> (b h) i j', i = i, h = h)
        mask = torch.cat((~mask, causal_mask), dim = -1)

        # image can attend to all of text

        dots = torch.cat((dots_image_to_text, dots_image), dim = -1)
        dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim = -1)

        # aggregate

        attn_image_to_text, attn_image = attn[..., :text_len], attn[..., text_len:]

        out_image_to_image = einsum('b i j, b i j d -> b i d', attn_image, v_img)
        out_image_to_text = einsum('b i j, b j d -> b i d', attn_image_to_text, v_text)

        out_image = out_image_to_image + out_image_to_text

        # combine attended values for both text and image

        out = torch.cat((out_text, out_image), dim = 1)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return out[:, :n]

# sparse axial causal attention


class SparseAxialCausalAttention(nn.Module):
    def __init__(self, dim, seq_len, image_size = 32, axis = 0, heads = 8, dim_head = 64, dropout = 0., fp16 = False, **kwargs):
        super().__init__()
        assert axis in {0, 1}, 'axis must be either 0 (along height) or 1 (along width)'
        self.axis = axis

        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size

        self.fp16 = fp16
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h, img_size, axis, seq_len = *x.shape, self.heads, self.image_size, self.axis, self.seq_len
        softmax = torch.softmax

        img_seq_len = img_size ** 2
        text_len = seq_len + 1 - img_seq_len

        # padding

        padding = seq_len - n + 1
        mask = default(mask, lambda: torch.ones(b, text_len, device=x.device).bool())

        x = F.pad(x, (0, 0, 0, padding), value = 0)
        mask = mask[:, :text_len]

        # derive queries / keys / values

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        q = q * self.scale

        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(lambda t: (t[:, :-img_seq_len], t[:, -img_seq_len:]), (q, k, v))

        # text attention

        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(self.fp16)

        i, j = dots_text.shape[-2:]
        text_causal_mask = torch.ones(i, j, device=x.device).triu_(j - i + 1).bool()
        dots_text.masked_fill_(text_causal_mask, mask_value)

        attn_text = softmax(dots_text, dim = -1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)

        # image attention

        split_axis_einops = 'b (h w) c -> b h w c' if axis == 0 else 'b (h w) c -> b w h c'
        merge_axis_einops = 'b x n d -> b (x n) d' if axis == 0 else 'b x n d -> b (n x) d'

        # split out axis

        q_img, k_img, v_img = map(lambda t: rearrange(t, split_axis_einops, h = img_size), (q_img, k_img, v_img))

        # similarity

        dots_image_to_image = einsum('b x i d, b x j d -> b x i j', q_img, k_img)
        dots_image_to_text = einsum('b x i d, b j d -> b x i j', q_img, k_text)

        dots = torch.cat((dots_image_to_text, dots_image_to_image), dim = -1)

        # mask so image has full attention to text, but causal along axis

        bh, x, i, j = dots.shape
        causal_mask = torch.ones(i, img_size, device=text_causal_mask.device).triu_(img_size - i + 1).bool()
        causal_mask = repeat(causal_mask, 'i j -> b x i j', b = bh, x = x)

        mask = repeat(mask, 'b j -> (b h) x i j', h = h, x = x, i = i)
        mask = torch.cat((~mask, causal_mask), dim = -1)

        dots.masked_fill_(mask, mask_value)

        # attention.

        attn = softmax(dots, dim = -1)

        # aggregate

        attn_image_to_text, attn_image_to_image = attn[..., :text_len], attn[..., text_len:]

        out_image_to_image = einsum('b x i j, b x j d -> b x i d', attn_image_to_image, v_img)
        out_image_to_text = einsum('b x i j, b j d -> b x i d', attn_image_to_text, v_text)

        out_image = out_image_to_image + out_image_to_text

        # merge back axis

        out_image = rearrange(out_image, merge_axis_einops, x = img_size)

        # combine attended values for both text and image

        out = torch.cat((out_text, out_image), dim = 1)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return out[:, :n]
