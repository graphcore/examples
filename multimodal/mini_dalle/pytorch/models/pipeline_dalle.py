# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from torch import nn
from models.dalle import DALLE
import poptorch


# IPU related functions


class SerializedLinear(nn.Linear):
    def __init__(self, in_features, out_features, factor=1, bias=True,
                 mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output.view(x.shape[0], -1, self.out_features)


class SerializedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, serialization_factor=1):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Num embeddings should be divisible by the serialization factor
        assert self.num_embeddings % self.serialization_factor == 0
        self.split_size = self.num_embeddings // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [nn.Embedding(self.split_size, embedding_dim)
             for i in range(self.serialization_factor)])

    def forward(self, indices):
        # iterate through the splits
        x_sum = None
        for i in range(self.serialization_factor):
            # mask out the indices not in this split
            split_indices = indices - i * self.split_size
            mask = (split_indices >= 0) * (split_indices < self.split_size)
            mask = mask.detach()
            split_indices *= mask

            # do the embedding lookup
            x = self.split_embeddings[i](split_indices)

            # multiply the output by mask
            x *= mask.unsqueeze(-1)

            # add to partial
            if x_sum is not None:
                x_sum += x
            else:
                x_sum = x
        return x_sum


class WrappedDALLE(nn.Module):
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
                 embedding_ipu_id = 0,
                 embedding_serialization_factor = 1,
                 layers_per_ipu = [0, 0, 8, 8],
                 cls_ipu_id = None,
                 fp16 = False,
                 byteio = False):
        super().__init__()
        self.model = DALLE(dim=dim,
                           vae=vae,
                           num_text_tokens=num_text_tokens,
                           text_seq_len=text_seq_len,
                           depth=depth,
                           heads=heads,
                           dim_head=dim_head,
                           attn_dropout=attn_dropout,
                           ff_dropout=ff_dropout,
                           sparse_attn=sparse_attn,
                           attn_types=attn_types,
                           loss_img_weight=loss_img_weight,
                           sandwich_norm=sandwich_norm,
                           fp16=fp16,
                           byteio=byteio)

        assert(sum(layers_per_ipu) == depth)
        if embedding_serialization_factor > 1:
            self.model.text_emb = SerializedEmbedding(self.model.num_text_tokens, dim, embedding_serialization_factor)
            self.model.to_logits[1] = SerializedLinear(dim, self.model.total_tokens, factor=embedding_serialization_factor)
        self.model.vae = poptorch.BeginBlock(self.model.vae, "VAE", ipu_id=0)
        self.model.image_emb = poptorch.BeginBlock(self.model.image_emb, "image_emb", ipu_id=embedding_ipu_id)
        layer = 0
        for i in range(len(layers_per_ipu)):
            if layers_per_ipu[i] > 0:
                self.model.transformer.layers[layer] = poptorch.BeginBlock(self.model.transformer.layers[layer], "Transformer_"+str(layer), ipu_id=i)
                layer = layer + layers_per_ipu[i]
        if cls_ipu_id is not None:
            self.model.to_logits = poptorch.BeginBlock(self.model.to_logits, "cls", ipu_id=cls_ipu_id)

    def generate_texts(self,
                       text=None,
                       *,
                       filter_thres = 0.5,
                       temperature = 1.):
        return self.model.generate_texts(text=text, filter_thres=filter_thres, temperature=temperature)

    def generate_images(self,
                        text,
                        *,
                        clip = None,
                        mask = None,
                        filter_thres = 0.5,
                        temperature = 1.,
                        img = None,
                        num_init_img_tokens = None):
        return self.model.generate_images(text=text, clip=clip, mask=mask, filter_thres=filter_thres, temperature=temperature, img=img, num_init_img_tokens=num_init_img_tokens)

    def forward(self,
                text,
                image = None,
                mask = None):
        return self.model.forward(text=text, image=image, mask=mask)
