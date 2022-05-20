# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
import poptorch
import transformers

from bert_fused_attention import BertFusedSelfAttention
from utils import logger
from transformers.models.bert.modeling_bert import BertLMPredictionHead


class OnehotGather(nn.Module):
    """
    Gathers selected indices from a tensor by transforming the list of indices
    into a one-hot matrix and then multiplying the tensor by that matrix.
    """
    def __init__(self):
        super().__init__()
        self._is_half = False

    def half(self):
        super().half()
        # Tracing is always executed in float as there are missing
        # implementations of operations in half on the CPU.
        # So we cannot query the inputs to know if we are running
        # with a model that has had .half() called on it.
        # To work around it nn.Module::half is overridden
        self._is_half = True

    def forward(self, sequence, positions):
        """
        Gather the vectors at the specific positions over a batch.
        """
        num_classes = int(sequence.shape[1])
        one_hot_positions = F.one_hot(positions, num_classes)
        if self._is_half:
            one_hot_positions = one_hot_positions.half()
        else:
            one_hot_positions = one_hot_positions.float()
        return torch.matmul(one_hot_positions.detach(), sequence)


class SerializedLinear(nn.Linear):
    """
    Exactly equivalent to `nn.Linear` layer, but with the matrix multiplication replaced with
    a serialized matrix multiplication: `poptorch.serializedMatMul`.
    The matrix multiplication is split into separate smaller multiplications, calculated one after the other,
    to reduce the memory requirements of the multiplication and its gradient calculation.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        factor: Number of serialized multiplications. Must be a factor of
            the dimension to serialize on.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
        mode: Which dimension of the matmul to serialize on:
            for matrix A (m by n) multiplied by matrix B (n by p).
            * InputChannels: Split across the input channels (dimension m).
            * ReducingDim: Split across the reducing dimension (n).
            * OutputChannels: Split across the output channels (dimension p).
            * Disabled: Same as an ordinary matrix multiplication.
    """
    def __init__(self, in_features, out_features, factor, bias=False,
                 mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
        recomputed"""
    def recompute_outputs(module, inputs, outputs):
        return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
    module.register_forward_hook(recompute_outputs)


def outline_attribute(module: nn.Module, value: str):
    """Adds an attribute to a module. This attribute will be used
        when comparing operation equivalence in outlining. For example:

        layer1 = nn.Linear(...)
        layer2 = nn.Linear(...)
        layer3 = nn.Linear(...)
        layer4 = nn.Linear(...)
        outline_attribute(layer1, "A")
        outline_attribute(layer2, "A")
        outline_attribute(layer3, "B")

        The code for layer1 can be reused for layer2.
        But it can't be used for layer3 or layer4.
    """
    context = poptorch.Attribute(__outline={"layer": value})

    def enable(*args):
        context.__enter__()

    def disable(*args):
        context.__exit__(None, None, None)
    module.register_forward_pre_hook(enable)
    module.register_forward_hook(disable)


def accuracy(out, targ):
    return (out.argmax(dim=-1) == targ).float().mean()


def accuracy_masked(out, targ, mask_val):
    mask = (targ != mask_val).float()
    num_unmasked = mask.sum(1).unsqueeze(1)
    return (out.argmax(dim=-1) == targ).float().mul(mask).div(num_unmasked).sum(1).mean()


class PipelinedBertForPretraining(transformers.BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedBertForPretraining(config).parallelize().half().train()
        ```
        """
        # Use faster fused-qkv self-attention
        for layer in self.bert.encoder.layer:
            fused = BertFusedSelfAttention(self.config)
            fused.load_state_dict(layer.attention.self.state_dict())
            layer.attention.self = fused

        if self.config.embedding_serialization_factor > 1:
            serialized_decoder = SerializedLinear(self.config.hidden_size,
                                                  self.config.vocab_size,
                                                  self.config.embedding_serialization_factor,
                                                  bias=True,
                                                  mode=poptorch.MatMulSerializationMode.OutputChannels)
            serialized_decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
            self.cls.predictions.decoder = serialized_decoder
            self.tie_weights()

        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger("-------------------- Device Allocation --------------------")
        logger("Embedding  --> IPU 0")
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        # Preventing the embeddings.LayerNorm from being outlined with the encoder.layer.LayerNorm
        # improves the tile mapping of the pipeline stashes
        outline_attribute(self.bert.embeddings.LayerNorm, "embeddings")

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer:
                recomputation_checkpoint(layer)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Encoder {index:<2} --> IPU {ipu}")

        logger("Pooler     --> IPU 0")
        self.bert.pooler = poptorch.BeginBlock(self.bert.pooler, "Pooler", ipu_id=0)

        logger("Classifier --> IPU 0")
        self.cls = poptorch.BeginBlock(self.cls, "Classifier", ipu_id=0)
        logger("-----------------------------------------------------------")
        return self

    def _init_weights(self, module):
        """Initialize the weights"""
        def truncated_normal_(tensor, mean=0, std=1):
            """
            Truncated Normal distribution, truncated at 2 sigma
            """
            r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
            tensor.data.copy_(r)

        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids, masked_lm_positions, masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output, pooled_output = outputs[:2]

        # Select only the masked tokens for the classifier
        masked_output = self.gather_indices(sequence_output, masked_lm_positions)

        prediction_scores, sequential_relationship_score = self.cls(masked_output, pooled_output)
        outputs = (prediction_scores, sequential_relationship_score,) + outputs[2:]

        if masked_lm_labels is not None and next_sentence_label is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=0).float()
            next_sentence_loss = F.cross_entropy(sequential_relationship_score.view(-1, 2), next_sentence_label.view(-1)).float()
            total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")

            next_sentence_acc = accuracy(sequential_relationship_score.view([-1, 2]), next_sentence_label.view(-1))
            # masked_lm_labels: 0 if corresponding token not masked, original value otherwise
            masked_lm_acc = accuracy_masked(prediction_scores.view([-1, self.config.mask_tokens, self.config.vocab_size]), masked_lm_labels, 0)
            outputs = (total_loss, masked_lm_loss, next_sentence_loss, masked_lm_acc, next_sentence_acc)

        return outputs


class PipelinedPackedBertForPretraining(PipelinedBertForPretraining):
    def __init__(self, config):
        super().__init__(config)
        self.bert.pooler = PackedBertPooler(config)
        self.cls = PackedBertPreTrainingHeads(config)
        # Since we're redefining the output embedding in self.cls, we need to make sure we are sharing the input and output embeddings.
        self.tie_weights()

    def forward(self, packed_input_ids, packed_input_mask, packed_segment_ids, packed_position_ids, packed_masked_lm_positions,
                packed_masked_lm_ids, packed_masked_lm_mask, packed_next_sentence_labels, packed_next_sentence_mask):
        bs, seq_len = packed_input_mask.shape
        # bs, seq_len -> bs, 1, seq_len -> bs, seq_len, seq_len
        attention_mask = packed_input_mask[:, None, :].repeat(1, seq_len, 1)
        attention_mask = attention_mask == attention_mask.transpose(1, 2)

        outputs = self.bert(input_ids=packed_input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=packed_segment_ids,
                            position_ids=packed_position_ids)
        sequence_output, pooled_output_list = outputs[:2]

        # Select only the masked tokens for the classifier
        masked_output = self.gather_indices(sequence_output, packed_masked_lm_positions)

        prediction_scores, seq_relationship_scores = self.cls(masked_output, pooled_output_list)
        outputs = (prediction_scores, seq_relationship_scores,) + outputs[2:]

        if packed_masked_lm_ids is not None and packed_next_sentence_labels is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                packed_masked_lm_ids.view(-1),
                ignore_index=0).float()
            next_sentence_loss = F.cross_entropy(seq_relationship_scores.transpose(1, 2), packed_next_sentence_labels, reduction='none').float()
            next_sentence_loss *= packed_next_sentence_mask
            next_sentence_loss = next_sentence_loss.sum() / packed_next_sentence_mask.sum()
            total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")

            next_sentence_acc = accuracy_packed(seq_relationship_scores.transpose(1, 2), packed_next_sentence_labels, packed_next_sentence_mask)
            masked_lm_acc = accuracy_masked(prediction_scores.view([-1, self.config.mask_tokens, self.config.vocab_size]), packed_masked_lm_ids, 0)
            packing_ratio = torch.mean(torch.sum(packed_next_sentence_mask, 1))
            outputs = (total_loss, masked_lm_loss, next_sentence_loss, masked_lm_acc, next_sentence_acc, packing_ratio)

        return outputs


def accuracy_packed(out, targ, packed_next_sentence_mask):
    # acc: (bs, max_sequences_per_pack)
    acc = (out.argmax(dim=-2) == targ).float()
    acc *= packed_next_sentence_mask
    acc = acc.sum() / packed_next_sentence_mask.sum()
    return acc


class PackedBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_sequences_per_pack = config.max_sequences_per_pack
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden states corresponding
        # to the last max_sequences_per_pack tokens. Note that the [CLS] tokens
        # are always located at the end of the pack. When the actual number of
        # sequences is lower than max_sequences_per_pack, we still slice out
        # the last max_sequences_per_pack tokens, but we will not use all of
        # them during loss calculation.
        last_tokens_tensors = hidden_states[:, -self.max_sequences_per_pack:]
        pooled_output_list = []
        for i in range(self.max_sequences_per_pack):
            output = self.dense(last_tokens_tensors[:, i])
            output = self.activation(output)
            pooled_output_list.append(output)
        return pooled_output_list


class PackedBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_sequences_per_pack = config.max_sequences_per_pack
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output_list):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score_list = []
        for i in range(self.max_sequences_per_pack):
            score = self.seq_relationship(pooled_output_list[i])
            seq_relationship_score_list.append(score)
        seq_relationship_scores = torch.stack(seq_relationship_score_list, dim=1)
        return prediction_scores, seq_relationship_scores


class SerializedEmbedding(nn.Module):
    """
    Wrapper for `nn.Embedding` layer that performs the embedding look-up into
    smaller serialized steps in order to reduce memory in the embedding gradient
    calculation.

    Args:
        embedding: A `nn.Embedding` to wrap
        serialization_factor: The number of serialized embedding look-ups
    """
    def __init__(self, embedding: nn.Embedding, serialization_factor: int):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.num_embeddings = embedding.num_embeddings

        # Num embeddings should be divisible by the serialization factor
        assert self.num_embeddings % self.serialization_factor == 0
        self.split_size = self.num_embeddings // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [nn.Embedding.from_pretrained(embedding.weight[i*self.split_size:(i+1)*self.split_size, :].detach(),
                                          freeze=False,
                                          padding_idx=embedding.padding_idx if i == 0 else None)
             for i in range(self.serialization_factor)])

    def deserialize(self):
        """
        Deserialize the internal wrapped embedding layer and return it as a
        `nn.Embedding` object.

        Returns:
            `nn.Embedding` layer
        """
        return nn.Embedding.from_pretrained(torch.vstack([l.weight for l in self.split_embeddings]), padding_idx=0)

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


class PipelinedBertForQuestionAnswering(transformers.BertForQuestionAnswering):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedBertForQuestionAnswering(config).parallelize().half()
        ```
        """
        # Use faster fused-qkv self-attention
        for layer in self.bert.encoder.layer:
            fused = BertFusedSelfAttention(self.config)
            fused.load_state_dict(layer.attention.self.state_dict())
            layer.attention.self = fused

        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger("-------------------- Device Allocation --------------------")
        logger("Embedding  --> IPU 0")
        if self.config.embedding_serialization_factor > 1:
            self.bert.embeddings.word_embeddings = SerializedEmbedding(self.bert.embeddings.word_embeddings,
                                                                       self.config.embedding_serialization_factor)
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        outline_attribute(self.bert.embeddings.LayerNorm, "embedding")

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                recomputation_checkpoint(layer)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Encoder {index:<2} --> IPU {ipu}")

        logger(f"QA Outputs --> IPU {ipu}")
        self.qa_outputs = poptorch.BeginBlock(self.qa_outputs, "QA Outputs", ipu_id=ipu)
        logger("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.BertForQuestionAnswering`.
        """
        # Deserialize the serialized word embedding
        if self.config.embedding_serialization_factor > 1:
            self.bert.embeddings.word_embeddings = self.bert.embeddings.word_embeddings.deserialize()
        return self

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        output = super().forward(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 start_positions=start_positions,
                                 end_positions=end_positions)
        if self.training:
            final_loss = poptorch.identity_loss(output.loss, reduction="none")
            return final_loss, output.start_logits, output.end_logits
        else:
            return output.start_logits, output.end_logits
