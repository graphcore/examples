# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import popart
import numpy as np
from scipy.stats import truncnorm
from typing import NamedTuple, Optional
from contextlib import ExitStack
from collections import defaultdict
from enum import Enum
import logging
import math
import os
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from enum import Enum
from functools import reduce
from typing import List, NamedTuple, Optional

import numpy as np
from scipy.stats import truncnorm

import popart
from phased_execution.scope_manager import ScopeProvider
from utils import packed_bert_utils

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    DEFAULT = "DEFAULT"
    PIPELINE = "PIPELINE"
    PHASED = "PHASED"


class BertConfig(NamedTuple):
    micro_batch_size: int = 1
    sequence_length: int = 128
    max_positional_length: int = 512

    # Look up embedding on CPU
    # Possible values:
    #   NONE  = all embeddings on IPU
    #   WORD  = word embeddings on CPU, position embeddings on IPU
    #   ALL   = all embeddings on CPU, both word and position embeddings sent to IPU
    #   MERGE = all embeddings on CPU, sum of word and position embeddings sent to IPU
    host_embedding: str = "NONE"
    # PRETRAINING Only
    mask_tokens: int = 20

    vocab_length: int = 30400
    hidden_size: int = 768

    # Feed Forward is 4 * hidden_size unless specified by --ff-size
    ff_size__: Optional[int] = None

    @property
    def ff_size(self):
        if self.ff_size__ is not None:
            return self.ff_size__
        return self.hidden_size * 4

    attention_heads: int = 12

    attention_bias: bool = False

    inference: bool = False

    num_layers: int = 2

    # Specify the ipu to start adding encoders.
    # if encoder_start_ipu >= 2: two IPUs will be used for the embeddings
    # else: one IPU will be used for the embeddings
    encoder_start_ipu: int = 2

    # Placement of layers can be specified by either:
    # a single element list, which will place num_layers/layers_per_ipu[0] on each IPU
    # Or a list specifying the placement on each IPU.
    layers_per_ipu: List[int] = [2]

    # Number of layers per phased execution phase (only used in phased mode)
    layers_per_phase: int = 3
    # Specify the available memory proportion to be used by the Encoder MatMuls.
    # The same as `layers_per_ipu` this can be a single element list or a
    # list providing a specific value for each IPU.
    available_memory_proportion: List[float] = [0.1525878906]

    # This controls how recomputation is handled in pipelining.
    # If True the output of each layer will be stashed keeping the max liveness
    # of activations to be at most one Encoder layer on each IPU.
    # However, the stash size scales with the number of pipeline stages so this may not always be beneficial.
    # The added stash + code could be greater than the reduction in temporary memory.
    recompute_checkpoint_every_layer: bool = True

    split_transformer: bool = False

    no_dropout: bool = False
    no_attn_dropout: bool = False
    dropout_prob: float = 0.1
    attn_dropout_prob: float = 0.1

    layer_norm_eps: float = 0.001

    # Choices: PRETRAINING (MLM + NSP), SQUAD
    task: str = "PRETRAINING"

    # This option serializes all matmul layers to multiples
    # {N, hidden_size} x {hidden_size, hidden_size}.
    # This is required for sequence length 384.
    split_linear_layers: bool = False

    split_qkv: bool = False

    no_mask: bool = False

    activation_type: str = 'Gelu'

    relu_leak: float = 0.1

    # Choices: FLOAT, FLOAT16
    popart_dtype: str = "FLOAT16"

    @property
    def dtype(self):
        if self.popart_dtype == "FLOAT":
            return np.float32
        elif self.popart_dtype == "FLOAT16":
            return np.float16
        else:
            raise ValueError("BertConfig.dtype must be 'FLOAT' or 'FLOAT16'")

    @property
    def qkv_length(self):
        return self.hidden_size / self.attention_heads

    @property
    def max_lm_predictions(self):
        # When using the packed sequence data format, the number of mask_tokens is
        # increased by the number of sequences per pack.
        if not self.use_packed_sequence_format:
            return self.mask_tokens
        else:
            return self.mask_tokens + self.max_sequences_per_pack

    use_packed_sequence_format: bool = False
    max_sequences_per_pack: int = 1

    # In PRETRAINING this sets how many steps to serialise both the
    # embedding and projection
    embedding_serialization_vocab_steps: int = 1

    update_embedding_dict: bool = True

    use_default_available_memory_proportion: bool = False

    no_cls_layer: bool = False

    projection_bias: bool = False

    num_attention_splits: int = 1
    num_ffwd_splits: int = 1

    execution_mode: ExecutionMode = ExecutionMode.DEFAULT
    num_io_tiles: int = 0
    phased_execution_type: str = "SINGLE"

    # Return the start and end logits as a single tensor to be sliced on the host.
    squad_single_output: bool = True
    # Execute the squad task head on IPU0. If False, execute on the final encoder IPU/stage.
    squad_wrap_final_layer: bool = True


class DeviceScope(object):
    def __init__(self,
                 builder,
                 execution_mode=ExecutionMode.DEFAULT,
                 virtualGraph=None,
                 pipelineStage=None,
                 executionPhase=None,
                 nameScope=None,
                 additional_scopes=None):
        self.builder = builder
        self.execution_mode = execution_mode
        self.virtualGraph = virtualGraph
        self.pipelineStage = pipelineStage
        self.executionPhase = executionPhase
        self.nameScope = nameScope
        self.additional_scopes = additional_scopes or []

    def __enter__(self):
        self.stack = ExitStack()
        # ExecutionPhase will automatically set the virtualGraph attributes based on execution phase
        if self.execution_mode != ExecutionMode.PHASED \
                and self.virtualGraph is not None:
            self.stack.enter_context(
                self.builder.virtualGraph(self.virtualGraph))

        if self.execution_mode == ExecutionMode.PIPELINE\
                and self.pipelineStage is not None:
            self.stack.enter_context(
                self.builder.pipelineStage(self.pipelineStage))

        if self.execution_mode == ExecutionMode.PHASED\
                and self.executionPhase is not None:
            self.stack.enter_context(
                self.builder.executionPhase(self.executionPhase))

        if self.nameScope is not None:
            self.stack.enter_context(self.builder.nameScope(self.nameScope))
        for scope in self.additional_scopes:
            self.stack.enter_context(scope)
        return self

    def __exit__(self, *exp):
        self.stack.close()
        return False


class Model(object):
    def __init__(self, builder=popart.Builder(), initializers=None, execution_mode=ExecutionMode.DEFAULT):
        if initializers is None:
            initializers = {}
        self.builder = builder
        self.initializers = initializers
        if type(execution_mode) == str:
            execution_mode = ExecutionMode(execution_mode)
        self.execution_mode = execution_mode

        # Keep track of tensors in order to give them different parameters
        self.tensors = defaultdict(set)

    def normal_init_tensor(self, dtype, shape, mean, std_dev, debug_name=""):
        data = self.normal_init_data(dtype, shape, mean, std_dev, debug_name)
        tensor = self.builder.addInitializedInputTensor(data, debug_name)
        self._add_to_tensor_map(tensor)
        return tensor

    def normal_init_data(self, dtype, shape, mean, std_dev, debug_name=""):
        name = self.builder.getNameScope(debug_name)
        data = self.initializers.get(name, None)
        if data is None:
            # Truncated random normal between 2 standard devations
            data = truncnorm.rvs(-2, 2, loc=mean,
                                 scale=std_dev, size=np.prod(shape))
            data = data.reshape(shape).astype(dtype)
            self.initializers[name] = data
        else:
            if np.any(data.shape != np.array(shape)):
                if np.all(data.T.shape == np.array(shape)):
                    data = data.T.copy()
                    logger.warning(
                        f"Initializer for {name} was provided transposed.")
                else:
                    raise RuntimeError(f"Initializer {name} does not match shapes. \n"
                                       f" Provided {data.shape}. Required {shape}")
        return data

    def constant_init_tensor(self, dtype, shape, scalar, debug_name="", is_const=False):
        data = self.initializers.get(
            self.builder.getNameScope(debug_name), None)
        if data is None:
            data = np.full(shape, scalar).astype(dtype)
        else:
            if np.any(data.shape != shape):
                raise RuntimeError(f"Initializer {self.builder.getNameScope(debug_name)} does not match shapes. \n"
                                   f" Provided {data.shape}. Required {shape}")
        if is_const:
            return self.builder.aiOnnx.constant(data, debug_name)
        tensor = self.builder.addInitializedInputTensor(data, debug_name)
        self._add_to_tensor_map(tensor)
        return tensor

    def constant_tensor(self, value, dtype=None, debug_name=""):
        value = np.array(value)
        if dtype is not None:
            value = value.astype(dtype)
        return self.builder.aiOnnx.constant(value, debug_name)

    def device_scope(self, virtualGraph=None, pipelineStage=None, executionPhase=None, nameScope=None, additional_scopes=None):
        return DeviceScope(self.builder, self.execution_mode, virtualGraph, pipelineStage, executionPhase, nameScope, additional_scopes)

    def _add_to_tensor_map(self, tensor):
        if self.builder.hasPipelineStage():
            pipeline_stage = self.builder.getPipelineStage()
            self.tensors[pipeline_stage].add(tensor)
        else:
            self.tensors[0].add(tensor)


class Bert(Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.dropout_modifier = 256

        # This is the TensorId for the shared embedding & projection
        self.embedding_dict = None

        # This dict[ipu,TensorId] reuses any mask already generated on an IPU/pipeline stage
        self.masks = {}

        self.init_device_placement()

    def init_device_placement(self):
        ''' Create a DeviceScope for each layer, ie Embedding, SQUAD, NSP
        '''
        # Precompute offset for masks, which are prepared in phase 0 and 1
        self.execution_phase_precompute_offset = 2

        layer_offset = self.config.encoder_start_ipu

        # Embedding
        self.embedding_scope = self.device_scope(
            0, 0, self.execution_phase_precompute_offset)
        ipu = max(layer_offset - 1, 0)
        self.embedding_split_scope = self.device_scope(
            ipu, ipu, ipu + self.execution_phase_precompute_offset)
        # Transformer Layers
        pipeline_stage = self.encoder_layer_ipu(self.config.num_layers - 1) + 1
        execution_phase = pipeline_stage + self.execution_phase_precompute_offset
        # Task Layers
        if self.config.task in ("NSP", "PRETRAINING"):
            self.nsp_scope = self.device_scope(
                self.embedding_split_scope.virtualGraph, pipeline_stage, execution_phase, "NSP")
            self.cls_scope = self.device_scope(
                self.embedding_split_scope.virtualGraph, pipeline_stage, execution_phase, "CLS")
        if self.config.task == "PRETRAINING":
            if self.embedding_scope.virtualGraph != self.embedding_split_scope.virtualGraph:
                pipeline_stage += 1
                execution_phase += 1
            self.mlm_scope = self.device_scope(
                self.embedding_scope.virtualGraph, pipeline_stage, execution_phase, "MLM")
            self.final_loss_scope = self.mlm_scope
        if self.config.task == "SQUAD":
            ipu = self.embedding_scope.virtualGraph
            if not self.config.squad_wrap_final_layer:
                pipeline_stage -= 1
                ipu = pipeline_stage
            self.squad_scope = self.device_scope(
                ipu, pipeline_stage, execution_phase, "Squad")
            self.final_loss_scope = self.squad_scope

        # Scope to place all IO on first IPU for inference:
        if self.config.inference:
            pipeline_stage += 1
            execution_phase += 1
            self.output_scope = self.device_scope(
                self.embedding_scope.virtualGraph, pipeline_stage, execution_phase, "Output")
        else:
            self.output_scope = None

        self.total_pipeline_stages = pipeline_stage + 1
        self.total_execution_phases = execution_phase + 1

    @property
    def total_ipus(self):
        return self.encoder_layer_ipu(self.config.num_layers - 1) + 1

    def encoder_scope(self, layer_index):
        ipu = self.encoder_layer_ipu(layer_index)
        execution_phase = ipu + self.execution_phase_precompute_offset
        logger.debug(f"Encoder Layer {layer_index} -> IPU {ipu}")
        return self.device_scope(ipu, ipu, execution_phase, f"Layer{layer_index}")

    def _encoder_layer_ipu_offset(self, layer_index):
        encoder_index = 0
        if len(self.config.layers_per_ipu) == 1:
            encoder_index = layer_index // self.config.layers_per_ipu[0]
        else:
            for ipu, num_layers in enumerate(self.config.layers_per_ipu):
                layer_index -= num_layers
                if layer_index < 0:
                    encoder_index = ipu
                    break
        return encoder_index

    def encoder_layer_ipu(self, layer_index):
        return self._encoder_layer_ipu_offset(layer_index) + self.config.encoder_start_ipu

    def should_checkpoint(self, layer_index):
        '''Only checkpoint tensors that are not to be copied to the next pipelineStage'''
        if not self.config.recompute_checkpoint_every_layer:
            return False

        encoder_index = self._encoder_layer_ipu_offset(layer_index)
        if len(self.config.layers_per_ipu) == 1:
            layers = self.config.layers_per_ipu[0]
            layer_index -= encoder_index*layers
        else:
            layers = self.config.layers_per_ipu[encoder_index]
            layer_index -= sum(self.config.layers_per_ipu[:encoder_index])
        return layer_index < (layers - 1)

    def set_available_memory_proportion(self, x):
        if self.config.use_default_available_memory_proportion:
            return x

        if len(self.config.available_memory_proportion) == 1:
            amp = self.config.available_memory_proportion[0]
        else:
            vgraph = self.builder.getVirtualGraph()
            amp = self.config.available_memory_proportion[vgraph - self.config.encoder_start_ipu]

        self.builder.setAvailableMemoryProportion(x, amp)
        return x

    def build_graph(self, indices, positions, segments, masks=None):
        # Embedding
        with self.builder.nameScope("Embedding"):
            x = self.embedding(indices, positions, segments)

        # This forces the masks to be streamed on to the IPU before the encoder layers.
        # Allowing the communication to be better overlapped with compute as the
        # compute dominant encoder IPUs will not participate in any streamCopies.
        if masks is not None:
            with self.embedding_split_scope:
                masks = [self.detach(mask) for mask in masks]

        # Encoder Layers
        for i in range(self.config.num_layers):
            with self.encoder_scope(i):
                with self.builder.nameScope("Attention"):
                    x = self.attention(x, masks)

                with self.builder.nameScope("FF"):
                    x = self.feed_forward(x)

                if self.should_checkpoint(i):
                    x = self.builder.checkpointOutput([x])[0]
        outputs = []

        # PreTraining tasks
        if self.config.task in ("NSP", "PRETRAINING"):
            with self.nsp_scope:
                outputs.append(self.nsp_head(x))

        if self.config.task == "PRETRAINING":
            with self.cls_scope:
                if self.config.no_cls_layer:
                    predictions = self.builder.aiOnnx.identity([x])
                else:
                    predictions = self.lm_prediction_head(x)
            with self.mlm_scope:
                outputs = [self.projection(predictions)] + outputs

        # Fine Tuning tasks
        if self.config.task == "SQUAD":
            with self.squad_scope:
                squad_outputs = self.squad_projection(x)

            if self.output_scope:
                with self.output_scope:
                    outputs += [self.detach(tensor)
                                for tensor in squad_outputs]
            else:
                outputs += squad_outputs

        if self.config.task == "MRPC":
            # TODO: Implement this: T11026
            raise NotImplementedError()

        return tuple(outputs)

    def norm(self, input_x):
        gamma = self.constant_init_tensor(
            self.config.dtype, (self.config.hidden_size,), 1, "Gamma")
        beta = self.constant_init_tensor(
            self.config.dtype, (self.config.hidden_size,), 0, "Beta")

        outs = self.builder.aiGraphcore.groupnormalization(
            [input_x, gamma, beta], 1, self.config.layer_norm_eps)
        return outs[0]

    def dropout(self, input_x):
        if not self.config.no_dropout:
            return self.builder.aiOnnx.dropout([input_x], 1, self.config.dropout_prob)[0]
        return input_x

    def leaky_relu(self, input_x, alpha):
        """
            This function implements the leaky relu activation function.
            The mathematical function is:
            Leaky_Relu(x) = Relu(x) - alpha*Relu(-x)
        """
        alpha_t = self.builder.aiOnnx.constant(
            np.asarray([alpha], dtype=self.config.dtype)
        )
        result_plus = self.builder.aiOnnx.relu([input_x])
        minus_x = self.builder.aiOnnx.neg([input_x])
        result_minus = self.builder.aiOnnx.relu([minus_x])
        result_minus = self.builder.aiOnnx.mul([alpha_t, result_minus])
        result = self.builder.aiOnnx.sub([result_plus, result_minus])
        return result

    def simplified_gelu(self, input_x):
        """
            Simpler implementation of the GELU based on the sigmoid.
            Coming from the original Gelu paper (https://arxiv.org/abs/1606.08415).
        """
        scale = self.builder.aiOnnx.constant(
            np.asarray([1.702], dtype=self.config.dtype))
        result = self.builder.aiOnnx.mul([scale, input_x])
        result = self.builder.aiOnnx.sigmoid([result])
        result = self.builder.aiOnnx.mul([input_x, result])
        return result

    def intermediate_activation_function(self, input_x):
        if self.config.activation_type == 'Gelu':
            return self.builder.aiGraphcore.gelu([input_x])
        elif self.config.activation_type == 'SGelu':
            return self.simplified_gelu(input_x)
        elif self.config.activation_type == 'LRelu':
            return self.leaky_relu(input_x, alpha=self.config.relu_leak)
        else:
            return self.builder.aiOnnx.relu([input_x])

    def feed_forward(self, input_x):
        # If using `split_linear_layers` num_splits should make each matmul of size [hidden, hidden]
        num_splits = self.config.ff_size // self.config.hidden_size
        with self.builder.nameScope("1"):
            weight1 = self.normal_init_tensor(self.config.dtype,
                                              [self.config.hidden_size,
                                               self.config.ff_size],
                                              0, 0.02,
                                              "W")
            bias1 = self.constant_init_tensor(self.config.dtype,
                                              (self.config.ff_size,),
                                              0,
                                              "B")
            x = self.builder.aiOnnx.matmul([input_x, weight1])
            self.set_available_memory_proportion(x)
            if self.config.split_linear_layers:
                self.builder.setSerializeMatMul({x},
                                                'output_channels',
                                                num_splits,
                                                keep_precision=True)
            x = self.builder.aiOnnx.add([x, bias1])

        x = self.intermediate_activation_function(x)

        with self.builder.nameScope("2"):
            weight2 = self.normal_init_tensor(self.config.dtype,
                                              [self.config.ff_size,
                                               self.config.hidden_size],
                                              0, 0.02,
                                              "W")
            bias2 = self.constant_init_tensor(self.config.dtype,
                                              (self.config.hidden_size,),
                                              0,
                                              "B")
            x = self.builder.aiOnnx.matmul([x, weight2])
            self.set_available_memory_proportion(x)
            if self.config.split_linear_layers:
                self.builder.setSerializeMatMul({x},
                                                'reducing_dim',
                                                num_splits,
                                                keep_precision=True)
            x = self.builder.aiOnnx.add([x, bias2])

        # google-research/bert puts dropout here
        x = self.dropout(x)
        x = self.builder.aiOnnx.add([input_x, x])
        x = self.norm(x)
        return x

    def detach(self, input_x, pass_through_creation=1):
        if self.config.inference:
            return input_x
        return self.builder.customOp(opName="Detach",
                                     opVersion=1,
                                     domain="ai.graphcore",
                                     inputs=[input_x],
                                     attributes={
                                         "pass_through_creation": pass_through_creation
                                     })[0]

    def generate_simplified_periodic_pos_data(self, dtype, shape, scale=4):
        def value(x, y):
            return .02/.707*np.cos(2*scale*np.pi*x*y/shape[1])
        X, Y = np.mgrid[:shape[0], :shape[1]]
        return np.vectorize(value)(X, Y,).astype(dtype)

    def generate_transformer_periodic_pos_data(self, dtype, shape, min_timescale=1.0, max_timescale=1.0e4):
        """
        Periodic position initialiser, from 3.5 of "Attention is All You Need". Adapted from:
        https://github.com/tensorflow/models/tree/master/official/transformer/v2
        """
        position = np.arange(0, shape[0], dtype=dtype)
        num_timescales = shape[1] // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))

        hidden_idx = np.arange(0, num_timescales, dtype=dtype)
        inv_timescales = min_timescale * np.exp(
            hidden_idx * -log_timescale_increment)

        expanded_pos = np.expand_dims(position, 1)
        expanded_ts = np.expand_dims(inv_timescales, 0)
        scaled_time = expanded_pos * expanded_ts

        signal = np.concatenate(
            [np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        return signal

    def get_model_embeddings(self):
        embedding_dict = None
        positional_dict = None
        if self.config.host_embedding in ("ALL", "WORD", "MERGE"):
            with self.builder.nameScope("Embedding"):
                embedding_dict = self.normal_init_data(self.config.dtype,
                                                       (self.config.vocab_length, self.config.hidden_size),
                                                       0, 0.02,
                                                       "Embedding_Dict")
                if self.config.host_embedding in ("ALL", "MERGE"):
                    positional_dict = self.normal_init_data(self.config.dtype,
                                                            (self.config.max_positional_length, self.config.hidden_size),
                                                            0, 0.02,
                                                            "Positional_Dict")
        return embedding_dict, positional_dict


    def _split_word_embedding_initializer(self):
        def get_split(idx, full_t):
            num_splits = self.config.embedding_serialization_vocab_steps
            vocab_axis = full_t.shape.index(self.config.vocab_length)
            return np.split(full_t, num_splits, axis=vocab_axis)[idx]

        num_splits = self.config.embedding_serialization_vocab_steps
        embedding_dict = self.initializers["Embedding/Embedding_Dict"]

        embedding_dict_split = {}
        for i in range(num_splits):
            embedding_dict_split[f"Embedding/Embedding_Dict/split{i}"] = get_split(i, embedding_dict)

        self.initializers.update(embedding_dict_split)
        del self.initializers["Embedding/Embedding_Dict"]


    def word_embedding_serialized(self, indices, num_splits):
        def mask_input_indices(x_in, i):
            split_input_dim = self.config.vocab_length // num_splits
            x_split = self.builder.aiOnnx.sub([
                x_in,
                self.constant_tensor(i * split_input_dim,
                                     np.uint32)
            ])
            mask = self.builder.aiOnnx.less([
                x_split,
                self.constant_tensor(split_input_dim,
                                     np.uint32)
            ])
            mask = self.detach(mask)
            masked_indices = self.builder.aiOnnx.mul([x_split,
                                                      self.builder.aiOnnx.cast([mask], "UINT32")])
            return masked_indices, mask

        if self.initializers and self.config.embedding_serialization_vocab_steps > 1:
            self._split_word_embedding_initializer()

        x_sum = None
        for i in range(num_splits):
            masked_indices, mask = mask_input_indices(indices, i)
            x = self.gather(masked_indices,
                            self.config.vocab_length // num_splits,
                            f"Embedding_Dict/split{i}")
            fp_mask = self.builder.aiOnnx.cast([mask], self.config.popart_dtype)
            fp_mask = self.builder.aiOnnx.unsqueeze([fp_mask], [1])
            x = self.builder.aiOnnx.mul([x, fp_mask])

            if x_sum:
                x_sum = self.builder.aiOnnx.add([x, x_sum])
            else:
                x_sum = x
        return x_sum


    def embedding(self, indices, positions, segments):
        with self.embedding_scope:
            if self.config.task != "PRETRAINING" and self.config.embedding_serialization_vocab_steps > 1:
                x = self.word_embedding_serialized(indices, self.config.embedding_serialization_vocab_steps)
            else:
                x = self.gather(indices,
                                self.config.vocab_length,
                                "Embedding_Dict")

        with self.embedding_split_scope:

            segments_onehot = self.builder.aiOnnx.onehot([
                segments,
                self.constant_tensor(2, dtype=np.int32),
                self.constant_tensor([0, 1], dtype=self.config.dtype)])
            segments_weights = self.normal_init_tensor(
                self.config.dtype,
                [2, self.config.hidden_size],
                0, 0.02, "Segment_Dict")
            x_seg = self.builder.aiOnnx.matmul(
                [segments_onehot, segments_weights])

            if self.config.host_embedding != "MERGE":
                x_pos = self.gather(positions,
                                    self.config.max_positional_length,
                                    "Positional_Dict")
                x = self.builder.aiOnnx.add([x, x_pos])
            x = self.builder.aiOnnx.add([x, x_seg])

            # When outlining is enabled, under certain situations, the `add` above resolves
            # to an AddLhsInPlace, which then causes the output to be laid out incorrectly
            # for SQuAD. This workaround ensures it stays as an AddRhsInPlace.
            self.builder.setInplacePreferences(x, {"AddRhsInplace": 1000.0})

            x = self.norm(x)
            x = self.dropout(x)
        return x

    def gather(self, indices, embedding_size, name):
        if self.config.host_embedding in ("ALL", "WORD", "MERGE") and name == "Embedding_Dict":
            return indices
        if self.config.host_embedding in ("ALL", "MERGE") and name == "Positional_Dict":
            return indices
        if name.startswith("Embedding_Dict") and self.config.task == "PRETRAINING":
            # Important that the tied gather/matmul weight with transpose before the gather.
            # This will ensure it matches the custom_ops/tied_gather_pattern.
            embedding_dict = self.normal_init_tensor(
                self.config.dtype,
                (self.config.hidden_size, embedding_size),
                0, 0.02,
                name)
            self.embedding_dict = embedding_dict

            if self.config.inference:
                embedding_dict = self.builder.customOp(opName="PreventConstFolding",
                                                       opVersion=1,
                                                       domain="ai.graphcore",
                                                       inputs=[embedding_dict],
                                                       attributes={})[0]
            embedding_dict = self.builder.aiOnnx.transpose([embedding_dict])
        else:
            embedding_dict = self.normal_init_tensor(
                self.config.dtype,
                (embedding_size, self.config.hidden_size),
                0, 0.02,
                name)

        x = self.builder.aiOnnx.gather([embedding_dict, indices])

        if name == "Embedding_Dict" and not self.config.update_embedding_dict:
            x = self.detach(x)
        return x

    def attention_mask(self, masks):
        """
        Create a mask tensor that has -1000 in positions to be masked and 0 otherwise.
        If the task is MLM or PRETRAINING:
            masks[0] is the index that masking starts in the mask_tokens
            masks[1] is the index that masking starts in the rest of the sequence
        Otherwise
            masks[0] is the index that masking starts in the rest of the sequence
        When use_packed_sequence_format is turned on:
            model.input["input_mask"] is used to create the block-diagonal mask which
            prevents cross-contamination between sequences in a pack
        Example:
            Task: PRETRAINING
            masks: [2, 5]
            mask_tokens: 4
            returns: [0,0,-1000.0, -1000.0, 0, -1000.0, ...]
        """
        if self.execution_mode == ExecutionMode.PHASED:
            mask_idx = self.builder.getExecutionPhase() % 2
            additional_scopes = [
                self.builder.recomputeOutput(popart.RecomputeType.Checkpoint),
                self.builder.outputTensorLocation(popart.TensorLocation.OnChip)
            ]
        else:
            mask_idx = self.builder.getVirtualGraph() if self.builder.hasVirtualGraph() else None
            additional_scopes = None

        if mask_idx in self.masks:
            return self.masks[mask_idx]

        mask_scope = self.device_scope(mask_idx,
                                       self.builder.getPipelineStage() if self.builder.hasPipelineStage() else None,
                                       mask_idx,
                                       "Mask",
                                       additional_scopes=additional_scopes)
        with mask_scope:
            if not self.config.use_packed_sequence_format:
                base_value = np.arange(self.config.sequence_length)
                base = self.constant_tensor(base_value, np.uint32, "mask_sequence")
                if self.config.task == "PRETRAINING":
                    # Mask tokens mask
                    mmask = self.builder.aiOnnx.less([base, masks[0]])
                    # No constexpr for greater. Create as const instead
                    _mask = self.constant_tensor(np.greater_equal(
                        base_value, self.config.max_lm_predictions), np.bool)
                    mmask = self.builder.aiOnnx.logical_or([mmask, _mask])
                    # Sequence mask
                    smask = self.builder.aiOnnx.less([base, masks[1]])
                    final_mask = self.builder.aiOnnx.logical_and([mmask, smask])
                else:
                    final_mask = self.builder.aiOnnx.less([base, masks[0]])

                final_mask = self.builder.aiOnnx.cast(
                    [final_mask], self.config.popart_dtype)
                final_mask = self.builder.aiOnnx.sub(
                    [final_mask, self.constant_tensor(1.0, self.config.dtype)])
                final_mask = self.builder.aiOnnx.mul(
                    [final_mask, self.constant_tensor(1000.0, self.config.dtype)])
                final_mask = self.builder.reshape_const(
                    self.builder.aiOnnx,
                    [final_mask],
                    [self.config.micro_batch_size, 1, 1, self.config.sequence_length])
            else:
                final_mask = self.builder.aiOnnx.less([base, masks[0]])
            final_mask = self.builder.aiOnnx.cast(
                [final_mask], self.config.popart_dtype)
            final_mask = self.builder.aiOnnx.sub(
                [final_mask, self.constant_tensor(1.0, self.config.dtype)])
            final_mask = self.builder.aiOnnx.mul(
                [final_mask, self.constant_tensor(1000.0, self.config.dtype)])
            final_mask = self.builder.reshape_const(
                self.builder.aiOnnx,
                [final_mask],
                [self.config.micro_batch_size, 1, 1, self.config.sequence_length])
            # TODO: This shouldn't be needed. No Variables on this path.
            final_mask = self.detach(final_mask)
            self.masks[mask_idx] = final_mask
        return final_mask


    def qkv_weights(self):
        if self.config.split_qkv:
            weights = []
            biases = []
            full_t = self.initializers.get(
                self.builder.getNameScope("QKV"), None)
            for idx, name in enumerate("QKV"):
                if full_t is not None:
                    long_name = self.builder.getNameScope(name)
                    self.initializers[long_name] = np.split(full_t, 3, axis=1)[idx]
                weights.append(self.normal_init_tensor(
                    self.config.dtype,
                    [self.config.hidden_size, self.config.hidden_size],
                    0, 0.02,
                    name))
                if self.config.attention_bias:
                    full_b = self.initializers.get(self.builder.getNameScope("QKV_Bias"), None)
                    if full_b is not None:
                        idx = "QKV".index(name)
                        long_name = self.builder.getNameScope(name) + '_Bias'
                        self.initializers[long_name] = np.split(full_b, 3, axis=0)[idx]
                    biases.append(self.constant_init_tensor(
                        self.config.dtype,
                        [self.config.hidden_size],
                        0,
                        name+'_Bias'))

            if self.config.attention_bias:
                qkv_biases = self.builder.aiOnnx.concat(biases, axis=0)
            else:
                qkv_biases = None
            qkv = self.builder.aiOnnx.concat(weights, axis=1)
        else:
            qkv = self.normal_init_tensor(
                self.config.dtype,
                [self.config.hidden_size, 3 * self.config.hidden_size],
                0, 0.02,
                "QKV")
            if self.config.attention_bias:
                qkv_biases = self.constant_init_tensor(
                    self.config.dtype,
                    (3 * self.config.hidden_size,),
                    0,
                    "QKV_Bias")
            else:
                qkv_biases = None
        return qkv, qkv_biases


    def attention(self, input_x, masks=None):
        qkv_weights, qkv_biases = self.qkv_weights()
        qkv = self.builder.aiOnnx.matmul([input_x, qkv_weights])
        self.set_available_memory_proportion(qkv)
        if self.config.split_linear_layers:
            self.builder.setSerializeMatMul({qkv}, 'output_channels', 3, True)
        if qkv_biases is not None:
            qkv = self.builder.aiOnnx.add([qkv, qkv_biases])

        x = self.attention_onnx(qkv, masks)

        projection_weights = self.normal_init_tensor(
            self.config.dtype,
            [self.config.hidden_size, self.config.hidden_size],
            0, 0.02,
            "Out")
        x = self.builder.aiOnnx.matmul([x, projection_weights])
        self.set_available_memory_proportion(x)

        if self.config.attention_bias:
            projection_bias = self.constant_init_tensor(
                self.config.dtype,
                (self.config.hidden_size,),
                0,
                "Out_Bias")
            x = self.builder.aiOnnx.add([x, projection_bias])

        x = self.dropout(x)
        x = self.builder.aiOnnx.add([input_x, x])
        x = self.norm(x)
        return x


    def attention_onnx(self, qkv, masks):
        comb_shape = [self.config.micro_batch_size, self.config.sequence_length,
                      self.config.attention_heads, self.config.qkv_length]

        if isinstance(qkv, list):
            split_qkv = qkv
        else:
            split_qkv = self.builder.aiOnnx.split(
                [qkv],
                num_outputs=3,
                axis=1,
                split=[self.config.hidden_size]*3,
                debugContext="QKV_Split")

        def extract_heads(tensor, transpose=False):
            tensor = self.builder.reshape_const(
                self.builder.aiOnnx, [tensor], comb_shape)
            perm = [0, 2, 1, 3] if not transpose else [0, 2, 3, 1]
            return self.builder.aiOnnx.transpose([tensor], perm=perm)

        q, kt, v = [extract_heads(t, i == 1) for i, t in enumerate(split_qkv)]

        # Attention calculation
        with self.builder.nameScope('Z'):
            x = self.builder.aiOnnx.matmul([q, kt])
            self.set_available_memory_proportion(x)

            c = self.constant_tensor(
                1 / np.sqrt(self.config.qkv_length), self.config.dtype)
            x = self.builder.aiOnnx.mul([x, c])

            if not self.config.no_mask or masks is not None:
                mask = self.attention_mask(masks)
                x = self.builder.aiOnnx.add([x, mask], "ApplyMask")

            x = self.builder.aiOnnx.softmax([x], axis=-1)

            if not self.config.no_attn_dropout:
                x = self.dropout(x)

            # x[micro_batch_size, attention_heads, sequence_length, sequence_length] * v[micro_batch_size, attention_heads, sequence_length, qkv_length]
            z = self.builder.aiOnnx.matmul([x, v])
            self.set_available_memory_proportion(z)

            # [micro_batch_size, attention_heads, sequence_length, qkv_length] -> [micro_batch_size, sequence_length, attention_heads, qkv_length]
            z = self.builder.aiOnnx.transpose([z], perm=[0, 2, 1, 3])
            # [micro_batch_size, sequence_length, attention_heads, qkv_length] -> [micro_batch_size*sequence_length, attention_heads*qkv_length]
            z = self.builder.reshape_const(self.builder.aiOnnx, [z], [
                                           self.config.sequence_length * self.config.micro_batch_size, self.config.hidden_size])
        return z

    def projection(self, input_x):
        x = self.builder.reshape_const(self.builder.aiOnnx, [input_x], [
                                       self.config.micro_batch_size, self.config.sequence_length, self.config.hidden_size])

        if self.config.use_packed_sequence_format:
            # MLM tokens can be at arbitrary positions in the sequence
            x = packed_bert_utils.mlm_projection_gather_indexes(self, x)

        else:
            # MLM tokens have been pre-arranged to the front of the sequence
            x = self.builder.aiOnnxOpset9.slice([x], axes=[1], starts=[
                0], ends=[self.config.max_lm_predictions])
            x = self.builder.reshape_const(self.builder.aiOnnx, [x], [
                                        self.config.micro_batch_size * self.config.max_lm_predictions, self.config.hidden_size])

        weight = self.embedding_dict

        # Move the weight to the current pipeline stage
        if weight in self.tensors[self.embedding_scope.pipelineStage]:
            embedding_stage = self.embedding_scope.pipelineStage
            self.tensors[embedding_stage].remove(weight)
            self._add_to_tensor_map(weight)

        x = self.builder.aiOnnx.matmul([x, weight])
        num_splits = self.config.embedding_serialization_vocab_steps
        self.builder.setSerializeMatMul(
            {x}, 'output_channels', num_splits, True)

        if self.config.projection_bias:
            bias = self.constant_init_tensor(self.config.dtype, (self.config.vocab_length,), 0, "ProjectionB")
            x = self.builder.aiOnnx.add([x, bias])

        x = self.builder.reshape_const(self.builder.aiOnnx, [x], [
                                       self.config.micro_batch_size, self.config.max_lm_predictions, self.config.vocab_length])
        return x

    def squad_projection(self, input_x):
        weight = self.normal_init_tensor(self.config.dtype,
                                         [self.config.hidden_size, 2],
                                         0, 0.02,
                                         "SquadW")
        bias = self.constant_init_tensor(self.config.dtype, (2,), 0, "SquadB")
        x = self.builder.aiOnnx.gemm([input_x, weight, bias])

        if self.config.inference and self.config.squad_single_output:
            return [x]

        # x.shape: [micro_batch_size * sequence_length, 2]
        start_logits = self.builder.aiOnnxOpset9.slice(
            [x], axes=[1], starts=[0], ends=[1])
        end_logits = self.builder.aiOnnxOpset9.slice(
            [x], axes=[1], starts=[1], ends=[2])

        start_logits = self.builder.reshape_const(
            self.builder.aiOnnx,
            [start_logits], [self.config.micro_batch_size, self.config.sequence_length], debugContext="answer_start")
        end_logits = self.builder.reshape_const(
            self.builder.aiOnnx,
            [end_logits], [self.config.micro_batch_size, self.config.sequence_length], debugContext="answer_end")

        return start_logits, end_logits

    def pooler(self, pooler_input):
        """
        Take the [CLS] token as a sentence embedding (assuming it's already been
        fine-tuned), then run a FC layer with tanh activation
        """
        if self.config.use_packed_sequence_format:
            # The [CLS] tokens can occur anywhere in the sequence
            pooler_input = packed_bert_utils.pooler_gather_indexes(self, pooler_input)

        else:
            # The [CLS] token occurs at the start of the token
            pooler_input = self.builder.aiOnnxOpset9.slice(
                [pooler_input], axes=[1], starts=[self.config.max_lm_predictions], ends=[self.config.max_lm_predictions + 1]
            )
            # This reshape is doing the job of a squeeze, but allows for in-place operation.
            pooler_input = self.builder.reshape_const(self.builder.aiOnnx, [pooler_input], [
                self.config.micro_batch_size, self.config.hidden_size])

        weight = self.normal_init_tensor(
            self.config.dtype,
            [self.config.hidden_size, self.config.hidden_size],
            0,
            0.02,
            "PoolW",
        )
        bias = self.constant_init_tensor(
            self.config.dtype, (self.config.hidden_size,), 0, "PoolB"
        )
        x = self.builder.aiOnnx.matmul([pooler_input, weight])
        x = self.builder.aiOnnx.add([x, bias])

        return self.builder.aiOnnx.tanh([x])

    def nsp_head(self, input_x):
        x = self.builder.reshape_const(self.builder.aiOnnx, [input_x], [
                                       self.config.micro_batch_size, self.config.sequence_length, self.config.hidden_size])

        x = self.pooler(x)

        cls_weight = self.normal_init_tensor(
            self.config.dtype, [self.config.hidden_size, 2], 0, 0.02, "NspW"
        )
        cls_bias = self.constant_init_tensor(
            self.config.dtype, (2,), 0, "NspB")
        x = self.builder.aiOnnx.matmul([x, cls_weight])
        x = self.builder.aiOnnx.add([x, cls_bias])
        return x

    def lm_prediction_head(self, input_x):
        dense_weight = self.normal_init_tensor(self.config.dtype,
                                               [self.config.hidden_size,
                                                self.config.hidden_size],
                                               0,
                                               0.02,
                                               "LMPredictionW")

        dense_bias = self.constant_init_tensor(
            self.config.dtype, (self.config.hidden_size,), 0, "LMPredictionB")

        x = self.builder.aiOnnx.matmul([input_x, dense_weight])
        x = self.builder.aiOnnx.add([x, dense_bias])

        x = self.intermediate_activation_function(x)
        x = self.norm(x)
        return x


def get_model(config, mode, block=None, initializers=None):
    # Specifying ai.onnx opset9 for the slice syntax
    builder = popart.Builder(opsets={
        "ai.onnx": 9,
        "ai.onnx.ml": 1,
        "ai.graphcore": 1
    })

    if mode == ExecutionMode.PHASED:
        scope_provider = ScopeProvider(phased_execution_type=config.phased_execution_type)
        if not block:
            from phased_execution.bert_phased import BertModel
            return BertModel(config,
                             builder=builder,
                             initializers=initializers,
                             scope_provider=scope_provider)

        if block.lower() == 'embedding':
            from phased_execution.bert_layers_serialised import BertEmbedding
            return BertEmbedding(config.vocab_length,
                                 config.hidden_size,
                                 config.sequence_length,
                                 config.max_positional_length,
                                 config.embedding_serialization_vocab_steps,
                                 config.layer_norm_eps,
                                 not config.no_dropout,
                                 config.dropout_prob,
                                 mode,
                                 config.dtype,
                                 not config.update_embedding_dict,
                                 weight_transposed=False,
                                 builder=builder,
                                 scope_provider=scope_provider)

        if block.lower() == 'attention':
            from phased_execution.bert_layers import Attention
            attention_params = {
                'input_size': config.hidden_size,
                'hidden_size': config.hidden_size,
                'num_heads': config.attention_heads,
                'serialize_matmul': config.split_linear_layers,
                'available_memory_proportion': config.available_memory_proportion,
                'epsilon': config.layer_norm_eps,
                'dropout': not config.no_dropout,
                'dropout_prob': config.dropout_prob,
                'attn_dropout': not config.no_attn_dropout,
                'attn_dropout_prob': config.attn_dropout_prob,
                'micro_batch_size': config.micro_batch_size,
                'sequence_length': config.sequence_length,
                'dtype': config.dtype,
                'task': config.task,
                'num_mask_tokens': config.mask_tokens,
                'split_qkv': config.split_qkv,
                'attention_bias': config.attention_bias
            }
            return Attention('Attention', **attention_params, builder=builder, scope_provider=scope_provider)

        if block.lower() == 'feedforward':
            from phased_execution.bert_layers import FeedForward
            return FeedForward('FF',
                               config.hidden_size,
                               config.ff_size,
                               dropout=not config.no_dropout,
                               dropout_prob=config.dropout_prob,
                               epsilon=config.layer_norm_eps,
                               intermediate_act_func=config.activation_type,
                               dtype=config.dtype,
                               alpha=config.relu_leak,
                               serialize_matmul=config.split_linear_layers,
                               builder=builder,
                               scope_provider=scope_provider)
    else:
        return Bert(config,
                    builder=builder,
                    initializers=initializers,
                    execution_mode=mode)
