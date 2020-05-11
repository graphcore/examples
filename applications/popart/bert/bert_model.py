# Copyright 2019 Graphcore Ltd.
import os
import ctypes
import popart
import numpy as np
from scipy.stats import truncnorm
from typing import NamedTuple, List, Optional
from functools import reduce
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class BertConfig(NamedTuple):
    batch_size: int = 1
    sequence_length: int = 128
    max_positional_length: int = 512

    # Choices: "DEFAULT", "TRANSFORMER", "SIMPLIFIED"
    positional_embedding_init_fn: str = "DEFAULT"
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

    inference: bool = False

    num_layers: int = 2
    layers_per_ipu: int = 2

    no_dropout: bool = False
    no_attn_dropout: bool = False
    dropout_prob: float = 0.1
    attn_dropout_prob: float = 0.1

    layer_norm_eps: float = 0.001

    # Choices: PRETRAINING (MLM + NSP), SQUAD
    task: str = "PRETRAINING"

    # Choices: attention
    custom_ops: List[str] = []

    # This option serializes all matmul layers to multiples
    # {N, hidden_size} x {hidden_size, hidden_size}.
    # This is required for sequence length 384.
    split_linear_layers: bool = False

    # Try and fit the model onto fewer IPUs. Intended for inference modes:
    squeeze_model: bool = False

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

    # In PRETRAINING this sets how many steps to serialise both the
    # embedding and projection
    projection_serialization_steps: int = 5

    update_embedding_dict: bool = True

    use_default_available_memory_proportion: bool = False

    no_cls_layer: bool = False

    max_matmul_memory: int = 40000

    @property
    def available_memory_proportion(self):
        return self.max_matmul_memory / (2**18)


class ExecutionMode(str, Enum):
    DEFAULT = "DEFAULT"
    PIPELINE = "PIPELINE"


class DeviceScope(object):
    def __init__(self,
                 builder,
                 execution_mode=ExecutionMode.DEFAULT,
                 virtualGraph=None,
                 pipelineStage=None,
                 nameScope=None,
                 additional_scopes=None):
        self.builder = builder
        self.execution_mode = execution_mode
        self.virtualGraph = virtualGraph
        self.pipelineStage = pipelineStage
        self.nameScope = nameScope
        self.additional_scopes = additional_scopes or []

    def __enter__(self):
        self.stack = ExitStack()
        if self.virtualGraph is not None:
            self.stack.enter_context(self.builder.virtualGraph(self.virtualGraph))
        # Adding pipelineStage attributes can have side effects on the schedule.
        # even if it's disabled. FIXME: T13889
        if self.execution_mode == ExecutionMode.PIPELINE\
                and self.pipelineStage is not None:
            self.stack.enter_context(
                self.builder.pipelineStage(self.pipelineStage))

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
        self.tensors = defaultdict(list)

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
                    logger.warn(
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

    def device_scope(self, virtualGraph=None, pipelineStage=None, nameScope=None, additional_scopes=None):
        return DeviceScope(self.builder, self.execution_mode, virtualGraph, pipelineStage, nameScope, additional_scopes)

    def _add_to_tensor_map(self, tensor):
        if self.builder.hasPipelineStage():
            pipeline_stage = self.builder.getPipelineStage()
            self.tensors[pipeline_stage].append(tensor)
        else:
            self.tensors[0].append(tensor)


class Bert(Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.dropout_modifier = 256

        # This is the TensorId for the shared embedding & projection
        self.embedding_dict = None

        # This dict[ipu,TensorId] reuses any mask already generated on an IPU/pipeline stage
        # TODO: Should recompute instead?
        self.masks = {}

        self.init_device_placement()

    def init_device_placement(self):
        ''' Create a DeviceScope for each layer, ie Embedding, SQUAD, NSP
        '''
        self.layer_offset = 1
        if not self.config.inference:
            self.layer_offset += 1
        if self.config.squeeze_model:
            self.layer_offset -= 1

        # Embedding
        self.embedding_scope = self.device_scope(0, 0)
        ipu = max(self.layer_offset - 1, 0)
        self.embedding_split_scope = self.device_scope(ipu, ipu)
        # Transformer Layers
        pipeline_stage = ((self.config.num_layers - 1) // self.config.layers_per_ipu) + self.layer_offset + 1
        # Task Layers
        if self.config.task in ("NSP", "PRETRAINING"):
            self.nsp_scope = self.device_scope(self.embedding_split_scope.virtualGraph, pipeline_stage, "NSP")
            self.cls_scope = self.device_scope(self.embedding_split_scope.virtualGraph, pipeline_stage, "CLS")
        if self.config.task == "PRETRAINING":
            if (self.layer_offset - 1) != 0:
                pipeline_stage += 1
            self.mlm_scope = self.device_scope(self.embedding_scope.virtualGraph, pipeline_stage, "MLM")
        if self.config.task == "SQUAD":
            ipu = self.embedding_scope.virtualGraph
            if self.config.inference:
                pipeline_stage -= 1
                ipu = pipeline_stage
            self.squad_scope = self.device_scope(ipu, pipeline_stage, "Squad")

        # Scope to place all IO on first IPU for inference:
        if self.config.inference:
            pipeline_stage += 1
            self.output_scope = self.device_scope(self.embedding_scope.virtualGraph, pipeline_stage, "Output")
        else:
            self.output_scope = None

        self.total_pipeline_stages = pipeline_stage + 1

    def encoder_scope(self, layer_index):
        ipu = (layer_index // self.config.layers_per_ipu) + self.layer_offset

        if self.config.squeeze_model and self.config.inference and self.config.hidden_size == 1024:
            # Special case to help squeeze BERT LARGE 128 inference onto 4 IPUs instead of 7
            # by moving some of the transformer layers around. New intended split is:
            # 3 Encoders -> IPU0
            # 7 Encoders -> IPU1
            # 7 Encoders -> IPU2
            # 7 Encoders -> IPU3
            if layer_index in [3, 4, 5]:
                ipu = 1
            if layer_index in [10, 11]:
                ipu = 2
            if layer_index in [17]:
                ipu = 3

        logger.debug(f"Encoder Layer {layer_index} -> IPU {ipu}")
        return self.device_scope(ipu, ipu, f"Layer{layer_index}")

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
            if self.config.split_linear_layers:
                self.builder.setSerializeMatMul({x},
                                                'output_channels',
                                                num_splits,
                                                keep_precision=True)
            if not self.config.use_default_available_memory_proportion:
                self.builder.setAvailableMemoryProportion(
                    x, self.config.available_memory_proportion)
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
            if self.config.split_linear_layers:
                self.builder.setSerializeMatMul({x},
                                                'reducing_dim',
                                                num_splits,
                                                keep_precision=True)
            if not self.config.use_default_available_memory_proportion:
                self.builder.setAvailableMemoryProportion(
                    x, self.config.available_memory_proportion)
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

    def get_embedding_data(self, dtype, shape, init_fn, debug_name=""):
        # Unless specifcally set, fall back to normal tensor initialisation
        if init_fn not in ("TRANSFORMER", "SIMPLIFIED"):
            return self.normal_init_data(dtype, shape, 0, 0.02, debug_name)

        data = self.initializers.get(
            self.builder.getNameScope(debug_name), None)
        if data is None:
            if init_fn == "TRANSFORMER":
                data = self.generate_transformer_periodic_pos_data(
                    dtype, shape)
            else:
                data = self.generate_simplified_periodic_pos_data(dtype, shape)
        else:
            if np.any(data.shape != np.array(shape)):
                raise RuntimeError(f"Initializer {self.builder.getNameScope(debug_name)} does not match shapes. \n"
                                   f" Provided {data.shape}. Required {shape}")
        return data

    def embedding_init_tensor(self, dtype, shape, init_fn, debug_name=""):
        # Unless specifcally set, fall back to normal tensor initialisation
        data = self.get_embedding_data(dtype, shape, init_fn, debug_name)
        tensor = self.builder.addInitializedInputTensor(data, debug_name)
        self._add_to_tensor_map(tensor)
        return tensor

    def get_model_embeddings(self):
        embedding_dict = None
        positional_dict = None
        if self.config.host_embedding in ("ALL", "WORD", "MERGE"):
            with self.builder.nameScope("Embedding"):
                embedding_dict = self.get_embedding_data(self.config.dtype,
                                                         (self.config.vocab_length, self.config.hidden_size),
                                                         "DEFAULT",
                                                         "Embedding_Dict")
                if self.config.host_embedding in ("ALL", "MERGE"):
                    positional_dict = self.get_embedding_data(self.config.dtype,
                                                              (self.config.max_positional_length, self.config.hidden_size),
                                                              self.config.positional_embedding_init_fn,
                                                              "Positional_Dict")
        return embedding_dict, positional_dict

    def embedding(self, indices, positions, segments):
        with self.embedding_scope:
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
                                    "Positional_Dict",
                                    init_fn=self.config.positional_embedding_init_fn)
                x = self.builder.aiOnnx.add([x, x_pos])
            x = self.builder.aiOnnx.add([x, x_seg])

            # When outlining is enabled, under certain situations, the `add` above resolves
            # to an AddLhsInPlace, which then causes the output to be laid out incorrectly
            # for SQuAD. This workaround ensures it stays as an AddRhsInPlace.
            self.builder.setInplacePreferences(x, {"AddRhsInplace": 1000.0})

            x = self.norm(x)
            x = self.dropout(x)
        return x

    def gather(self, indices, embedding_size, name, init_fn="DEFAULT"):
        if self.config.host_embedding in ("ALL", "WORD", "MERGE") and name == "Embedding_Dict":
            return indices
        if self.config.host_embedding in ("ALL", "MERGE") and name == "Positional_Dict":
            return indices
        if name == "Embedding_Dict" and self.config.task == "PRETRAINING":
            # Important that the tied gather/matmul weight with transpose before the gather.
            # This will ensure it matches the custom_ops/tied_gather_pattern.
            embedding_dict = self.embedding_init_tensor(
                self.config.dtype,
                (self.config.hidden_size, embedding_size),
                init_fn,
                name)
            self.embedding_dict = embedding_dict
            embedding_dict = self.builder.aiOnnx.transpose([embedding_dict])
        else:
            embedding_dict = self.embedding_init_tensor(
                self.config.dtype,
                (embedding_size, self.config.hidden_size),
                init_fn,
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
        Example:
            Task: PRETRAINING
            masks: [2, 5]
            mask_tokens: 4
            returns: [0,0,-1000.0, -1000.0, 0, -1000.0, ...]
        """
        mask_idx = self.builder.getVirtualGraph() if self.builder.hasVirtualGraph() else None

        if mask_idx in self.masks:
            return self.masks[mask_idx]

        mask_scope = self.device_scope(mask_idx,
                                       self.builder.getPipelineStage() if self.builder.hasPipelineStage() else None,
                                       "Mask")
        with mask_scope:
            base_value = np.arange(self.config.sequence_length)
            base = self.constant_tensor(base_value, np.uint32, "mask_sequence")
            if self.config.task == "PRETRAINING":
                # Mask tokens mask
                mmask = self.builder.aiOnnx.less([base, masks[0]])
                # No constexpr for greater. Create as const instead
                _mask = self.constant_tensor(np.greater_equal(
                    base_value, self.config.mask_tokens), np.bool)
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
                [self.config.batch_size, 1, 1, self.config.sequence_length])
            # TODO: This shouldn't be needed. No Variables on this path.
            final_mask = self.detach(final_mask)
            self.masks[mask_idx] = final_mask
        return final_mask

    def attention(self, input_x, masks=None):
        qkv_weights = self.normal_init_tensor(
            self.config.dtype,
            [self.config.hidden_size, 3 * self.config.hidden_size],
            0, 0.02,
            "QKV")
        qkv = self.builder.aiOnnx.matmul([input_x, qkv_weights])
        if self.config.split_linear_layers:
            self.builder.setSerializeMatMul({qkv}, 'output_channels', 3, True)
        if not self.config.use_default_available_memory_proportion:
            self.builder.setAvailableMemoryProportion(
                qkv, self.config.available_memory_proportion)

        if 'attention' in self.config.custom_ops:
            x = self.attention_custom(qkv, masks)
        else:
            x = self.attention_onnx(qkv, masks)

        projection_weights = self.normal_init_tensor(
            self.config.dtype,
            [self.config.hidden_size, self.config.hidden_size],
            0, 0.02,
            "Out")
        x = self.builder.aiOnnx.matmul([x, projection_weights])
        if not self.config.use_default_available_memory_proportion:
            self.builder.setAvailableMemoryProportion(
                x, self.config.available_memory_proportion)

        x = self.dropout(x)
        x = self.builder.aiOnnx.add([input_x, x])
        x = self.norm(x)
        return x

    def attention_custom(self, qkv, masks):
        if self.config.no_mask or masks is not None:
            mask = self.attention_mask(masks)
        else:
            mask = self.constant_tensor(
                np.zeros([self.config.sequence_length]), self.config.dtype)

        available_memory_proportion = self.config.available_memory_proportion
        if self.config.use_default_available_memory_proportion:
            available_memory_proportion = -1

        dropout_modifier = self.dropout_modifier
        if self.config.no_dropout or self.config.no_attn_dropout:
            dropout_modifier = -1

        x = self.builder.customOp(opName="Attention",
                                  opVersion=1,
                                  domain="ai.graphcore",
                                  inputs=[qkv, mask],
                                  attributes={
                                      "heads": self.config.attention_heads,
                                      "sequence_length": self.config.sequence_length,
                                      "available_memory_proportion": available_memory_proportion,
                                      "dropout_modifier": dropout_modifier,
                                      "dropout_ratio": self.config.attn_dropout_prob
                                  },
                                  numOutputs=6)[0]
        self.dropout_modifier += 1
        return x

    def attention_onnx(self, qkv, masks):
        comb_shape = [self.config.batch_size, self.config.sequence_length,
                      self.config.attention_heads, self.config.qkv_length]

        def extract_heads(tensor, transpose=False):
            tensor = self.builder.reshape_const(
                self.builder.aiOnnx, [tensor], comb_shape)
            perm = [0, 2, 1, 3] if not transpose else [0, 2, 3, 1]
            return self.builder.aiOnnx.transpose([tensor], perm=perm)

        split_qkv = self.builder.aiOnnx.split(
            [qkv],
            num_outputs=3,
            axis=1,
            split=[self.config.hidden_size]*3,
            debugPrefix="QKV_Split")

        q, kt, v = [extract_heads(t, i == 1) for i, t in enumerate(split_qkv)]

        # Attention calculation
        with self.builder.nameScope('Z'):
            x = self.builder.aiOnnx.matmul([q, kt])
            if not self.config.use_default_available_memory_proportion:
                self.builder.setAvailableMemoryProportion(
                    x, self.config.available_memory_proportion)

            c = self.constant_tensor(
                1 / np.sqrt(self.config.qkv_length), self.config.dtype)
            x = self.builder.aiOnnx.mul([x, c])

            if self.config.no_mask or masks is not None:
                mask = self.attention_mask(masks)
                x = self.builder.aiOnnx.add([x, mask], "ApplyMask")

            x = self.builder.aiOnnx.softmax([x], axis=-1)

            if not self.config.no_attn_dropout:
                x = self.dropout(x)

            # x[batch_size, attention_heads, sequence_length, sequence_length] * v[batch_size, attention_heads, sequence_length, qkv_length]
            z = self.builder.aiOnnx.matmul([x, v])
            if not self.config.use_default_available_memory_proportion:
                self.builder.setAvailableMemoryProportion(
                    z, self.config.available_memory_proportion)

            # [batch_size, attention_heads, sequence_length, qkv_length] -> [batch_size, sequence_length, attention_heads, qkv_length]
            z = self.builder.aiOnnx.transpose([z], perm=[0, 2, 1, 3])
            # [batch_size, sequence_length, attention_heads, qkv_length] -> [batch_size*sequence_length, attention_heads*qkv_length]
            z = self.builder.reshape_const(self.builder.aiOnnx, [z], [
                                           self.config.sequence_length * self.config.batch_size, self.config.hidden_size])
        return z

    def projection(self, input_x):
        x = self.builder.reshape_const(self.builder.aiOnnx, [input_x], [
                                       self.config.batch_size, self.config.sequence_length, self.config.hidden_size])

        x = self.builder.aiOnnxOpset9.slice([x], axes=[1], starts=[
            0], ends=[self.config.mask_tokens])

        weight = self.embedding_dict

        # Move the weight to the current pipeline stage
        if weight in self.tensors[self.embedding_scope.pipelineStage]:
            embedding_stage = self.embedding_scope.pipelineStage
            self.tensors[embedding_stage].remove(weight)
            self._add_to_tensor_map(weight)

        x = self.builder.aiOnnx.matmul([x, weight])
        num_splits = self.config.projection_serialization_steps
        self.builder.setSerializeMatMul(
            {x}, 'output_channels', num_splits, True)

        x = self.builder.reshape_const(self.builder.aiOnnx, [x], [
                                       self.config.batch_size, self.config.mask_tokens, self.config.vocab_length])
        return x

    def squad_projection(self, input_x):
        weight = self.normal_init_tensor(self.config.dtype,
                                         [self.config.hidden_size, 2],
                                         0, 0.02,
                                         "SquadW")
        bias = self.constant_init_tensor(self.config.dtype, (2,), 0, "SquadB")
        x = self.builder.aiOnnx.gemm([input_x, weight, bias])
        # x.shape: [batch_size * sequence_length, 2]
        start_logits = self.builder.aiOnnxOpset9.slice(
            [x], axes=[1], starts=[0], ends=[1])
        end_logits = self.builder.aiOnnxOpset9.slice(
            [x], axes=[1], starts=[1], ends=[2])

        start_logits = self.builder.reshape_const(
            self.builder.aiOnnx,
            [start_logits], [self.config.batch_size, self.config.sequence_length], debugPrefix="answer_start")
        end_logits = self.builder.reshape_const(
            self.builder.aiOnnx,
            [end_logits], [self.config.batch_size, self.config.sequence_length], debugPrefix="answer_end")

        return start_logits, end_logits

    def pooler(self, pooler_input):
        """
        Take the [CLS] token as a sentence embedding (assuming it's already been
        fine-tuned), then run a FC layer with tanh activation
        """
        pooler_input = self.builder.aiOnnxOpset9.slice(
            [pooler_input], axes=[1], starts=[self.config.mask_tokens], ends=[self.config.mask_tokens + 1]
        )

        # This reshape is doing the job of a squeeze, but allows for in-place operation.
        pooler_input = self.builder.reshape_const(self.builder.aiOnnx, [pooler_input], [
            self.config.batch_size, self.config.hidden_size])

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
        x = self.builder.aiOnnx.gemm([pooler_input, weight, bias])

        return self.builder.aiOnnx.tanh([x])

    def nsp_head(self, input_x):
        x = self.builder.reshape_const(self.builder.aiOnnx, [input_x], [
                                       self.config.batch_size, self.config.sequence_length, self.config.hidden_size])

        x = self.pooler(x)

        cls_weight = self.normal_init_tensor(
            self.config.dtype, [self.config.hidden_size, 2], 0, 0.02, "NspW"
        )
        cls_bias = self.constant_init_tensor(
            self.config.dtype, (2,), 0, "NspB")
        x = self.builder.aiOnnx.gemm([x, cls_weight, cls_bias])
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

        x = self.builder.aiOnnx.gemm([input_x, dense_weight, dense_bias])

        x = self.intermediate_activation_function(x)
        x = self.norm(x)
        return x
