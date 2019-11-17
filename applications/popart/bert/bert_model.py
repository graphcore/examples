# Copyright 2019 Graphcore Ltd.
import popart
import numpy as np
from scipy.stats import truncnorm
from typing import NamedTuple, List, Optional
from functools import reduce
from contextlib import contextmanager, ExitStack
from collections import defaultdict
import math


class BertConfig(NamedTuple):
    batch_size: int = 1
    sequence_length: int = 128
    max_positional_length: int = 512

    # Choices: "DEFAULT", "TRANSFORMER", "SIMPLIFIED"
    positional_embedding_init_fn: str = "DEFAULT"

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
    dropout_prob: float = 0.1
    attn_dropout_prob: float = 0.1

    layer_norm_eps: float = 0.001

    # Choices: PRETRAINING (MLM + NSP), SQUAD, MRPC
    task: str = "PRETRAINING"

    # Choices: FLOAT, FLOAT16
    popart_dtype: str = "FLOAT16"

    # Choices: embedding, attention, feed_forward
    custom_ops: List[str] = []

    # This option uses the projection custom_op for all linear layers and
    # serialises them to multiples of (hidden_size, hidden_size) matmuls.
    # This is required for sequence length 384.
    split_linear_layers: bool = False

    # Try and fit the model onto fewer IPUs. Intended for inference modes:
    squeeze_model: bool = False

    no_mask: bool = False

    activation_type: str = 'Relu'

    relu_leak: float = 0.1

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

    projection_serialization_steps: int = 5

    @property
    def available_memory_proportion(self):
        '''This matmul option specifies the proportion of total tile memory the temporary values
        can use. If the operation exceeds this value it will be serialized by poplibs.
        Note: this is different to using PopART's setSerializeMatMul as the matmul will still be a single PopART Op
        meaning other operations cannot be scheduled between the serialised steps.
        BERT uses setSerializeMatMul so VarUpdate can execute between steps thus freeing the required gradient memory'''
        if self.sequence_length <= 128:
            return 0.6
        elif self.sequence_length <= 256:
            max_matmul_memory = 60000
        elif self.sequence_length <= 384:
            max_matmul_memory = 45000
        else:
            max_matmul_memory = 40000
        # max / 256KiB (total tile memory)
        return max_matmul_memory / (2**18)


class DeviceScope(object):
    def __init__(self,
                 builder,
                 virtualGraph=None,
                 pipelineStage=None,
                 nameScope=None):
        self.builder = builder
        self.virtualGraph = virtualGraph
        self.pipelineStage = pipelineStage
        self.nameScope = nameScope

    def __enter__(self):
        self.stack = ExitStack()
        if self.virtualGraph is not None:
            self.stack.enter_context(self.builder.virtualGraph(self.virtualGraph))
        if self.pipelineStage is not None:
            self.stack.enter_context(self.builder.pipelineStage(self.pipelineStage))
        if self.nameScope is not None:
            self.stack.enter_context(self.builder.nameScope(self.nameScope))
        return self

    def __exit__(self, *exp):
        self.stack.close()
        return False


class Model(object):
    def __init__(self, builder=popart.Builder(), initializers=None):
        if initializers is None:
            initializers = {}
        self.builder = builder
        self.initializers = initializers

        # Whenever a new tensor is created on a given pipeline stage, it should be added
        # to this dict to ensure it is given the correct learning rate
        self.pipeline_stage_tensors = defaultdict(list)

    def normal_init_tensor(self, dtype, shape, mean, std_dev, debug_name=""):
        data = self.initializers.get(
            self.builder.getNameScope(debug_name), None)
        if data is None:
            # Truncated random normal between 2 standard devations
            data = truncnorm.rvs(-2, 2, loc=mean,
                                 scale=std_dev, size=np.prod(shape))
            data = data.reshape(shape).astype(dtype)
        else:
            if np.any(data.shape != np.array(shape)):
                raise RuntimeError(f"Initializer {self.builder.getNameScope(debug_name)} does not match shapes. \n"
                                   f" Provided {data.shape}. Required {shape}")
        tensor = self.builder.addInitializedInputTensor(data, debug_name)
        self._add_to_tensor_map(tensor)
        return tensor

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

    def device_scope(self, virtualGraph=None, pipelineStage=None, nameScope=None):
        return DeviceScope(self.builder, virtualGraph, pipelineStage, nameScope)

    def _add_to_tensor_map(self, tensor):
        if not self.builder.hasPipelineStage():
            return
        pipeline_stage = self.builder.getPipelineStage()
        self.pipeline_stage_tensors[pipeline_stage].append(tensor)


class Bert(Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.dropout_modifier = 256

        # This is the TensorId for the shared embedding & projection
        self.embedding_dict = None
        self.projection_split = 4
        if (self.config.vocab_length % 5) == 0:
            self.projection_split = 5

        # This dict[ipu,TensorId] reuses any mask already generated on an IPU/pipeline stage
        # TODO: Should recompute instead?
        self.masks = {}

        self.init_device_placement()

    def init_device_placement(self):
        ''' Create a DeviceScope for each layer, ie Embedding, SQUAD, NSP
        '''
        self.layer_offset = 1
        if self.config.task == "PRETRAINING":
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
            self.nsp_scope = self.device_scope(self.embedding_split_scope.virtualGraph, pipeline_stage)
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

    def encoder_scope(self, layer_index):
        ipu = (layer_index // self.config.layers_per_ipu) + self.layer_offset
        return self.device_scope(ipu, ipu, f"Layer{layer_index}")

    def build_graph(self, indices, positions, segments, masks=None):
        # Embedding
        with self.builder.nameScope("Embedding"):
            x = self.embedding(indices, positions, segments)

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
            with self.nsp_scope, self.builder.nameScope("NSP"):
                outputs.append(self.nsp_head(x))

        if self.config.task == "PRETRAINING":
            # FIXME: T11914
            # with self.nsp_scope, self.builder.nameScope("CLS"):
            #     predictions = self.lm_prediction_head(x)

            with self.mlm_scope:
                outputs = [self.projection(x)] + outputs

        # Fine Tuning tasks
        if self.config.task == "SQUAD":
            with self.squad_scope:
                outputs += self.squad_projection(x)

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

    def matmul(self, input_x, weight, split=None):
        if split is None:
            split = [2, 1]
        x = self.builder.customOp(opName="Projection",
                                  opVersion=1,
                                  domain="ai.graphcore",
                                  inputs=[input_x, weight],
                                  attributes={
                                      "mask_tokens": 0,
                                      "split": split
                                  })[0]
        return x

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

    def gelu(self, input_x):
        """
            Implementation of the GELU function (https://arxiv.org/abs/1606.08415)
        """
        one_half = self.builder.aiOnnx.constant(
            np.asarray([0.5], dtype=self.config.dtype))
        one = self.builder.aiOnnx.constant(
            np.asarray([1], dtype=self.config.dtype))
        scale = self.builder.aiOnnx.constant(
            np.asarray([0.044715], dtype=self.config.dtype))
        two_on_pi = self.builder.aiOnnx.constant(np.asarray(
            [math.sqrt(2./math.pi)], dtype=self.config.dtype))
        result = self.builder.aiOnnx.mul([input_x, input_x])
        result = self.builder.aiOnnx.mul([result, input_x])
        result = self.builder.aiOnnx.mul([scale, result])
        result = self.builder.aiOnnx.add([input_x, result])
        result = self.builder.aiOnnx.mul([two_on_pi, result])
        result = self.builder.aiOnnx.tanh([result])
        result = self.builder.aiOnnx.add([one, result])
        result = self.builder.aiOnnx.mul([input_x, result])
        result = self.builder.aiOnnx.mul([one_half, result])
        return result

    def gelu_custom(self, input_x):
        return self.builder.customOp(opName="Gelu",
                                     opVersion=1,
                                     domain="ai.graphcore",
                                     inputs=[input_x],
                                     attributes={})[0]

    def intermediate_activation_function(self, input_x):
        if self.config.activation_type == 'GeluCustom':
            return self.gelu_custom(input_x)
        elif self.config.activation_type == 'Gelu':
            return self.gelu(input_x)
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
            if 'feed_forward' in self.config.custom_ops:
                split = [
                    2, num_splits] if self.config.split_linear_layers else None
                x = self.matmul(input_x, weight1, split=split)
            else:
                x = self.builder.aiOnnx.matmul([input_x, weight1])
                if self.config.split_linear_layers:
                    self.builder.setSerializeMatMul({x},
                                                    'output_channels',
                                                    num_splits,
                                                    keep_precision=True)
                    self.builder.setAvailableMemoryProportion(x, self.config.available_memory_proportion)
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
            if 'feed_forward' in self.config.custom_ops:
                split = [
                    1, num_splits] if self.config.split_linear_layers else None
                x = self.matmul(x, weight2, split=split)
            else:
                x = self.builder.aiOnnx.matmul([x, weight2])
                if self.config.split_linear_layers:
                    self.builder.setSerializeMatMul({x},
                                                    'reducing_dim',
                                                    num_splits,
                                                    keep_precision=True)
                    self.builder.setAvailableMemoryProportion(x, self.config.available_memory_proportion)
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

    def embedding_init_tensor(self, dtype, shape, init_fn, debug_name=""):
        # Unless specifcally set, fall back to normal tensor initialisation
        if init_fn not in ("TRANSFORMER", "SIMPLIFIED"):
            return self.normal_init_tensor(dtype, shape, 0, 0.02, debug_name)

        data = self.initializers.get(self.builder.getNameScope(debug_name), None)
        if data is None:
            if init_fn == "TRANSFORMER":
                data = self.generate_transformer_periodic_pos_data(dtype, shape)
            else:
                data = self.generate_simplified_periodic_pos_data(dtype, shape)
        else:
            if np.any(data.shape != np.array(shape)):
                raise RuntimeError(f"Initializer {self.builder.getNameScope(debug_name)} does not match shapes. \n"
                                   f" Provided {data.shape}. Required {shape}")
        tensor = self.builder.addInitializedInputTensor(data, debug_name)
        self._add_to_tensor_map(tensor)
        return tensor

    def embedding(self, indices, positions, segments):
        embedding_fn = self.embedding_onnx
        if 'gather' in self.config.custom_ops:
            embedding_fn = self.embedding_custom

        with self.embedding_scope:
            x = embedding_fn(indices, self.config.vocab_length, "Embedding_Dict", detach=True)

        with self.embedding_split_scope:
            x_pos = embedding_fn(positions,
                                 self.config.max_positional_length,
                                 "Positional_Dict",
                                 detach=False,
                                 init_fn=self.config.positional_embedding_init_fn)

            segments_onehot = self.builder.aiOnnx.onehot([
                segments,
                self.constant_tensor(2, dtype=np.int32),
                self.constant_tensor([0, 1], dtype=self.config.dtype)])
            segments_weights = self.normal_init_tensor(
                self.config.dtype,
                [2, self.config.hidden_size],
                0, 0.02, "Segment_Dict")
            x_seg = self.builder.aiOnnx.matmul([segments_onehot, segments_weights])

            x = self.builder.aiOnnx.add([x, x_pos])
            x = self.builder.aiOnnx.add([x, x_seg])
            x = self.norm(x)
            x = self.dropout(x)
        return x


    def embedding_custom(self, indices, embedding_size, name, detach=False, init_fn="DEFAULT"):
        embedding_dict = self.embedding_init_tensor(self.config.dtype,
                                                    (self.config.hidden_size, embedding_size),
                                                    init_fn,
                                                    name)
        attrs = {}
        # TODO: Whats the best layout if there is no projection?
        if name == "Embedding_Dict":
            attrs = {'split': [2, self.projection_split]}
            self.embedding_dict = embedding_dict

        if detach:
            embedding_dict = self.detach(embedding_dict)

        x = self.builder.customOp(opName="EmbeddingGather",
                                  opVersion=1,
                                  domain="ai.graphcore",
                                  inputs=[embedding_dict, indices],
                                  attributes=attrs)[0]
        return x


    def embedding_onnx(self, indices, embedding_size, name, detach=False, init_fn="DEFAULT"):
        embedding_dict = self.embedding_init_tensor(self.config.dtype,
                                                    (embedding_size, self.config.hidden_size),
                                                    init_fn,
                                                    name)
        self.embedding_dict = embedding_dict

        if detach:
            embedding_dict = self.detach(embedding_dict)

        x = self.builder.aiOnnx.gather([embedding_dict, indices])
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
        ipu = self.builder.getVirtualGraph() if self.builder.hasVirtualGraph() else 0
        if ipu in self.masks:
            return self.masks[ipu]
        with self.builder.nameScope("Mask"):
            base_value = np.arange(self.config.sequence_length)
            base = self.constant_tensor(base_value, np.int32, "mask_sequence")
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
            # TODO: This shouldn't be needed. No Variables on this path.
            final_mask = self.detach(final_mask)
            self.masks[ipu] = final_mask
        return final_mask

    def attention(self, input_x, masks=None):
        qkv_weights = self.normal_init_tensor(self.config.dtype,
                                              [self.config.hidden_size, 3 * self.config.hidden_size],
                                              0, 0.02,
                                              "QKV")
        qkv = self.builder.aiOnnx.matmul([input_x, qkv_weights])
        if self.config.split_linear_layers:
            self.builder.setSerializeMatMul({qkv}, 'output_channels', 3, True)
            self.builder.setAvailableMemoryProportion(qkv, self.config.available_memory_proportion)

        if 'attention' in self.config.custom_ops:
            x = self.attention_custom(qkv, masks)
        else:
            x = self.attention_onnx(qkv, masks)

        projection_weights = self.normal_init_tensor(self.config.dtype,
                                                     [self.config.hidden_size,
                                                         self.config.hidden_size],
                                                     0, 0.02,
                                                     "Out")
        x = self.builder.aiOnnx.matmul([x, projection_weights])
        self.builder.setAvailableMemoryProportion(x, self.config.available_memory_proportion)

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
        x = self.builder.customOp(opName="Attention",
                                  opVersion=1,
                                  domain="ai.graphcore",
                                  inputs=[qkv, mask],
                                  attributes={
                                      "heads": self.config.attention_heads,
                                      "sequence_length": self.config.sequence_length,
                                      "mask_tokens": self.config.mask_tokens,
                                      "dropout_modifier": self.dropout_modifier if not self.config.no_dropout else -1,
                                      "dropout_ratio": self.config.attn_dropout_prob
                                  },
                                  numOutputs=6)[0]
        self.dropout_modifier += 1
        return x

    def attention_onnx(self, qkv, masks):
        comb_shape = [self.config.batch_size, self.config.sequence_length,
                      self.config.attention_heads, self.config.qkv_length]

        def extract_heads(tensor, index, hidden_size, transpose=False):
            tensor = self.builder.aiOnnxOpset9.slice([qkv], axes=[1],
                                                     starts=[
                                                         index * hidden_size],
                                                     ends=[(index + 1) * hidden_size])
            tensor = self.builder.reshape_const(
                self.builder.aiOnnx, [tensor], comb_shape)
            perm = [0, 2, 1, 3] if not transpose else [0, 2, 3, 1]
            return self.builder.aiOnnx.transpose([tensor], perm=perm)

        q, kt, v = [extract_heads(
            qkv, i, self.config.hidden_size, i == 1) for i in range(3)]

        # Attention calculation
        with self.builder.nameScope('Z'):
            x = self.builder.aiOnnx.matmul([q, kt])

            c = self.constant_tensor(
                1 / np.sqrt(self.config.qkv_length), self.config.dtype)
            x = self.builder.aiOnnx.mul([x, c])

            if self.config.no_mask or masks is not None:
                mask = self.attention_mask(masks)
                x = self.builder.aiOnnx.add([x, mask], "ApplyMask")

            x = self.builder.aiOnnx.softmax([x], axis=-1)

            x = self.dropout(x)

            # x[batch_size, attention_heads, sequence_length, sequence_length] * v[batch_size, attention_heads, sequence_length, qkv_length]
            z = self.builder.aiOnnx.matmul([x, v])

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

        # The non-custom embedding creates the embedding_dict for the gather. So it needs transposing
        weight = self.embedding_dict
        if 'gather' not in self.config.custom_ops:
            weight = self.builder.aiOnnx.transpose([weight])

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
        Take the first token [CLS] as a sentence embedding (assuming it's already been
        fine-tuned), then run a FC layer with tanh activation
        """
        pooler_input = self.builder.aiOnnxOpset9.slice(
            [pooler_input], axes=[1], starts=[0], ends=[1]
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
        cls_bias = self.constant_init_tensor(self.config.dtype, (2,), 0, "NspB")
        x = self.builder.aiOnnx.gemm([x, cls_weight, cls_bias])
        return x

    def lm_prediction_head(self, input_x):
        dense_weight = self.normal_init_tensor(self.config.dtype,
                                               [self.config.hidden_size,
                                                self.config.hidden_size],
                                               0,
                                               0.02,
                                               "LMPredictionW")

        dense_bias = self.constant_init_tensor(self.config.dtype, [self.config.hidden_size], 0, "LMPredictionB")

        x = self.builder.aiOnnx.gemm([input_x, dense_weight, dense_bias])

        x = self.intermediate_activation_function(x)
        x = self.norm(x)
        return x
