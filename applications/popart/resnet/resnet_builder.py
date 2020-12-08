# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

from resnet_base import ResNet

import numpy as np
import popart

# This is all written for: NCHW


class PopartBuilderResNet(ResNet):
    def __init__(self, *args, **kwargs):
        self.builder = popart.Builder(opsets={"ai.onnx": 10, "ai.onnx.ml": 1, "ai.graphcore": 1})
        self.dtype = np.float16
        super(PopartBuilderResNet, self).__init__(*args, **kwargs)

    def xavier_init(self, shape, num_units_in, num_units_out):
        bound = np.sqrt(6. / (num_units_in + num_units_out))
        return np.random.uniform(-bound, bound, shape).astype(self.dtype)

    def he_init(self, shape):
        return np.random.randn(*shape)*np.sqrt(2/(shape[-2]+shape[-1]))

    def residual(self, x, shortcut, out_filters, stride, type='B'):
        with self.namescope("residual"):
            in_shape = self.builder.getTensorShape(shortcut)
            out_shape = self.builder.getTensorShape(x)
            pad = int(out_shape[1] - in_shape[1])
            if pad != 0 or type == 'C':
                if type == 'A':
                    if stride != 1:
                        shortcut = self.builder.aiGraphcore.subsample([shortcut], [1, 1, stride, stride])

                    if in_shape[1] != out_shape[1]:
                        shortcut = self.builder.aiOnnx.pad([shortcut], pads=[0, 0, 0, 0, 0, pad, 0, 0])
                else:
                    shortcut = self.conv(shortcut, 1, stride, out_filters)
                    shortcut = self.norm(shortcut)

            x = self.builder.aiOnnx.add([shortcut, x])
            x = self.relu(x)
        return x

    def relu(self, x):
        return self.builder.aiOnnx.relu([x])

    def conv(self, x, ksize, stride, filters_out, group=1, bias=True):
        with self.namescope("conv"):
            filters_in = self.builder.getTensorShape(x)[1]

            wshape = [filters_out, int(filters_in/group), ksize, ksize]
            init_weights = self.xavier_init(wshape, filters_in, filters_out)
            weights = self.builder.addInitializedInputTensor(init_weights, "weights")

            # pad = 1 if ksize > 1 else 0
            pad = int(ksize / 2)

            args = [x, weights]
            if bias:
                bshape = [filters_out]
                init_biases = np.zeros(bshape).astype(self.dtype)
                biases = self.builder.addInitializedInputTensor(init_biases, "bias")
                args += [biases]

            x = self.builder.aiOnnx.conv(args,
                                         dilations=[1, 1],
                                         kernel_shape=[ksize, ksize],
                                         strides=[stride, stride],
                                         group=group,
                                         pads=[pad, pad, pad, pad])
        return x

    def norm(self, x, type='BATCH', groups=32, training=False):
        if type == 'BATCH':
            with self.namescope('batch_norm'):
                C = self.builder.getTensorShape(x)[1]
                init_scale = np.ones([C]).astype(self.dtype)
                scale = self.builder.addInitializedInputTensor(init_scale, "scale")

                init_biases = np.zeros([C]).astype(self.dtype)
                biases = self.builder.addInitializedInputTensor(init_biases, "biases")

                mean = self.builder.addInitializedInputTensor(np.zeros([C]).astype(self.dtype), "mean")
                var = self.builder.addInitializedInputTensor(np.zeros([C]).astype(self.dtype), "var")

                (x, *__) = self.builder.aiOnnx.batchnormalization(
                    [x, scale, biases, mean, var],
                    num_outputs=5 if training else 1)
        elif type == 'GROUP':
            with self.namescope('group_norm/' + str(self.builder.getTensorShape(x)[1])):
                C = self.builder.getTensorShape(x)[1]
                init_scale = np.ones([C]).astype(self.dtype)
                scale = self.builder.addInitializedInputTensor(init_scale, "scale")

                init_biases = np.zeros([C]).astype(self.dtype)
                biases = self.builder.addInitializedInputTensor(init_biases, "biases")

                (x, _, _) = self.builder.aiGraphcore.groupnormalization([x, scale, biases], groups)
        return x

    def fc(self, x, num_units_out):
        with self.namescope("fc"):
            shape = self.builder.getTensorShape(x)
            num_units_in = shape[1]

            wshape = [num_units_in, num_units_out]
            init_weights = self.xavier_init(wshape, num_units_in, num_units_out)
            weights = self.builder.addInitializedInputTensor(init_weights, "weights")

            x = self.builder.aiOnnx.matmul([x, weights])  # This could be a gemm with bias

            init_biases = np.zeros([num_units_out]).astype(self.dtype)
            biases = self.builder.addInitializedInputTensor(init_biases, "biases")

            x = self.builder.aiOnnx.add([x, biases])

        return x

    def reduce_mean(self, x, indices=(2, 3)):
        with self.namescope("reduce_mean"):
            # Not implemented yet
            # x = self.builder.aiOnnx.reducemean([x], axes=indices)
            # Averagepool + reshape instead


            shape_in = self.builder.getTensorShape(x)
            x = self.builder.aiOnnx.averagepool([x],  # GlobalAveragePool?
                                                kernel_shape=shape_in[2:4],
                                                pads=[0, 0, 0, 0],
                                                strides=[1, 1])
            x = self.builder.reshape_const(self.builder.aiOnnx, [x], [shape_in[0], shape_in[1]])

        return x

    def maxpool(self, x):
        (x,) = self.builder.aiOnnx.maxpool(
            args=[x],
            num_outputs=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2])
        return x

    def namescope(self, debug_string):
        return self.builder.nameScope(debug_string)
