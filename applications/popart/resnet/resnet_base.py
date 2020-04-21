# Copyright 2019 Graphcore Ltd.
from collections import namedtuple
from functools import partial
from utils import NullContextManager

ResNetOptions = namedtuple(
    'ResNetOptions', [
        'size',
        'train',
        'norm_type',
        'norm_groups',
        'shortcut_type'])

ResNetDefinition = namedtuple(
    'ResNetDefinition', [
        'block',
        'shortcut_type',
        'block_counts',
        'out_filters'])


class ResNet(object):
    @property
    def definitions(self):
        return {
            # Cifar10 sized filters
            "8":  ResNetDefinition(
                                    self.block2, 'A',
                                    [1, 1, 1],
                                    [16, 16, 32, 64]),
            "14":  ResNetDefinition(
                                    self.block2, 'A',
                                    [2, 2, 2],
                                    [16, 16, 32, 64]),
            "20":  ResNetDefinition(
                                    self.block2, 'A',
                                    [3, 3, 3],
                                    [16, 16, 32, 64]),
            "32":  ResNetDefinition(
                                    self.block2, 'A',
                                    [5, 5, 5],
                                    [16, 16, 32, 64]),
            "44":  ResNetDefinition(
                                    self.block2, 'A',
                                    [7, 7, 7],
                                    [16, 16, 32, 64]),
            "56":  ResNetDefinition(
                                    self.block2, 'A',
                                    [9, 9, 9],
                                    [16, 16, 32, 64]),
            "110":  ResNetDefinition(
                                    self.block2, 'A',
                                    [18, 18, 18],
                                    [16, 16, 32, 64]),
            # ImageNet sized filters
            "18": ResNetDefinition(
                                    self.block2, 'B',
                                    [2, 2, 2, 2],
                                    [64, 64, 128, 256, 512]),
            "34": ResNetDefinition(
                                    self.block2, 'B',
                                    [3, 4, 6, 3],
                                    [64, 64, 128, 256, 512]),
            "50": ResNetDefinition(
                                    self.block3, 'B',
                                    [3, 4, 6, 3],
                                    [64, (64, 256), (128, 512),
                                        (256, 1024), (512, 2048)]),
            "101": ResNetDefinition(
                                    self.block3, 'B',
                                    [3, 4, 23, 3],
                                    [64, (64, 256), (128, 512),
                                        (256, 1024), (512, 2048)]),
            "152": ResNetDefinition(
                                    self.block3, 'B',
                                    [3, 8, 36, 3],
                                    [64, (64, 256), (128, 512),
                                        (256, 1024), (512, 2048)]),
            "x50": ResNetDefinition(
                                    self.resnext_block, 'B',
                                    [3, 4, 6, 3],
                                    [64, (128, 256), (256, 512),
                                        (512, 1024), (1024, 2048)])

        }

    def __init__(self, opts):
        definition = self.definitions[opts.size]

        # Apply options to layers
        self.conv = partial(self.conv,
                            bias=opts.norm_type == 'NONE')
        self.norm = partial(self.norm,
                            type=opts.norm_type,
                            groups=opts.norm_groups,
                            training=opts.train)

        # Apply changed layers to block functions
        self.block = definition.block
        self.block_counts = definition.block_counts
        self.out_filters = definition.out_filters
        self.residual = partial(
            self.residual,
            type=opts.shortcut_type if opts.shortcut_type is not None
            else definition.shortcut_type)

        # Apply dataset specific changes
        self.initial_k = 7
        self.initial_mp = True
        self.num_classes = 1000

        self.opts = opts

    def _build_graph(self, x):
        x = self.initial_block(
                                x,
                                self.out_filters[0],
                                ksize=self.initial_k,
                                include_mp=self.initial_mp)

        for n in range(len(self.block_counts)):
            first_stride = 1 if n is 0 else 2
            x = self.block(
                            x,
                            first_stride,
                            self.out_filters[n+1],
                            self.block_counts[n],
                            "b{}".format(n+1))

        with self.namescope("classifier"):
            x = self.reduce_mean(x)
            x = self.fc(x, num_units_out=self.num_classes)

        return x

    def __call__(self, x):
        return self._build_graph(x)

    def initial_block(self, x, filters, ksize=3, include_mp=False):
        with self.namescope("initial_block"):
            x = self.conv(x, ksize, 2 if include_mp else 1, filters)
            x = self.norm(x)
            x = self.relu(x)
            if include_mp:
                x = self.maxpool(x)
        return x

    def block2(self, x, first_stride, out_filters, count, debug_name=''):
        for i in range(count):
            shortcut = x
            stride = (first_stride if (i == 0) else 1)

            with self.namescope(debug_name + "/" + str(i) + "/1"):
                x = self.conv(x, 3, stride, out_filters)
                x = self.norm(x)
                x = self.relu(x)

            with self.namescope(debug_name + "/" + str(i) + "/2"):
                x = self.conv(x, 3, 1, out_filters)
                x = self.norm(x)

            with self.namescope(debug_name + "/" + str(i) + "/shortcut"):
                x = self.residual(x, shortcut, out_filters, stride)

        return x

    def block3(self, x, first_stride, filters, count, debug_name=''):
        for i in range(count):
            shortcut = x
            stride = (first_stride if (i == 0) else 1)

            with self.namescope(debug_name + "/" + str(i) + "/1"):
                # downsample on the first 1x1 conv is ResNetv1
                x = self.conv(x, 1, stride, filters[0])
                x = self.norm(x)
                x = self.relu(x)

            with self.namescope(debug_name + "/" + str(i) + "/2"):
                # downsample on the 3x3 conv is ResNetv1.5
                x = self.conv(x, 3, 1, filters[0])
                x = self.norm(x)
                x = self.relu(x)

            with self.namescope(debug_name + "/" + str(i) + "/3"):
                x = self.conv(x, 1, 1, filters[1])
                x = self.norm(x)

            with self.namescope(debug_name + "/" + str(i) + "/shortcut"):
                x = self.residual(x, shortcut, filters[1], stride)

        return x

    def resnext_block(self, x, first_stride, filters, count, debug_name=''):
        for i in range(count):
            shortcut = x
            stride = (first_stride if (i == 0) else 1)

            with self.namescope(debug_name + "/" + str(i) + "/1"):
                # downsample on the first 1x1 conv is ResNetv1
                x = self.conv(x, 1, stride, filters[0])
                x = self.norm(x)
                x = self.relu(x)

            with self.namescope(debug_name + "/" + str(i) + "/2"):
                # downsample on the 3x3 conv is ResNetv1.5
                x = self.conv(x, 3, 1, filters[0], 32)
                x = self.norm(x)
                x = self.relu(x)

            with self.namescope(debug_name + "/" + str(i) + "/3"):
                x = self.conv(x, 1, 1, filters[1])
                x = self.norm(x)

            with self.namescope(debug_name + "/" + str(i) + "/shortcut"):
                x = self.residual(x, shortcut, filters[1], stride)

        return x

    def residual(self, x, shortcut, out_filters, stride, type='B'):
        raise NotImplementedError

    def relu(self, x):
        raise NotImplementedError

    def conv(self, x, ksize, stride, filters_out, bias=True):
        raise NotImplementedError

    def norm(self, x, type='BATCH', groups=32, training=False):
        raise NotImplementedError

    def fc(self, x, num_units_out):
        raise NotImplementedError

    def reduce_mean(self, x, indices=(1, 2)):
        raise NotImplementedError

    def maxpool(self, x):
        raise NotImplementedError

    def namescope(self, debug_string):
        # suppress does nothing... Implement your own.
        return NullContextManager()
