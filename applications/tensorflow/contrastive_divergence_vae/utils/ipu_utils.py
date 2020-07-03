# Copyright 2019 Graphcore Ltd.
# coding=utf-8
import os
import tensorflow as tf
from tensorflow.contrib.compiler import xla


def get_ipu_tf():
    try:
        from tensorflow.python import ipu as ipu_tf
    except ImportError:
        ipu_tf = None
    return ipu_tf


def get_ipu_scope(ipu_tf):
    if ipu_tf is None:
        return None
    else:
        return ipu.scopes.ipu_scope

ipu = get_ipu_tf()
ipu_scope = get_ipu_scope(ipu)


def get_ipu_option_dict(ipu_id=None, prng=False, n_ipus=1):
    """
    Collates IPU config into single dict, to be used as **kwargs input to tf.ConfigProto

    Returns:
        dict of config
    """
    options = ipu.utils.create_ipu_config(prefetch_data_streams=True,
                                          merge_infeed_io_copies=True)

    if ipu_id is None:
        options = ipu.utils.auto_select_ipus(options, num_ipus=[n_ipus])
    else:
        options = ipu.utils.select_ipus(options, [ipu_id])
    options = ipu.utils.set_compilation_options(options, {
        "device.clearAtomicFlagAfterExchange": "false",
        "prng.enable": "true" if prng else "false"})   # Stochastic rounding

    return {'ipu_options': options}


def get_device_config(desired_device, num_ipus=1, only_ipu=False):

    def _else_cpu(desired_scope, candidate_device):
        if candidate_device in desired_scope.lower():
            return desired_scope
        else:
            return '/device:CPU:0'

    def _not_xla_compile(comp, inputs):
        return comp(*inputs)

    sess_opts = {}
    try:
        # Config to run on IPU - if appropriate IPU TF modules can be loaded
        from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

        if 'ipu' not in desired_device.lower():
            # On IPU server but want to run on CPU
            raise ImportError

        # Set XLA compilation
        ipu_options = get_ipu_option_dict(n_ipus=num_ipus)
        device_scope = _else_cpu(desired_device, 'ipu')
        maybe_xla_compile = xla.compile
        sess_opts['log_device_placement'] = False
        sess_opts['allow_soft_placement'] = False
        on_ipu = True

    except ImportError:

        if only_ipu:
            raise ImportError("IPU not available and IPU only mode selected.")

        ipu_options = {}
        if tf.test.is_gpu_available():
            # Run on GPU if machine has one
            maybe_xla_compile = xla.compile
            device_scope = '/device:GPU:0'
            sess_opts['log_device_placement'] = True
            sess_opts['allow_soft_placement'] = True
            on_ipu = False
        else:
            # No IPU or GPU - run on CPU
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
            maybe_xla_compile = _not_xla_compile
            device_scope = '/device:CPU:0'
            on_ipu = False
    scope_call = get_device_scope_call(device_scope)
    return {'device': device_scope, 'scoper': scope_call, 'sess_options': sess_opts,
            'ipu_options': ipu_options, 'maybe_xla_compile': maybe_xla_compile,
            'do_xla': maybe_xla_compile is xla.compile, 'on_ipu': on_ipu}


def get_device_scope_call(scope_str):
    """
    Logical flow to get device scope commands for train and test

    Args:
        scope_str (str): desired device

    Returns:
        tuple[scope statements]: train device scope, test device scope
    """
    scope_call = ScopeCall(ipu_scope, scope_str) if\
        'ipu' in scope_str.lower() else ScopeCall(tf.device, scope_str)

    return scope_call


class ScopeCall(object):
    def __init__(self, scope, *args, **kwargs):
        self.scope = scope
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args):
        if len(args):
            return self.scope(*args, **self.kwargs)
        else:
            return self.scope(*self.args, **self.kwargs)

    def __eq__(self, other):
        return (self.scope == other.scope) and \
               (sorted(self.args) == sorted(other.args)) and \
               (self.kwargs == other.kwargs)

    def __str__(self):
        return str(self.__dict__)


def get_device_str():
    try:
        try:
            from tensorflow.contrib.ipu import ops
        except ImportError:
            from tensorflow.python.ipu import ops
        return 'IPU'
    except ImportError:
        if tf.test.is_gpu_available():
            return 'GPU'
        else:
            return 'CPU'


def loops_repeat(device, n, body, inputs, infeed_queue, backprop=True, maybe_xla=lambda fn, args: fn(*args)):
    """
    To repeat same op `n` times in single session call. Returns tuple (function, args).
    Allows similar logic to IPU infeeds on other devices.

    :param device: str, TF device specification
    :param n: int, number of repeats
    :param body: fn, computation to run
    :param inputs: iterable of inputs which are updated in each iteration
    :param infeed_queue: either 1. tf.contrib.ipu.infeed_queue object, if running on ipu
                                2. tf.data.Iterator, if on CPU/GPU and not compiling
                                3. tuple(tf.Tensor), if on CPU/GPU and compiling.
                                    Dimensions of the Tensors should be `batch_size` X `n`
    :param backprop: bool, whether backprop should be enabled through the loop
    :param maybe_xla: a function, takes a 2-tuple, (function, arguments)
    """
    if 'ipu' in device.lower():
        return ipu.loops.repeat(n, body, inputs, infeed_queue)
    else:
        if maybe_xla is xla.compile:
            # Assumes infeed_queue is iterable of tensors with batch_size scaled by n
            def for_compile_fn(*infeed_args):

                def _iterate_compile(i, sample, *args):
                    if isinstance(sample, (list, tuple)):
                        arqs = list(args) + list(sample)
                        out = body(*arqs)
                    else:
                        out = body(*args, **sample)
                    if isinstance(out, (list, tuple)):
                        return (tf.add(i, 1),) + tuple(out)
                    else:
                        return tf.add(i, 1), out

                batch_size = infeed_args[0].get_shape()[0] // n
                superargs_rs = [tf.reshape(t, [n, batch_size] + t.get_shape().as_list()[1:]) for t in infeed_args]
                superargs_arr = [tf.TensorArray(t.dtype, n).unstack(t) for t in superargs_rs]

                def iterate_compile(i, *subargs):
                    infeed_batches = [ta.read(i) for ta in superargs_arr]
                    return _iterate_compile(i, infeed_batches, *subargs)

                return tf.while_loop(
                    cond=lambda i, *args, **kwargs: tf.less(i, n),
                    body=iterate_compile,
                    loop_vars=[tf.constant(0, dtype=tf.int32)] + list(inputs),
                    parallel_iterations=1,
                    back_prop=True)[1:]

            return for_compile_fn(*infeed_queue)

        else:
            def not_for_compile_fn():

                def iterate(i, *args):
                    if infeed_queue is None:
                        sample = {}
                    else:
                        sample = infeed_queue.get_next()
                    if isinstance(sample, (list, tuple)):
                        arqs = list(args) + list(sample)
                        out = body(*arqs)
                    else:
                        out = body(*args, **sample)
                    if isinstance(out, (list, tuple)):
                        return (tf.add(i, 1),) + tuple(out)
                    else:
                        return tf.add(i, 1), out

                return tf.while_loop(
                    cond=lambda i, *args, **kwargs: tf.less(i, n),
                    body=iterate,
                    loop_vars=[tf.constant(0, dtype=tf.int32)] + list(inputs),
                    parallel_iterations=1,
                    back_prop=backprop)[1:]

            return not_for_compile_fn()
