# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Utilities to support running TF1 models on IPU/CPU/etc."""

import abc
import enum
import itertools as it
from typing import Callable, Dict, Iterable

import numpy as np
import tensorflow.compat.v1 as tf

try:
    from tensorflow.python import ipu

    IPU_AVAILABLE = True
except ImportError:
    IPU_AVAILABLE = False


class Target(enum.Enum):
    """The top-level execution API, either default TF or XLA for IPU or CPU."""

    DEFAULT = 0
    XLA = 1
    IPU = 2

    def compile(
        self,
        fn: Callable[..., Dict[str, tf.Tensor]],
        inputs: Dict[str, tf.Tensor],
    ) -> tf.Operation:
        """Optionally compile an operation using XLA.

        Equivalent to: `return fn(**inputs)`
        """
        if self is self.DEFAULT:
            return fn(**inputs)
        if self is self.XLA:
            return tf.xla.experimental.compile(lambda _inputs: fn(**_inputs),
                                               [inputs])
        if self is self.IPU:
            with ipu.scopes.ipu_scope("/device:IPU:0"):
                return ipu.ipu_compiler.compile(lambda _inputs: fn(**_inputs),
                                                [inputs])
        assert False, f"Unexpected Target {self}"


class Runner(abc.ABC):
    @abc.abstractmethod
    def __call__(self, session: tf.Session) -> Iterable[Dict[str, np.ndarray]]:
        ...


class IteratorRunner(Runner):
    """Execute batches one by one using a dataset iterator."""
    def __init__(
        self,
        fn: Callable[..., Dict[str, tf.Tensor]],
        dataset: tf.data.Dataset,
        n_batch: int,
        target: Target,
    ):
        self.n_batch = n_batch
        with tf.device("cpu"):
            iterator = tf.data.make_initializable_iterator(dataset)
            self._reset = iterator.initializer
            it_next = iterator.get_next()
        self._step = target.compile(fn, it_next)

    def __call__(self, session: tf.Session) -> Iterable[Dict[str, np.ndarray]]:
        session.run(self._reset)
        try:
            for n in it.count():
                yield session.run(self._step)
        except tf.errors.OutOfRangeError:
            assert n == self.n_batch


class IpuLoopRunner(Runner):
    """Pass batches and results to device using infeeds/outfeeds."""
    def __init__(
        self,
        fn: Callable[..., Dict[str, tf.Tensor]],
        dataset: tf.data.Dataset,
        n_batch: int,
    ):
        infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset.repeat())
        outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

        def loop_body(*args: tf.Tensor, **kwargs: tf.Tensor) -> tf.Operation:
            assert not args, "*args are required compile loop_body(), although unused"
            return outfeed.enqueue(fn(**kwargs))

        self.n_batch = n_batch
        self._run_loop = Target.IPU.compile(
            lambda: ipu.loops.repeat(n_batch, loop_body, infeed_queue=infeed),
            {})
        self._dequeue = outfeed.dequeue()
        self._initializer = infeed.initializer

    def __call__(self, session: tf.Session) -> Iterable[Dict[str, np.ndarray]]:
        if self._initializer is not None:
            session.run(self._initializer)
            self._initializer = None
        session.run(self._run_loop)
        out = session.run(self._dequeue)
        for n in range(self.n_batch):
            yield {k: out[k][n] for k in out.keys()}
