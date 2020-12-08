# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import random
import torch
import signal
import functools
import re
import sys
import threading
import traceback
import os
import time
import atexit
from functools import partial
import popart

IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary
# scope) holding reference the traceback.


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""

MP_STATUS_CHECK_INTERVAL = 5.0
r"""Interval (in seconds) to check status of processes to avoid hanging in
    multiprocessing data loading. This is mainly used in getting data from
    another process, in which case we need to periodically check whether the
    sender is alive to prevent hanging."""

if IS_WINDOWS:
    # On Windows, the parent ID of the worker process remains unchanged when
    # the manager process is gone, and the only way to check it through OS is
    # to let the worker have a process handle of the manager and ask if the
    # process status has changed.
    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()

            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from
            # https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(
                SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from
                # https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(
                    self.manager_handle, 0) == 0
            return not self.manager_dead
else:

    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


def _worker_loop(dataset, index_queue, data_queue, done_event, collate_fn,
                 seed, init_fn, worker_id, log_statisics):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
    # the logic of this function.

    popart.getLogger().debug(
        "Starting dataloaderiterator worker process {} (stats:{})".format(
            worker_id, log_statisics))

    try:
        global _use_shared_memory
        _use_shared_memory = True

        # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python
        # signal module's handlers are executed after Python returns from C
        # low-level handlers, likely when the same fatal signal happened again
        # already. https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
        torch._C._set_worker_signal_handlers()

        # Reduce the priority of the work thread, so that the main thread
        # runs first
        os.nice(5)

        torch.set_num_threads(1)

        random.seed(seed)
        torch.manual_seed(seed)

        data_queue.cancel_join_thread()

        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog()

        processing_times = []
        waiting_times = []

        waiting_time_start = time.time()
        while watchdog.is_alive():

            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            waiting_times.append(time.time() - waiting_time_start)

            if r is None:
                # Received the final signal
                assert done_event.is_set()
                return
            elif done_event.is_set():
                # Done event is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue

            idx, batch_indices = r
            try:
                if log_statisics:
                    processing_time_start = time.time()
                    samples = collate_fn([dataset[i] for i in batch_indices])
                    processing_times.append(time.time() -
                                            processing_time_start)

                    if len(processing_times) > 8:
                        popart.getLogger().info(
                            "DataLoader worker:{0}  waiting: {1:6.4f} "
                            "processing:{2:6.4f}"
                            .format(
                                worker_id,
                                sum(waiting_times) / len(waiting_times),
                                sum(processing_times) / len(processing_times)))
                        processing_times.clear()
                        waiting_times.clear()

                else:
                    samples = collate_fn([dataset[i] for i in batch_indices])
            except Exception:
                # It is important that we don't store exc_info in a variable,
                # see NOTE [ Python Traceback Reference Cycle Problem ]
                data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                data_queue.put((idx, samples))
                del samples

            waiting_time_start = time.time()
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch, tensor_type=None):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)

        # Due to ' _th_cat is not implemented for type torch.HalfTensor' we can
        # not add a transform that created batches of HalfTensor's so for not
        # we will use an option
        if tensor_type is not None:
            return torch.stack(
                batch, 0, out=out).type(numpy_type_map[tensor_type])
        else:
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], torch._six.int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], torch._six.string_classes):
        return batch
    elif isinstance(batch[0], torch._six.container_abcs.Mapping):
        return {
            key: default_collate([d[key] for d in batch], tensor_type)
            for key in batch[0]
        }
    elif isinstance(batch[0], torch._six.container_abcs.Sequence):
        transposed = zip(*batch)
        return [
            default_collate(samples, tensor_type) for samples in transposed
        ]

    raise TypeError((error_msg.format(type(batch[0]))))


_python_exit_status = False
r"""Whether Python is shutting down. This flag is guaranteed to be set before
the Python core library resources are freed, but Python may already be exiting
for some time when this is set.

Hook to set this flag is `_set_python_exit_flag`, and is inspired by a similar
hook in Python 3.7 multiprocessing library:
https://github.com/python/cpython/blob/d4d60134b29290049e28df54f23493de4f1824b6/Lib/multiprocessing/util.py#L277-L327
"""


def _set_python_exit_flag():
    global _python_exit_status
    _python_exit_status = True


atexit.register(_set_python_exit_flag)


class _DataLoaderIter(object):
    r"""Iterates once over the DataLoader's dataset, as specified by the
    sampler"""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this
    # (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #                     |                                    ||     FLOW
    #                     |                                    ||
    #                     |                                    ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed.
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when
    #      the iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may alreay be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children. Thus, we register an `atexit` hook that sets a global flag
    #      `_python_exit_status`. Since `atexit` hooks are executed in reverse
    #      order of registration, we are guaranteed that this flag is set
    #      before library resources we use are freed. (Hooks freeing those
    #      resources are registered at importing the Python core libraries at
    #      the top of this file.) So in `__del__`, we check if
    #      `_python_exit_status` is set or `None` (freed), and perform no-op
    #      if so.
    #
    #      Another problem with `__del__` is also related to the library
    #      cleanup calls. When a process ends, it shuts the all its daemonic
    #      children down with a SIGTERM (instead of joining them without a
    #      timeout). Simiarly for threads, but by a different mechanism. This
    #      fact, together with a few implementation details of multiprocessing,
    #      forcesus to make workers daemonic. All of our problems arise when a
    #      DataLoader is used in a subprocess, and are caused by
    #      multiprocessing code which looks more or less like this:
    #
    #          try:
    #              your_function_using_a_dataloader()
    #          finally:
    #              multiprocessing.util._exit_function()
    #
    #      The joining/termination mentioned above happens inside
    #      `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #      throws, the stack trace stored in the exception will prevent the
    #      frame which uses `DataLoaderIter` to be freed. If the frame has any
    #      reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #      its  `__del__`, which starts the shutdown procedure, will not be
    #      called. That, in turn, means that workers aren't notified.
    #      Attempting to join in `_exit_function` will then result in a hang.
    #
    #      For context, `_exit_function` is also registered as an `atexit`
    #      call. So it is unclear to me (@ssnl) why this is needed in a finally
    #      block.
    #      The code dates back to 2008 and there is no comment on the original
    #      PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #      the finally block and the `atexit` registration) that explains this.
    #
    #      Another choice is to just shutdown workers with logic in 1 above
    #      whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can
    #           still happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we register SIGCHLD handler on main process,
    #           which checks if any of the workers fail in the (Python)
    #           handler.
    #           See DataLoader.cpp.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `ManagerWatchdog` class checks the main
    #               process status.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread
    #           is daemonic and usually automatically joined when the process
    #           exits.
    #
    #           However, in case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever. Therefore,
    #           for both `worker_result_queue` (worker -> main process)
    #           and each `index_queue` (main process -> worker), we use
    #           `q.cancel_join_thread()` in sender process before any `q.put`
    #           to prevent this automatic join.
    #
    #           Moreover, having all queues called `cancel_join_thread` makes
    #           implementing graceful shutdown logic in `__del__` much easier.
    #           It won't need to get from any queue, which would also need to
    #           be guarded by periodic status checks.
    #
    #           Note that this may leave corrupted data in the queue, but we
    #           don't care about the data anyways once we are shutting down.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iteartor is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # [worker processes]
    #   While loader process is alive:
    #     Get from index_queue.
    #       If got a `None`, exit.
    #       If get anything else,
    #          Check `done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data.
    #       If timed out,
    #          No matter `done_event` is set (still need to see `None`) or not,
    #          must continue to next iteration .
    #
    #
    # [main process]
    #   In the DataLoader Iter's `__del__`
    #     a. Set `done_event`
    #
    #        Note: from here on, the workers may exit at
    #              any time after they receive `None`.
    #
    #     c. Exit the workers.
    #          i.   Put `None` in each worker's `index_queue`.
    #          ii.  Join the workers.
    #
    #        NOTE: This has to be after (b) because it may leave corrupted data
    #              in `worker_result_queue`.
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already
    #     shutting down.

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.timeout = loader.timeout
        self.batches_outstanding = 0

        self.log_statisics = loader.log_statisics
        self.processing_times = []

        self.sample_iter = iter(self.batch_sampler)

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.worker_queue_idx = 0
            self.worker_result_queue = torch.multiprocessing.Queue()

            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            self.done_event = torch.multiprocessing.Event()

            # Set to try when resetting the iterator and we want to consume any
            # outstanding data from the workers
            self.flush_data_queue = False

            self.index_queues = []
            self.workers = []
            for i in range(self.num_workers):
                index_queue = torch.multiprocessing.Queue()
                index_queue.cancel_join_thread()
                w = torch.multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, index_queue, self.worker_result_queue,
                          self.done_event, self.collate_fn, base_seed + i,
                          self.worker_init_fn, i, self.log_statisics))
                w.daemon = True
                # NB: Process.start() actually take some time as it needs to
                #     start a process and pass the arguments over via a pipe.
                #     Therefore, we only add a worker to self.workers list
                #     after it started, so that we do not call .join() if
                #     program dies before it starts, and __del__ tries to join
                #     but will get:
                #     AssertionError: can only join a started process.
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            self.data_queue = self.worker_result_queue

            torch.utils.data._utils.signal_handling._set_worker_pids(
                id(self), tuple(w.pid for w in self.workers))
            torch.utils.data._utils.signal_handling._set_SIGCHLD_handler()
            self.worker_pids_set = True

    def reset(self):

        popart.getLogger().debug("Resetting the dataloaderiterator")

        # Drain the workers
        self.flush_data_queue = True
        while self.batches_outstanding > 0:
            self.__next__()
        self.flush_data_queue = False

        # Reset the sample iterator
        self.send_idx = 0
        self.rcvd_idx = 0
        self.sample_iter = iter(self.batch_sampler)

        # prime the prefetch loop
        for _ in range(2 * self.num_workers):
            self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def _get_batch(self):
        # In the non-timeout case, worker exit is covered by SIGCHLD handler.
        if self.timeout > 0:
            try:
                return self.data_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError(
                    'DataLoader timed out after {} seconds'.format(
                        self.timeout))
        else:
            return self.data_queue.get()

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            if self.log_statisics:
                t = time.time()
                batch = self.collate_fn([self.dataset[i] for i in indices])
                self.processing_times.append(time.time() - t)

                if len(self.processing_times) > 8:
                    popart.getLogger().info(
                        "DataLoader processing:{0:6.4f}".format(
                            sum(self.processing_times) / len(
                                self.processing_times)))
                    self.processing_times.clear()
            else:
                batch = self.collate_fn([self.dataset[i] for i in indices])
            return [x.numpy() for x in batch]

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        # raise the stop iteration execption when we have recevied all
        # batches in the data set
        if self.batches_outstanding == 0:
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):

        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return

        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        if not self.flush_data_queue:
            self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)

        # Convert the batch into numpy arrays
        return [x.numpy() for x in batch]

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("_DataLoaderIter cannot be pickled")

    def _shutdown_workers(self):
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details
        # on the logic of this function.
        if _python_exit_status is True or _python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self.shutdown:
            self.shutdown = True
            # Removes pids from the C side data structure first so worker
            # termination afterwards won't trigger false positive error report.
            if self.worker_pids_set:
                torch.utils.data._utils.signal_handling._remove_worker_pids(
                    id(self))
                self.worker_pids_set = False

            self.done_event.set()

            # Exit workers now.
            for q in self.index_queues:
                q.put(None)
                # Indicate that no more data will be put on this queue by the
                # current process.
                q.close()
            for w in self.workers:
                w.join()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class DataLoader(object):
    r"""
  Data loader. Combines a dataset and a sampler, and provides
  single- or multi-process iterators over the dataset.

  This is a customized DataLoader for popart which is inspired by the torch
  DataLoader. The difference being that the __iter__ call reuses the
  _DataLoader when called multiple times, to prevent the work processes being
  stopped and respawned.

  The DataLoader can be used with the 'enumerate' call to get an interator to
  the data set. The iterator will return numpy arrays.

  Additionally the cuda pin_memory option has been removed.

  Arguments:
      dataset (Dataset): dataset from which to load the data.
      batch_size (int, optional): how many samples per batch to load
          (default: ``1``).
      shuffle (bool, optional): set to ``True`` to have the data reshuffled
          at every epoch (default: ``False``).
      sampler (Sampler, optional): defines the strategy to draw samples from
          the dataset. If specified, ``shuffle`` must be False.
      batch_sampler (Sampler, optional): like sampler, but returns a batch of
          indices at a time. Mutually exclusive with :attr:`batch_size`,
          :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
      num_workers (int, optional): how many subprocesses to use for data
          loading. 0 means that the data will be loaded in the main process.
          (default: ``0``)
      collate_fn (callable, optional): merges a list of samples to form a
          mini-batch.
      drop_last (bool, optional): set to ``True`` to drop the last incomplete
          batch, if the dataset size is not divisible by the batch size.
          If ``False`` and the size of dataset is not divisible by the batch
          size, then the last batch will be smaller. (default: ``False``)
      tensor_type : The type of tensor to be returned. By default the type will
          be float32. This value can be set to 'float32','float16' to return
          the desired type
      timeout (numeric, optional): if positive, the timeout value for
          collecting a batch from workers. Should always be non-negative.
          (default: ``0``)
      worker_init_fn (callable, optional): If not ``None``, this will be called
          on each worker subprocess with the worker id (an int in
          ``[0, num_workers - 1]``) as input, after seeding and before data
          loading. (default: ``None``)

  .. note:: By default, each worker will have its PyTorch seed set to
            ``base_seed + worker_id``, where ``base_seed`` is a long generated
            by main process using its RNG. However, seeds for other libraies
            may be duplicated upon initializing workers (w.g., NumPy), causing
            each worker to return identical random numbers. (See
            :ref:`dataloader-workers-random-seed` section in FAQ.) You may
            use :func:`torch.initial_seed()` to access the PyTorch seed for
            each worker in :attr:`worker_init_fn`, and use it to set other
            seeds before data loading.

  .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot
            be an unpicklable object, e.g., a lambda function.
  """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=default_collate,
                 drop_last=False,
                 tensor_type=None,
                 timeout=0,
                 worker_init_fn=None,
                 log_statisics=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = partial(collate_fn, tensor_type=tensor_type)
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.log_statisics = log_statisics

        popart.getLogger().info(
            "DataLoader created batchsize:{} num_workers:{} "
            "shuffle:{} tensor_type:{}"
            .format(batch_size, num_workers, shuffle, tensor_type))

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

        self.dataloaderiter = None

    def __iter__(self):

        # Only create the iterator once, as will fork process when created
        # which causes problems if the current process already has
        # synchronisation objects
        if self.dataloaderiter is None:
            self.dataloaderiter = _DataLoaderIter(self)

        return self.dataloaderiter

    def __len__(self):
        return len(self.batch_sampler)
