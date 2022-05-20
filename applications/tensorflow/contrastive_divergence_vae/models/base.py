# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

# coding=utf-8
"""
Base class for probabilistic models
"""
import numpy as np
import os
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import time

from machinable.config_map import ConfigMap
from machinable.dot_dict import DotDict
from utils.ipu_utils import get_device_config, get_ipu_tf
from utils.run_utils import is_nearest_multiple
from utils.train_utils import optimiser_configs

try:
    from tensorflow.python.ipu.optimizers import CrossReplicaOptimizer
except ImportError:
    pass

tfd = tfp.distributions
ipu = get_ipu_tf()

# Defaults - may be overridden in config
N_BATCH_PRINT_LOSS = 1000           # How often to pring loss
N_BATCH_VALIDATION = 100000         # How often to run validation
N_INFEED_BATCHES_DEFAULT = 1000     # How many iterations per session call
N_TE_EVAL_DURING_TRAIN = 5          # How many times to run test evaluation during training
DEFAULT_BATCH_SIZE = 100            # How many samples in each batch

# Lengthscale for LR decay
DEFAULT_EPOCH_LS = 30
N_EPOCHS_REFERENCE = 800


class BaseModel(object):
    """
    Base machinable class for probabilistic models
    """
    def __init__(self, config, node):
        self.config = DotDict(config['args'])
        self.experiment = node

    def on_create(self):
        self.graph = tf.Graph()
        self.set_device(self.config['device'])
        self._t_start = np.nan
        self.latest_checkpoint_path = None

    def set_session(self):
        """
        Create tf session with config depending on device.
        Device config is set via utils.ipu_utils.get_device_config()
        """
        try:
            if not self.device_config['on_ipu']:
                raise AttributeError
            # this will work if version > 0.8.18
            self.device_config['ipu_options']['ipu_options'].configure_ipu_system()
            self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(**self.device_config['sess_options']))
        except AttributeError:
            sess_config = tf.ConfigProto(**self.device_config['sess_options'], **self.device_config['ipu_options'])
            self.sess = tf.Session(config=sess_config, graph=self.graph)

    def build_networks(self):
        raise NotImplementedError('"build_networks()" method must be implemented in child class')

    def build_loss_function(self):
        raise NotImplementedError('"build_loss_function()" method must be implemented in child class')

    def set_network_config(self):
        raise NotImplementedError('"set_network_config()" method must be implemented in child class')

    def set_device(self, device_str):
        """
        Works out tensorflow options and configs based on desired device
        :param device_str: str, desired device, e.g. '/device:IPU:0'
        :return: config_dict, which includes:
            - 'scoper': a function which sets with device scope, i.e.
                with device_config['scoper']():
                    run_this_code()
            - 'sess_opts': kwargs to the tf session config
            - 'ipu_opts': kwargs to the ipu_options in session config. Empty if not on IPU
            - 'maybe_xla_compile': is xla.compile function if on IPU or GPU, else is identity function
        """
        self.device_config = get_device_config(device_str,
                                               num_ipus=self.config.get('n_replicas', 1),
                                               only_ipu=self.config.get('only_ipu'))

    def set_training_config(self):
        """Set some of config relating to parameter optimisation of internal variables"""
        with self.graph.as_default():
            with tf.device('cpu'):
                self.global_step = tf.train.get_or_create_global_step()

        opt_config = optimiser_configs[self.config.training.optimiser_config]
        opt_args = opt_config[1].copy()
        self.optimiser_type = opt_config[0]   # optimiser type (child class to tf.train.Optimizer)

        # Learning rate function: takes global step, total number of steps, and base learning rate;
        # returns lr for this step
        self._learning_rate_func = opt_args.pop('learning_rate_func')
        self.optimiser_kwargs = opt_args    # Additional inputs to optimiser initialisation
        self.base_learning_rate = self.config.training.get('base_lr', opt_args.pop('base_learning_rate'))
        self.lr_kwargs = self.config.training.get('lr_kwargs', ConfigMap()).toDict()

        # If batch size specified in model config it will override that in data config
        self.micro_batch_size = self.config.get('micro_batch_size', DEFAULT_BATCH_SIZE)

        # Introduce ops to record how many epochs have elapsed, and number of training steps
        self.iters = 0
        self.max_iter = self.config.training.n_iter

        # How many replicas to train in parallel
        self.n_replicas = self.config.get('n_replicas', 1)
        assert self.n_replicas == 1 or self.device_config['on_ipu'], \
            'Replicas only supported when running on on IPU'

        # Infeed config
        self.use_infeed = self.config.training.get('use_infeed', True)
        n_infeed_batches_config = self.config.training.get('n_infeed_batches', N_INFEED_BATCHES_DEFAULT)
        self.iters_per_sess_run = n_infeed_batches_config if self.use_infeed else 1

        # Frequency of running validation
        self.n_batch_freq_val = self.config.training.get('n_batch_freq_val', N_BATCH_VALIDATION)

        # Set loss function from config
        self.build_loss_function()
        self.loss_shape = ()

        # Set length of period for LR decay
        if 'epoch_timescale' in self.lr_kwargs:
            epoch_decay_scale = self.lr_kwargs.pop('epoch_timescale')
        else:
            n_epochs = self.max_iter * self.micro_batch_size / self.experiment.data_meta['train_size']

            # Scale decay length proportional to number of epochs
            epoch_decay_scale = DEFAULT_EPOCH_LS * n_epochs / N_EPOCHS_REFERENCE
        self.lr_kwargs['iter_timescale'] = \
            int(epoch_decay_scale * self.experiment.data_meta['train_size'] / self.micro_batch_size)

        # Whether using convolutional architecture (dictates if data flattened)
        self.conv_flag = self.config.network.get('is_conv', False)

    def set_test_config(self):
        # How many batches training between test set evaluation
        self.n_batch_freq_test = \
            self.config.testing.get('n_batch_freq_test',
                                    int(self.max_iter / N_TE_EVAL_DURING_TRAIN))
        self.micro_batch_size_test = \
            self.config.testing.get('micro_batch_size_test', self.micro_batch_size)

    def get_optimiser(self):
        _learning_rate = self.get_current_learning_rate()
        opt_kwargs = self.optimiser_kwargs.copy()
        if 'dtype' in opt_kwargs:
            opt_kwargs['dtype'] = self.experiment.dtype
        if self.n_replicas == 1:
            return self.optimiser_type(_learning_rate, **opt_kwargs)
        else:
            return CrossReplicaOptimizer(self.optimiser_type(_learning_rate, **opt_kwargs))

    def learning_rate_func(self, base_lr, global_step, max_iter, epoch, **kwargs):
        # TODO: recursion through scope tree
        lrs = {}
        if isinstance(base_lr, dict):
            for scope, lr in base_lr.items():
                if isinstance(lr, dict):
                    for subscope, sublr in lr.items():
                        scope_fmt = f'{scope}/{subscope}'
                        lrs[scope_fmt] = self._learning_rate_func(sublr, global_step, max_iter, epoch, **kwargs)
                else:
                    lrs[scope] = self._learning_rate_func(lr, global_step, max_iter, epoch, **kwargs)
            return lrs
        else:
            return self._learning_rate_func(base_lr, global_step, max_iter, epoch, **kwargs)

    def get_epoch(self):
        """Calculates how many epochs have elapsed"""
        batches_per_epoch = self.experiment.dtype_np(self.experiment.data_meta['train_size'] / self.micro_batch_size)
        return tf.cast(self.global_step, self.experiment.dtype) / batches_per_epoch

    def get_current_learning_rate(self):
        """Find what current learning rate should be (if using e.g. cosine decay)"""
        with self.graph.as_default():
            epoch = self.get_epoch()
            return self.learning_rate_func(self.base_learning_rate,
                                           self.global_step,
                                           self.max_iter,
                                           epoch,
                                           **self.lr_kwargs)

    def get_train_ops(self, graph_ops, infeed_queue, i_train, X_micro_batch_train, y_micro_batch_train):
        raise NotImplementedError("'get_train_ops() must be implemented in child class")

    def get_validation_ops(self, graph_ops, i_val, X_micro_batch_val, y_micro_batch_val):
        raise NotImplementedError("'get_validation_ops() must be implemented in child class")

    def get_test_ops(self, graph_ops, i_test, X_micro_batch_test, y_micro_batch_test):
        raise NotImplementedError("'get_test_ops() must be implemented in child class")

    def get_load_data_ops(self):
        """Load minibatches"""
        with self.graph.as_default():
            micro_batch_size = self.micro_batch_size

            if self.device_config['do_xla'] and not self.device_config['on_ipu'] and self.use_infeed:
                # If using infeeds on GPU, and doing XLA compilation,
                # scale batch size by number of loops in each session call
                micro_batch_size *= self.iters_per_sess_run

            # Data iterator ops, format: <indices of batch elements in original dataset>, <batch>
            i_train, X_micro_batch_train, y_micro_batch_train = self.experiment.data_iters['train'].get_next()
            i_val, X_micro_batch_val, y_micro_batch_val = self.experiment.data_iters['validation'].get_next()
            i_test, X_micro_batch_test, y_micro_batch_test = self.experiment.data_iters['test'].get_next()
            if self.use_infeed:
                if self.device_config['on_ipu']:
                    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
                        self.experiment.data_sets['train'])
                    infeed_queue_init = infeed_queue.initializer
                elif self.device_config['do_xla']:
                    infeed_queue = (i_train, X_micro_batch_train, y_micro_batch_train)
                    infeed_queue_init = tf.no_op()
                else:
                    # CPU/GPU - will use tf.while_loop as infeed later
                    # infeed queue init is still no op as train data iter initialised elsewhere
                    infeed_queue = self.experiment.data_iters['train']
                    infeed_queue_init = infeed_queue.initializer
            else:
                infeed_queue = None
                infeed_queue_init = tf.no_op()

        return infeed_queue,\
            infeed_queue_init,\
            (i_train, X_micro_batch_train, y_micro_batch_train),\
            (i_val, X_micro_batch_val, y_micro_batch_val),\
            (i_test, X_micro_batch_test, y_micro_batch_test)

    def get_graph_ops(self):
        """Declare all operations on the graph"""
        ops = {}
        with self.graph.as_default():
            with tf.device('cpu'):
                # Update global step on CPU (slows down IPU)
                ops['incr_global_step'] = tf.assign_add(self.global_step, self.iters_per_sess_run)
            with self.device_config['scoper']():
                ops['lr'] = self.get_current_learning_rate()
                ops['epochs'] = self.get_epoch()

        infeed_queue,\
            infeed_queue_init,\
            (i_train, X_micro_batch_train, y_micro_batch_train),\
            (i_val, X_micro_batch_val, y_micro_batch_val),\
            (i_test, X_micro_batch_test, y_micro_batch_test) = \
            self.get_load_data_ops()

        ops = self.get_train_ops(ops, infeed_queue, i_train, X_micro_batch_train, y_micro_batch_train)
        ops = self.get_validation_ops(ops, i_val, X_micro_batch_val, y_micro_batch_val)
        ops = self.get_test_ops(ops, i_test, X_micro_batch_test, y_micro_batch_test)

        with self.graph.as_default():
            if self.device_config['on_ipu']:
                # Do initialisation ops on CPU to save code space on IPU
                ipu.utils.move_variable_initialization_to_cpu(graph=self.graph)

            # To initialise all variables on graph
            ops['variable_initialiser'] = [tf.global_variables_initializer(), infeed_queue_init]
            ops['vars'] = {v.name: v for v in tf.global_variables()}

            # To save checkpoints:
            self.saver = tf.train.Saver()

        # Fix graph
        self.graph.finalize()
        return ops

    def print_param_info(self):
        vs = self.sess.run(self.graph_ops['vars'])
        n_var = 0
        for v in vs.values():
            n_var += np.prod(v.shape)
        n_param_str = '\n\n-----------\n'\
                      f'Model has {int(n_var)} parameters in total.\n'\
                      '-----------\n'
        var_dim_info = '\n-----------\nVariables and their dimensions are:\n'
        for n, v in vs.items():
            var_dim_info += f'{n}: {v.shape}\n'
        var_dim_info += '-----------\n'
        self.experiment.log.info(n_param_str + var_dim_info)

    def prepare_session(self):
        """Do the tensorflow preamble necessary before training/testing"""
        self.set_training_config()
        self.set_test_config()
        self.build_networks()
        self.graph_ops = self.get_graph_ops()
        self.set_session()

        self.sess.run(self.experiment.data_iters['train'].initializer)
        self.sess.run(self.graph_ops['variable_initialiser'])
        self.print_param_info()

    def batch_scores(self, ops, op_names, max_batches=None, vbose=False):
        """
        Given an iterable of graph operations, and their names, run the operations for many batches, storing the
        results for each. Collate the statistics of these ops over batches into a dict which is returned. If `vbose` is
        True then some text is printed every 10 steps (mainly used for IWAE calculation which is slow

        Currently a little messy, both ops and op_names are expected to be lists of lists (or deeper)
        with the same ordering
        # TODO: make clearer
        """
        # Create dict to store the batch results in
        results = {op_nm: np.array([]) for op_type in op_names for op_nm in op_type}

        def _store_results(new_results, opnames):
            """Recursively delve into new_results lists, writing to the `results` dict"""
            if isinstance(opnames, str):
                # 'opnames' is just single op now
                while np.ndim(new_results) > 1:
                    new_results = new_results[0]
                results[opnames] = np.concatenate((results[opnames], new_results))
            else:
                return [_store_results(rs, nm) for rs, nm in zip(new_results, opnames)]

        # Iterate over batches, calculating and storing output of ops at each step
        # Stop when A. iterator runs out or B. max_batches (specified as argument) reached
        # NOTE if the iterator used in op 1 runs out before that for op 2 then the whole process will stop when op 1
        # triggers OutOfRangeError and not all data for op 2 will be seen
        batch_id = 0
        data_remains = True
        if max_batches is None:
            max_batches = np.inf
        while data_remains and batch_id < max_batches:
            try:
                if batch_id % 10 == 0 and vbose:
                    self.experiment.log.info(f'Evaluating model on batch: {batch_id} of {max_batches}')

                results_batch = self.sess.run(list(ops))
                _store_results(results_batch, op_names)
                batch_id += 1
            except tf.errors.OutOfRangeError:
                # If iterator of any datasetset runs out
                data_remains = False

        # Calculate and return aggregate statistics
        agg_results = {}
        for op_name in results:
            # Weight mean by batch size
            agg_results[f'{op_name}_mean'] = np.mean(results[op_name])
            agg_results[f'{op_name}_std'] = np.std(results[op_name])
            agg_results[f'{op_name}_n_examples'] = len(results[op_name])
            agg_results[f'{op_name}_n_batches'] = batch_id

        return agg_results

    def evaluation_scores(self, ops_sets_names, n_batches=None, verbose=False, iters_to_init=()):
        """Calculate validation metrics for model. Based around self.batch_scores() method."""
        # Initialise the data iterators if necessary
        for split in iters_to_init:
            self.sess.run(self.experiment.data_iters[split].initializer)

        # Find learning rate and number of epochs
        lr = self.sess.run(self.graph_ops['lr'])
        epochs = self.sess.run(self.graph_ops['epochs'])

        # Create `record` dict to store performance metrics
        record = {'n_train_examples': self.iters * self.micro_batch_size,
                  'n_iter': self.iters,
                  'learning_rate': lr,
                  'epochs': epochs}

        # As `ops_set_names` is a dict (and therefore unordered), convert to list to ensure `ops` and `op_names` inputs
        # to `self.batch_scores` are the same
        ordered_ops = [[op_set_nm, op_set] for op_set_nm, op_set in ops_sets_names.items()]

        # Calculate scores and return the `record`
        scores = self.batch_scores(ops=[self.graph_ops[op_set_nm] for op_set_nm, _ in ordered_ops],
                                   op_names=[ops_set for _, ops_set in ordered_ops],
                                   max_batches=n_batches,
                                   vbose=verbose)
        record.update(scores)
        return record

    def train(self):
        # Run the training update, get the loss
        tr_loss = self.sess.run(self.graph_ops['train'])

        self.sess.run(self.graph_ops['incr_global_step'])
        self.iters += self.iters_per_sess_run  # this increments by 1 if not using infeeds
        return tr_loss

    def train_next(self):
        """Single train step, maybe with validation depending on epoch"""
        # NB self.iters_per_sess_run = 1 if not using infeeds
        if self.iters < self.max_iter:

            if is_nearest_multiple(self.iters, self.iters_per_sess_run, self.n_batch_freq_val) and \
                    self.experiment.data_meta['validation_size'] > 0 and \
                    self.experiment.config.validation and \
                    self.iters != 0 and \
                    self.max_iter - self.iters > self.iters_per_sess_run:
                # Evaluate model on validation set
                self.validation()

            if is_nearest_multiple(self.iters, self.iters_per_sess_run, int(self.max_iter / 20)) and \
                    self.iters != 0 and \
                    self.config.get('save_checkpoints', True):    # Don't save CP on first or last iteration

                # Checkpoint session - overwrite previous
                self.save_checkpoint(timestep=-1)

            if is_nearest_multiple(self.iters, self.iters_per_sess_run, self.n_batch_freq_test) and \
                    self.experiment.config.testing and \
                    self.iters != 0 and \
                    self.max_iter - self.iters > self.iters_per_sess_run:
                # Don't do on penultimate iteration - will do testing after training anyway
                self.test()

            if is_nearest_multiple(self.iters, self.iters_per_sess_run, N_BATCH_PRINT_LOSS):
                self.train_time = 0.

            # Do training update and increment global step, time it
            t_before = time.time()
            train_out = self.train()
            t_after = time.time()
            self.train_time += t_after - t_before

            if is_nearest_multiple(self.iters, self.iters_per_sess_run, N_BATCH_PRINT_LOSS):
                # Print training progress and save to file
                tr_out_labelled = dict(zip(self.train_output_labels, train_out))
                self.experiment.log.info(self.train_update_str(n_iter=self.iters,
                                                               tr_out=tr_out_labelled,
                                                               time_diff=self.train_time))
                record_train = {'n_iters': self.iters,
                                'train output': tr_out_labelled,
                                'seconds_taken': self.train_time}
                self.experiment.save_record(record_train, scope='train_speed')

            return True
        else:
            if self.config.get('save_checkpoints', True):
                self.save_checkpoint(timestep=self.iters)

            return False

    def train_update_str(self, n_iter, tr_out, time_diff):
        tr_out_fmt = " | ".join([f"{k}: {v}" for k, v in tr_out.items()])
        return f'Number of iterations: {n_iter} of {self.max_iter} | {tr_out_fmt} | Time taken: {time_diff}'

    def on_save(self, checkpoint_dir, timestep=-1):
        if timestep == -1:
            timestep = None
        cp_path = self.saver.save(self.sess, f'{checkpoint_dir}/cp', global_step=timestep)

        # Save indices of train/validation split
        np.save(f'{checkpoint_dir}/train_idx.npy', self.experiment.idx_train_validation['train'])
        np.save(f'{checkpoint_dir}/validation_idx.npy', self.experiment.idx_train_validation['validation'])

        return cp_path

    def on_restore(self, checkpoint_file):
        self.latest_checkpoint_path = checkpoint_file
        self.saver.restore(self.sess, checkpoint_file)
        self.iters = self.sess.run(self.global_step)

    def validation(self):
        raise NotImplementedError('Model evaluation not implemented in base class.')

    def test(self, *args, **kwargs):
        raise NotImplementedError('Model testing not implemented in base class.')

    def config_dtype(self, dtype):
        """Parses string of datatype to datatype object"""
        return eval(dtype)

    def save_checkpoint(self, path=None, timestep=None):
        if path is None:
            if not self.experiment.observer:
                raise ValueError('You need to specify a checkpoint path')

            fs, basepath = self.experiment.observer.config['storage'].split('://')
            checkpoint_path = self.experiment.observer.get_path('checkpoints', create=True)
            path = os.path.join(os.path.expanduser(basepath), checkpoint_path)

        return self.on_save(path, timestep)
