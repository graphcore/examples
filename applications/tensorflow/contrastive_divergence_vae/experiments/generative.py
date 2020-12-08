# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Generic functionality for generative modelling experiments
"""
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np       # Necessary for parsing dtypes
import tensorflow as tf  # Necessary for parsing dtypes

from machinable.dot_dict import DotDict
from machinable.observer import Observer

# Add models module to path
models_path = Path(Path(__file__).absolute().parent.parent)
sys.path.append(str(models_path))
from models.vae.vcd_vae import VCDVAE
from utils.data_utils import download_dataset, make_iterator_from_np, preprocess_np_inputs, split_xy_array

DEFAULT_BATCH_SIZE = 100

# For parsing dtype from string in config
TF_DTYPES = {'tf.float16': tf.float16, 'tf.float32': tf.float32}
NP_DTYPES = {'np.float16': np.float16, 'np.float32': np.float32}


class Generative(object):
    def __init__(self, config):
        self.config = DotDict(config['args'])
        self.observer = Observer({'storage': self.config['results_location'],
                                  'group': self.config['task_name'],
                                  'log': self.config['log']})

    def execute(self, model_config=None):
        self.on_create()
        self.model = VCDVAE(config=model_config[0],
                            node=self)
        self.model.observer = self.observer
        self.model.experiment = self
        self.model.on_create()
        self.on_execute()

    def on_create(self):
        self.dtype = TF_DTYPES[self.config.data.formats.tf]
        self.dtype_np = NP_DTYPES[self.config.data.formats.np]
        self.idx_train_validation = None

    def on_execute(self):

        if self.config.get('checkpoint_path', None) is not None:
            self.log.info(f'Loading train and validation split idx.')
            self.load_train_validation_idx()

        self.prepare_data()
        self.model.prepare_session()

        if self.config.get('checkpoint_path', None) is not None:
            self.log.info(f'Restoring session: {self.config.checkpoint_path}')
            self.model.on_restore(self.config.checkpoint_path)
        if self.config.training:
            self.log.info('Starting training...')
            self.train_model()
            self.log.info('Finished training.')
        if self.config.testing:
            self.log.info('Testing the model...')
            self.test_model()
            self.log.info('Done testing.')
        if self.config.validation:
            self.log.info('Doing final validation...')
            self.model.validation()
            self.log.info('Finished validation.')

    def load_data(self, loading_config):
        self.np_data = {'train_and_validation': {}, 'test': {}}
        (self.np_data['train_and_validation']['x'], self.np_data['train_and_validation']['y']),\
            (self.np_data['test']['x'], self.np_data['test']['y']) = \
            download_dataset(dataset_name=loading_config['set'])

    def preprocess_data(self, data_config):
        # Split train set into train and validation
        n_val = data_config.n_validation

        if self.config.validation:
            assert n_val > 0, 'Need to specify validation set with > 0 to do validation'
        else:
            assert n_val == 0, 'Not running validation but still creating validation set'

        n_train = self.np_data['train_and_validation']['x'].shape[0] - n_val
        (self.np_data['train'], self.np_data['validation']), idx_train_validation = \
            split_xy_array(self.np_data['train_and_validation'],
                           sizes=[n_train],
                           split_idx=self.idx_train_validation,
                           shuffle_before=data_config.loading.get('shuffle_pre_split', True))

        if n_val == 0:
            # Do not preproces empty dataset
            self.np_data.pop('validation')

        # Make record of indices of train and validation split
        self.idx_train_validation = dict(zip(['train', 'validation'], idx_train_validation))

        # Build prepocessing function, apply it to tr/val/te
        preproc = partial(preprocess_np_inputs,
                          datatype=self.dtype_np,
                          flatten_images=True)

        self.np_data = {split: {'x': preproc(np_arrs['x']), 'y': np_arrs['y'].astype(np.int32)}
                        for split, np_arrs in self.np_data.items()}

        # Data centring
        if data_config.loading.get('subtract_train_mean', False):
            self.np_data = {n: {'x': arrs['x'] - self.np_data['train']['x'].mean(axis=0), 'y': arrs['y']}
                            for n, arrs in self.np_data.items()}

        if n_val == 0:
            self.np_data['validation'] = {'x': np.array(()), 'y': np.array(())}

        # Can override batch size in data config if specified in model
        batch_size = self.model.config.get('batch_size', DEFAULT_BATCH_SIZE)
        batch_size_te = self.model.config.testing.get('batch_size_te', batch_size)

        # If using mock infeed on CPU/GPU need to multiply batchsize by number of infeed loops
        use_infeed = self.model.config.training.get('use_infeed', False)
        n_infeed_loops = self.model.config.training.get('n_infeed_batches', 1)
        scale_batch = use_infeed and not self.model.device_config['on_ipu'] and self.model.device_config['do_xla']
        batch_factor = n_infeed_loops if scale_batch else 1
        batch_sizes = {'train': batch_size * batch_factor, 'validation': batch_size_te, 'test': batch_size_te}

        # Only drop remainder of training set
        drop_remainders = {'train': True, 'validation': False, 'test': False}
        repeats = {'train': True, 'validation': False, 'test': False}

        # Convert numpy arrays into tf datasets and iterators
        tf_data = {split: make_iterator_from_np(np_arrays=np_arrs,
                                                shuffle=True,
                                                batch_size=batch_sizes[split],
                                                rand_binarise=False,
                                                drop_remain=drop_remainders[split],
                                                repeat=repeats[split])
                   for split, np_arrs in self.np_data.items() if split != 'train_and_validation'}
        self.data_iters = {split: d[0] for split, d in tf_data.items()}
        self.data_sets = {split: d[1] for split, d in tf_data.items()}

    def load_train_validation_idx(self):
        # Load indices of original train/validation split
        checkpoint_dir = os.path.dirname(self.config.checkpoint_path)
        if os.path.exists(f'{checkpoint_dir}/train_idx.npy'):
            self.idx_train_validation = {}
            self.idx_train_validation['train'] = np.load(f'{checkpoint_dir}/train_idx.npy')
            self.idx_train_validation['validation'] = np.load(f'{checkpoint_dir}/validation_idx.npy')

    def set_metadata(self, loading_config):
        """Make record of the important quantities of the dataset splits for later use"""
        self.data_meta = {}
        self.data_meta['data_dims'] = list(self.np_data['train']['x'].shape[1:])
        self.data_meta['train_size'] = self.np_data['train']['x'].shape[0]
        self.data_meta['validation_size'] = self.np_data['validation']['x'].shape[0]
        self.data_meta['test_size'] = self.np_data['test']['x'].shape[0]
        self.data_meta['train_mean'] = np.mean(self.np_data['train']['x'], axis=0)
        self.data_meta['name'] = loading_config['set']
        self.data_meta['n_classes'] = len(np.unique(self.np_data['train']['y']))
        self.data_meta['train_log_var'] = np.log(np.var(self.np_data['train']['x']))

    def save_record(self, record, scope=None):
        if scope is None:
            writer = self.record
        else:
            writer = self.observer.get_record_writer(scope)
        for name, value in record.items():
            writer[name] = value
        writer['_path'] = self.observer.config['storage'] + self.observer.get_path()
        writer.save()

    def prepare_data(self):
        with self.model.graph.as_default():
            self.load_data(self.config.data.loading)
            self.preprocess_data(self.config.data)
            self.set_metadata(self.config.data.loading)

    def train_model(self):
        unfinished_train = True
        while unfinished_train:
            unfinished_train = self.model.train_next()

    def config_dtype(self, dtype):
        return eval(dtype)

    def test_model(self):
        self.log.info('Testing model...')
        self.model.test()
        self.log.info('Done testing.')

    @property
    def log(self):
        return self.observer.log
