# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import pickle
import random
import logging
import argparse
import multiprocessing

import numpy as np
import poptorch
import popdist
import popdist.poptorch
import torch
import torch.nn as nn
import horovod.torch as hvd
import torch.nn.utils.rnn as rnn_utils

from torch import float16, float32
from torch.utils.data import Dataset, IterableDataset
from poptorch.optim import LAMB, AdamW, Adam
from tfrecord.reader import tfrecord_loader
from transformers import (get_constant_schedule,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

TFRECORD_KEYS = ['input_ids']  # Torch Model Keys


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def expand_glob_files(files):
    result = []
    for filepath in files:
        expanded = glob.glob(filepath)
        if len(expanded) < 1:
            raise FileNotFoundError(f"Could not find file: {filepath}")
        result += expanded
    return result


class TFRecordPretrainingDataset(IterableDataset):
    """
    Preprocessed GPT2 pretraining dataset read from TFRecord files.


    This Dataset is compatible with multiprocessing. Each Dataloader worker
    will only read a shard of each TFRecord file, which will speed up the Dataloader
    and ensure no worker loads the same data as another worker. You are strongly
    advised to use a large number (e.g. 64) of dataloader workers because firstly,
    more workers could support high throughput, and secondly, more workers could
    give us more stochasticity and thus better convergence.


    Parameters
    ----------
    files: List of TFRecord files containing the preprocessed pretraining data
    shuffle: Shuffle the data?
    """

    def __init__(self,
                 input_files,
                 shuffle=True):
        self.files = expand_glob_files(input_files)
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.file_index = 0
        self.reader = iter([])

    def samples_per_file(self, filename):
        index_filename = filename.replace(".tfrecord", ".index")
        count = sum(1 for _ in open(index_filename))
        return count

    def __len__(self):
        if getattr(self, "_len", None) is None:
            pool = multiprocessing.Pool(
                min(multiprocessing.cpu_count(), len(self.files)))
            num_samples = pool.map(self.samples_per_file, self.files)
            pool.close()
            pool.join()
            self._len = sum(num_samples)
        return self._len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if popdist.isPopdistEnvSet():
                self.worker_id = worker_info.id + \
                    worker_info.num_workers * popdist.getInstanceIndex()
                self.shard = worker_info.id + worker_info.num_workers * \
                    popdist.getInstanceIndex(), worker_info.num_workers * popdist.getNumInstances()
            else:
                self.worker_id = worker_info.id
                self.shard = worker_info.id, worker_info.num_workers
        else:
            self.shard = None
        self.reset()
        if self.shuffle:
            np.random.shuffle(self.files)
        return self

    def __next__(self):
        try:
            datum = next(self.reader)
        except StopIteration:
            if self.file_index >= len(self.files):
                raise StopIteration
            self.reader = tfrecord_loader(self.files[self.file_index],
                                          self.files[self.file_index].replace(
                                              ".tfrecord", ".index"),
                                          list(TFRECORD_KEYS),
                                          self.shard)
            self.file_index += 1
            datum = next(self.reader)
        input_ids = torch.tensor(datum[TFRECORD_KEYS[0]], dtype=torch.long)
        return input_ids


class MyDataset(Dataset):
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def load_dataset(logger, args, vocab_size):
    """
    load train and valid dataset
    """
    logger("loading training dataset and validating dataset")

    if args.dataset == 'generated':
        num_instances = args.popdist_size if args.use_popdist else 1
        generated = np.random.randint(low=1, high=vocab_size,
                                      size=(4 * num_instances * args.replication_factor *
                                            args.device_iterations * args.batch_size * args.gradient_accumulation,
                                            args.max_len + 1))
        train_dataset = MyDataset(generated, args.max_len + 1)
        val_dataset = MyDataset(generated, args.max_len + 1)
    elif args.dataset == 'tfrecord':
        train_dataset = TFRecordPretrainingDataset(args.input_files[:])
        val_dataset = TFRecordPretrainingDataset(args.input_files[-1:])
    elif args.dataset == 'mmap':
        from data.indexed_dataset import make_indexed_dataset, GPTDataset
        data_prefix = args.input_files
        indexed_dataset = make_indexed_dataset(data_prefix)
        total_num_of_documents = indexed_dataset.sizes.shape[0]
        documents = np.arange(
            start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
        global_batch_size = args.batch_size * args.gradient_accumulation * args.replication_factor
        num_train_samples = args.training_steps * global_batch_size
        train_dataset = GPTDataset(args, data_prefix, documents[:int(
            total_num_of_documents*0.997)], indexed_dataset, num_samples=num_train_samples)
        val_dataset = GPTDataset(args, data_prefix, documents[int(
            total_num_of_documents*0.997):], indexed_dataset, num_epochs=1)
    else:
        try:
            with open(args.input_files, "rb") as f:
                input_list = pickle.load(f)

            samples = []
            for article in input_list:
                start_point = 0
                while start_point < len(article) - args.max_len:
                    samples.append(
                        article[start_point: start_point + args.max_len])
                    start_point += args.stride
                if start_point < len(article) and len(article) >= (args.max_len // 2):
                    samples.append(article[len(article) - args.max_len:])
            random.shuffle(samples)

            # split train and valid dataset
            val_num = args.val_num
            input_list_train = samples[val_num:]
            input_list_val = samples[:val_num]

            train_dataset = MyDataset(input_list_train, args.max_len)
            val_dataset = MyDataset(input_list_val, args.max_len)
        except:
            raise RuntimeError(
                f"Unknown dataset '{args.input_files}'.")

    return train_dataset, val_dataset


class GeneratedPretrainingDataset(Dataset):
    def __init__(self, vocab_size, sequence_length, seed=42):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.data = self.generate_data()

    def generate_data(self):
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            input_ids = torch.randint(
                0, self.vocab_size, [self.sequence_length], dtype=torch.long)
            label = input_ids
        return input_ids, label

    def __len__(self):
        return self.length

    def __getitem__(self, __):
        return self.data


def get_generated_datum(config, vocab_size):
    samples_per_step = config.replication_factor * \
        config.gradient_accumulation * config.batch_size * config.device_iterations
    result = []
    generated_dataset = GeneratedPretrainingDataset(vocab_size, config.max_len)
    data = (generated_dataset[i] for i in range(samples_per_step))
    for batches in zip(*data):
        result.append(torch.stack(batches))
    return result


def calculate_acc(logit, labels, ignore_index=-100, reduction='mean'):
    mask = (labels != ignore_index).float()
    non_pad_mask = mask.sum(-1).unsqueeze(-1)
    if reduction == 'sum':
        return (logit.argmax(dim=-1) == labels).float().mul(mask).sum(-1)
    return (logit.argmax(dim=-1) == labels).float().mul(mask).div(non_pad_mask).sum(-1).mean()


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(
        batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(
        batch, batch_first=True, padding_value=-100)
    return input_ids, labels


class _WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)


def logger(msg):
    if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
        logging.info(msg)


def cycle(iterator):
    """
    Loop `iterator` forever
    """
    while True:
        for item in iterator:
            yield item


def get_lr_scheduler(optimizer,
                     scheduler_type,
                     lr_warmup=None,
                     num_steps=None):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, lr_warmup, num_steps)
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, lr_warmup, num_steps)
    else:
        raise ValueError("Unknown scheduler_type:", scheduler_type)

    # Initialize step as Poptorch does not call optimizer.step() explicitly
    optimizer._step_count = 1

    return scheduler


def get_optimizer(optimizer, weight_decay, learning_rate, loss_scaling, model, use_popdist=False,
                  enable_half_first_order_momentum=True):
    # Do not apply weight_decay for one-dimensional parameters
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {"params": regularized_params, "weight_decay": weight_decay},
        {"params": non_regularized_params, "weight_decay": 0}
    ]

    first_order_type = float16 if enable_half_first_order_momentum else float32

    if optimizer == "AdamW":
        optimizer = AdamW(params,
                          lr=learning_rate,
                          weight_decay=0.01,
                          eps=1e-6,
                          bias_correction=False,
                          loss_scaling=loss_scaling,
                          accum_type=float16,
                          first_order_momentum_accum_type=first_order_type,
                          second_order_momentum_accum_type=float32)
    elif optimizer == "Adam":
        optimizer = Adam(params,
                         lr=learning_rate,
                         weight_decay=0.01,
                         eps=1e-6,
                         loss_scaling=loss_scaling,
                         accum_type=float16,
                         first_order_momentum_accum_type=first_order_type,
                         second_order_momentum_accum_type=float32)
    elif optimizer == "LAMBNoBiasCorrection":
        optimizer = LAMB(params,
                         lr=learning_rate,
                         weight_decay=0,
                         eps=1e-6,
                         loss_scaling=loss_scaling,
                         max_weight_norm=None,
                         accum_type=float16,
                         first_order_momentum_accum_type=first_order_type,
                         second_order_momentum_accum_type=float32,
                         bias_correction=False)
    elif optimizer == "LAMB":
        optimizer = LAMB(params,
                         lr=learning_rate,
                         weight_decay=0,
                         eps=1e-6,
                         loss_scaling=loss_scaling,
                         max_weight_norm=None,
                         accum_type=float16,
                         first_order_momentum_accum_type=first_order_type,
                         second_order_momentum_accum_type=float32,
                         bias_correction=True)
    else:
        raise ValueError("Unknown Optimizer:", optimizer)

    # Make optimizers distributed
    if use_popdist:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    return optimizer


def sync_metrics(outputs, factor=1, average=True):
    if popdist.isPopdistEnvSet():
        if isinstance(outputs, float):
            return float(hvd.allreduce(torch.Tensor([outputs]), average=average).item())
        else:
            return [hvd.allreduce(output.div(factor), average=average).mean().item() for output in outputs]
    else:
        if isinstance(outputs, float):
            return outputs
        else:
            return [output.div(factor).mean().item() for output in outputs]


def outline_attribute(module: nn.Module, value: str):
    """Adds an attribute to a module. This attribute will be used
        when comparing operation equivalence in outlining. For example:

        layer1 = nn.Linear(...)
        layer2 = nn.Linear(...)
        layer3 = nn.Linear(...)
        layer4 = nn.Linear(...)
        outline_attribute(layer1, "A")
        outline_attribute(layer2, "A")
        outline_attribute(layer3, "B")

        The code for layer1 can be reused for layer2.
        But it can't be used for layer3 or layer4.
    """
    context = poptorch.Attribute(__outline={"layer": value})

    def enable(*args):
        context.__enter__()

    def disable(*args):
        context.__exit__(None, None, None)

    module.register_forward_pre_hook(enable)
    module.register_forward_hook(disable)


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
        recomputed"""

    def recompute_outputs(module, inputs, outputs):
        return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)

    module.register_forward_hook(recompute_outputs)


class SerializedLinear(nn.Linear):
    def __init__(self, in_features, out_features, factor, bias=False,
                 mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_features,)
        output = poptorch.serializedMatMul(
            x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output.view(*size_out)


class SerializedEmbedding(nn.Module):
    """
    Wrapper for `nn.Embedding` layer that performs the embedding look-up into
    smaller serialized steps in order to reduce memory in the embedding gradient
    calculation.

    Args:
        embedding: A `nn.Embedding` to wrap
        serialization_factor: The number of serialized embedding look-ups
    """

    def __init__(self, embedding: nn.Embedding, serialization_factor: int):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.num_embeddings = embedding.num_embeddings

        # Num embeddings should be divisible by the serialization factor
        assert self.num_embeddings % self.serialization_factor == 0
        self.split_size = self.num_embeddings // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [nn.Embedding.from_pretrained(embedding.weight[i * self.split_size:(i + 1) * self.split_size, :].detach(),
                                          freeze=False,
                                          padding_idx=embedding.padding_idx if i == 0 else None)
             for i in range(self.serialization_factor)])

    def deserialize(self):
        """
        Deserialize the internal wrapped embedding layer and return it as a
        `nn.Embedding` object.

        Returns:
            `nn.Embedding` layer
        """
        return nn.Embedding.from_pretrained(torch.vstack([l.weight for l in self.split_embeddings]), padding_idx=0)

    def forward(self, indices):
        # iterate through the splits
        x_sum = None
        for i in range(self.serialization_factor):
            # mask out the indices not in this split
            split_indices = indices - i * self.split_size
            mask = (split_indices >= 0) * (split_indices < self.split_size)
            mask = mask.detach()
            split_indices *= mask

            # do the embedding lookup
            x = self.split_embeddings[i](split_indices)

            # multiply the output by mask
            x *= mask.unsqueeze(-1)

            # add to partial
            if x_sum is not None:
                x_sum += x
            else:
                x_sum = x
        return x_sum
