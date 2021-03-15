# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training CNNs on Graphcore IPUs.

See the README and the --help option for more information.
"""
import tensorflow.compat.v1 as tf
import os
import re
import time
import math
import argparse
import datetime
import random
from socket import gethostname
from collections import deque, OrderedDict, namedtuple
from contextlib import ExitStack
from functools import partial
import numpy as np
import sys
import importlib
from pathlib import Path
import validation
import log as logging
from tensorflow.python import ipu
from ipu_utils import get_config
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu.autoshard import automatic_sharding
from tensorflow.python.ipu import loops, ipu_infeed_queue, ipu_outfeed_queue, ipu_compiler
from tensorflow.python.ipu.utils import reset_ipu_seed
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import ipu_multi_replica_strategy
from ipu_optimizer import IPUOptimizer
from tensorflow.python.ipu.scopes import ipu_scope
import Datasets.data as dataset
from Datasets import imagenet_dataset
from weight_avg import average_ckpts, save_ckpt
from Models.batch_norm import add_bn_moving_average_updates
import popdist
import popdist.tensorflow
from Datasets import augmentations
import json
import configurations

DATASET_CONSTANTS = dataset.DATASET_CONSTANTS
config_file = Path(Path(__file__).parent, Path("configs.yml"))

GraphOps = namedtuple(
    'graphOps', ['graph',
                 'session',
                 'init',
                 'ops',
                 'placeholders',
                 'iterator',
                 'outfeed',
                 'saver',
                 'profile'])

pipeline_schedule_options = [str(p).split(".")[-1] for p in list(pipelining_ops.PipelineSchedule)]


def integer_labels_to_dense(data_dict, opts, num_classes):
    """
    Function tranforms integer labels into their dense representation.
    This takes into acount the data augmentations of label smoothing, mixup and cutmix

    :param data_dict:
    :param opts: global options
    :param num_classes: number of classes for the classification problem
    :return: tensor representing the target labels, of shape [batch_size, num_classes]
    """
    smooth_negatives = opts["label_smoothing"] / (num_classes - 1)
    smooth_positives = 1.0 - opts["label_smoothing"]
    smoothed_one_hot_fn = partial(tf.one_hot, depth=num_classes, on_value=smooth_positives, off_value=smooth_negatives)
    smoothed_labels = smoothed_one_hot_fn(data_dict['label'])
    if opts["mixup_alpha"] > 0:
        # linear mix of the one-hot labels
        smoothed_mixup_labels = smoothed_one_hot_fn(data_dict['label_mixed_up'])
        # mix must be broadcastable to [batch_size, n_labels]
        mix = tf.cast(tf.squeeze(data_dict["mixup_coefficients"]), smoothed_labels.dtype)[:, None]
        smoothed_labels = mix * smoothed_labels + (1. - mix) * smoothed_mixup_labels
    if opts['cutmix_lambda'] < 1.:
        smoothed_cutmix_labels = smoothed_one_hot_fn(data_dict['cutmix_label'])
        cutmix_lambda = tf.cast(tf.squeeze(data_dict["cutmix_lambda"]), smoothed_labels.dtype)[:, None]
        smoothed_labels = cutmix_lambda * smoothed_labels + (1. - cutmix_lambda) * smoothed_cutmix_labels
    if opts['cutmix_lambda'] < 1. and opts['mixup_alpha'] > 0:
        # we have 4 labels per one image, split into four parts:
        # 1. the original label, multiplied by 'mix' and 'cutout_lambda'
        one_hot_1 = mix * cutmix_lambda * smoothed_one_hot_fn(data_dict['label'])
        # 2. the mixup label, 'label2', multiplied by 'mix' and '1 - cutout'
        one_hot_2 = (1. - mix) * cutmix_lambda * smoothed_one_hot_fn(data_dict['label_mixed_up'])
        # now, the image that will be pasted in with cutmix is a combination of two images itself -- with the
        # mixing coefficient given in 'mix2'
        mix2 = tf.cast(tf.squeeze(data_dict["mixup_coefficients_2"]), smoothed_labels.dtype)[:, None]
        # 3. the foreground image of the cut-in patch
        one_hot_3 = mix2 * (1. - cutmix_lambda) * smoothed_one_hot_fn(data_dict['cutmix_label'])
        # 4. the background image of the cut-in patch
        one_hot_4 = (1. - mix2) * (1. - cutmix_lambda) * smoothed_one_hot_fn(data_dict['cutmix_label2'])
        smoothed_labels = one_hot_1 + one_hot_2 + one_hot_3 + one_hot_4
    return smoothed_labels


def calculate_loss(logits, data_dict, opts):
    predictions = tf.argmax(logits, 1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, data_dict['label']), tf.float16))

    if opts["label_smoothing"] > 0 or opts["mixup_alpha"] > 0 or opts['cutmix_lambda'] < 1.:
        num_classes = int(logits.shape[1])
        smoothed_labels = integer_labels_to_dense(data_dict, opts, num_classes=num_classes)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=smoothed_labels))
    else:
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                      labels=data_dict['label']))
    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss, cross_entropy, accuracy


def get_optimizer(opts):
    moving_avg_update = partial(add_bn_moving_average_updates, momentum=opts.get('BN_decay'))
    if opts['optimiser'] == 'SGD':
        opt_fun = moving_avg_update(tf.train.GradientDescentOptimizer)
    elif opts['optimiser'] == 'momentum':
        opt_fun = partial(moving_avg_update(tf.train.MomentumOptimizer),
                          momentum=opts['momentum'])
    elif opts['optimiser'] == 'RMSprop':
        opt_fun = partial(moving_avg_update(tf.train.RMSPropOptimizer), momentum=opts['momentum'],
                          decay=opts['rmsprop_decay'], epsilon=opts['rmsprop_epsilon'])
    elif opts['optimiser'] == 'LARS':
        from tensorflow.contrib.opt.python.training.lars_optimizer import LARSOptimizer
        opts['lars_skip_list'] += ['batch_norm/moving_']
        opt_fun = partial(moving_avg_update(LARSOptimizer),
                          weight_decay=opts['lars_weight_decay'],
                          eeta=opts['lars_eeta'],
                          epsilon=opts['lars_epsilon'],
                          momentum=opts['momentum'],
                          skip_list=opts['lars_skip_list'])
    else:
        raise ValueError("Optimizer {} not recognised".format(opts['optimiser']))

    wd_exclude = opts["wd_exclude"] if "wd_exclude" in opts.keys() else []

    wd_exclude += ['batch_norm/moving_']  # always exclude moving averages from weight decay

    # get variables to include in training
    if opts["variable_filter"]:
        var_list = [v for v in tf.trainable_variables() if any(s in v.name for s in opts["variable_filter"])]
    else:
        var_list = tf.trainable_variables()

    def filter_fn(name):
        return not any(s in name for s in wd_exclude)

    return lambda lr: IPUOptimizer(opt_fun(lr),
                                   sharded=opts["shards"] > 1 and not opts['pipeline'],
                                   replicas=opts["total_replicas"],
                                   gradient_accumulation_count=opts["gradient_accumulation_count"],
                                   pipelining=opts['pipeline'],
                                   grad_scale=opts["grad_scale"],
                                   weight_decay=opts["weight_decay"] * opts['loss_scaling'],
                                   weight_decay_filter_fn=filter_fn,
                                   var_list=var_list)


def calculate_and_apply_gradients(loss, opts=None, learning_rate=None):
    optimizer = get_optimizer(opts)(learning_rate / opts['lr_scale'])
    grads_and_vars = optimizer.compute_gradients(loss * opts['loss_scaling'])
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        return learning_rate, optimizer.apply_gradients(grads_and_vars)


def basic_training_step(data_dict, model, opts, learning_rate):
    """
    A basic training step that will work on all hardware
    """
    if opts['dataset'] == 'imagenet' and not opts['hostside_norm']:
        data_dict['image'] = imagenet_dataset.accelerator_side_preprocessing(data_dict['image'], opts=opts)
    if not opts['hostside_image_mixing']:
        # apply augmentation on the accelerator
        if opts['mixup_alpha'] > 0.:
            data_dict = augmentations.mixup_image(data_dict)
        if opts['cutmix_lambda'] < 1.:
            data_dict = augmentations.cutmix(data_dict, cutmix_lambda=opts['cutmix_lambda'])

    image, label = data_dict['image'], data_dict['label']
    logits = model(opts, training=True, image=image)
    loss, cross_entropy, accuracy = calculate_loss(logits, data_dict, opts)

    learning_rate, train_op = calculate_and_apply_gradients(loss, opts, learning_rate=learning_rate)
    if opts['shards'] > 1:
        def filter(edge):
            return (any(f in e for e in edge for f in opts["sharding_exclude_filter"]) or
                    not any(f in e for e in edge for f in opts["sharding_include_filter"]))

        automatic_sharding(opts['shards'], image, cross_entropy, edge_filter=filter)

    return loss, cross_entropy, accuracy, learning_rate, train_op


def basic_pipelined_training_step(model, opts, learning_rate, infeed, outfeed, iterations_per_step=1):

    def first_stage(learning_rate, data_dict, pipeline_stage=None, idx=None):
        if opts['dataset'] == 'imagenet' and not opts['hostside_norm'] and idx == 0:
            data_dict['image'] = imagenet_dataset.accelerator_side_preprocessing(data_dict['image'], opts=opts)
        if not opts['hostside_image_mixing']:
            if opts['mixup_alpha'] > 0:
                data_dict = augmentations.mixup_image(data_dict)
            if opts['cutmix_lambda'] < 1.:
                data_dict = augmentations.cutmix(data_dict, cutmix_lambda=opts['cutmix_lambda'])
        image, label = data_dict['image'], data_dict['label']

        outputs = [learning_rate, pipeline_stage(image), label]

        # if both are applied, the ordering is important for unpacking these later
        if opts['mixup_alpha'] > 0.:
            outputs += [data_dict['label_mixed_up'], data_dict['mixup_coefficients']]
        if opts['cutmix_lambda'] < 1.:
            outputs += [data_dict['cutmix_label'], data_dict['cutmix_lambda']]
        if opts['mixup_alpha'] > 0. and opts['cutmix_lambda'] < 1.:
            outputs += [data_dict['cutmix_label2'], data_dict['mixup_coefficients_2']]

        return outputs

    def later_stage(learning_rate, x, label, *inputs, pipeline_stage=None, final_stage=False):
        inputs = list(inputs)
        x = pipeline_stage(x)
        if not final_stage:
            return [learning_rate, x, label] + inputs
        else:
            data_dict = {'label': label}
            if opts['mixup_alpha'] > 0.:
                data_dict.update({'label_mixed_up': inputs.pop(0), 'mixup_coefficients': inputs.pop(0)})
            if opts['cutmix_lambda'] < 1.:
                data_dict.update({'cutmix_label': inputs.pop(0), 'cutmix_lambda': inputs.pop(0)})
            if opts['cutmix_lambda'] < 1. and opts['mixup_alpha'] > 0.:
                data_dict.update({'cutmix_label2': inputs.pop(0), 'mixup_coefficients_2': inputs.pop(0)})

            loss, cross_entropy, accuracy = calculate_loss(x, data_dict, opts)
            # note: would ideally add in scaling in optimizer_function for learning_rate
            return loss, cross_entropy, accuracy, learning_rate / opts["lr_scale"]

    model_stages = model(opts)
    computational_stages = [partial(first_stage, pipeline_stage=model_stages[0], idx=0)]
    computational_stages += [partial(later_stage, pipeline_stage=model_stages[idx+1],
                                     final_stage=idx == len(model_stages)-2) for idx in range(len(model_stages) - 1)]

    def optimizer_function(loss, _, __, lr):
        optimizer = get_optimizer(opts)(lr)
        return pipelining_ops.OptimizerFunctionOutput(optimizer, loss * opts["loss_scaling"])

    options = None
    amps = opts['available_memory_proportion']
    if amps and len(amps) > 1:
        # Map values to the different pipeline stages
        options = []
        for i in range(len(amps) // 2):
            options.append(pipelining_ops.PipelineStageOptions({"availableMemoryProportion": amps[2 * i]},
                                                               {"availableMemoryProportion": amps[2 * i + 1]}))

    return pipelining_ops.pipeline(computational_stages=computational_stages,
                                   gradient_accumulation_count=int(opts['gradient_accumulation_count']),
                                   repeat_count=iterations_per_step,
                                   inputs=[learning_rate],
                                   infeed_queue=infeed,
                                   outfeed_queue=outfeed,
                                   optimizer_function=optimizer_function,
                                   forward_propagation_stages_poplar_options=options,
                                   backward_propagation_stages_poplar_options=options,
                                   pipeline_schedule=next(p for p in list(pipelining_ops.PipelineSchedule)
                                                          if opts["pipeline_schedule"] == str(p).split(".")[-1]),
                                   offload_weight_update_variables=not opts['disable_variable_offloading'],
                                   name="Pipeline")


def distributed_per_replica(function):
    """Run the function with the distribution strategy (if any) in a per-replica context."""

    def wrapper(*arguments):
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            return strategy.experimental_run_v2(function, args=arguments)
        else:
            return function(*arguments)

    return wrapper


@distributed_per_replica
def training_step_with_infeeds_and_outfeeds(train_iterator, outfeed, model, opts, learning_rate, iterations_per_step=1):
    """
    Training step that uses an infeed loop with outfeeds. This runs 'iterations_per_step' steps per session call. This leads to
    significant speed ups on IPU. Not compatible with running on CPU or GPU.
    """

    if opts['pipeline']:
        training_step = partial(basic_pipelined_training_step,
                                model=model.staged_model,
                                opts=opts,
                                learning_rate=learning_rate,
                                infeed=train_iterator,
                                outfeed=outfeed,
                                iterations_per_step=iterations_per_step)

        return ipu_compiler.compile(training_step, [])
    else:
        training_step = partial(basic_training_step,
                                model=model.Model,
                                opts=opts,
                                learning_rate=learning_rate)

        def training_step_loop(data_dict, outfeed=None):
            loss, cross_ent, accuracy, lr_out, apply_grads = training_step(data_dict)
            outfeed = outfeed.enqueue((loss, cross_ent, accuracy, lr_out))
            return outfeed, apply_grads

        def compiled_fn():
            return loops.repeat(iterations_per_step,
                                partial(training_step_loop, outfeed=outfeed),
                                [],
                                train_iterator)

        return ipu_compiler.compile(compiled_fn, [])


def configure_distribution(opts, sess_config):
    """
    Creates the distribution strategy, updates the given session configuration
    accordingly, and starts a distributed server that allows the workers to connect.
    Returns the strategy, the session target address and the updated session configuration.
    """

    cluster = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec=tf.train.ClusterSpec(opts['distributed_cluster']),
        task_id=opts['distributed_worker_index'],
        task_type="worker")

    strategy = ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategy(cluster)
    sess_config = strategy.update_config_proto(sess_config)
    server = tf.distribute.Server(cluster.cluster_spec(),
                                  job_name=cluster.task_type,
                                  task_index=cluster.task_id,
                                  protocol=cluster.rpc_layer,
                                  config=sess_config)

    return strategy, server.target, sess_config


def create_popdist_strategy():
    """
    Creates a distribution strategy for use with popdist. We use the
    Horovod-based IPUMultiReplicaStrategy. Horovod is used for the initial
    broadcast of the weights and when reductions are requested on the host.
    """
    hvd.init()

    # We add the IPU cross replica reductions explicitly in the IPUOptimizer,
    # so disable them in the IPUMultiReplicaStrategy.
    return ipu_multi_replica_strategy.IPUMultiReplicaStrategy(
        add_ipu_cross_replica_reductions=False)


def training_graph(model, opts, iterations_per_step=1):
    train_graph = tf.Graph()
    sess_config = tf.ConfigProto()
    sess_target = None
    strategy = None

    if opts['distributed_cluster']:
        strategy, sess_target, sess_config = configure_distribution(opts, sess_config)
    elif opts['use_popdist']:
        strategy = create_popdist_strategy()

    with train_graph.as_default(), ExitStack() as stack:
        if strategy:
            stack.enter_context(strategy.scope())

        placeholders = dict()
        placeholders['learning_rate'] = tf.placeholder(tf.float32, shape=[])
        learning_rate = placeholders['learning_rate']

        # all data-consuming functions operate on a 'data_dict'
        training_dataset = dataset.data(opts, is_training=True).map(lambda x: {'data_dict': x})

        # datasets must be defined outside the ipu device scope
        train_iterator = ipu_infeed_queue.IPUInfeedQueue(training_dataset,
                                                         feed_name='training_feed',
                                                         replication_factor=opts['replicas'])
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed",
                                                          replication_factor=opts['replicas'])

        with ipu_scope('/device:IPU:0'):
            train = training_step_with_infeeds_and_outfeeds(train_iterator, outfeed_queue, model,
                                                            opts, learning_rate, iterations_per_step)

        outfeed = outfeed_queue.dequeue()
        if strategy:
            # Take the mean of all the outputs across the distributed workers
            outfeed = [strategy.reduce(tf.distribute.ReduceOp.MEAN, v) for v in outfeed]

        logging.print_trainable_variables(opts)

        train_saver = tf.train.Saver(max_to_keep=999999)
        with tf.device('cpu'):
            profile_report = gen_ipu_ops.ipu_event_trace()
        ipu.utils.move_variable_initialization_to_cpu(graph=None)
        train_init = tf.global_variables_initializer()

    globalAMP = None
    if opts["available_memory_proportion"] and len(opts["available_memory_proportion"]) == 1:
        globalAMP = opts["available_memory_proportion"][0]

    ipu_options = get_config(ipu_id=opts["select_ipu"],
                             prng=not opts["no_stochastic_rounding"],
                             shards=opts["shards"],
                             number_of_replicas=opts['replicas'],
                             max_cross_replica_buffer_size=opts["max_cross_replica_buffer_size"],
                             fp_exceptions=opts["fp_exceptions"],
                             half_partials=opts["enable_half_partials"],
                             conv_dithering=opts["enable_conv_dithering"],
                             xla_recompute=opts["xla_recompute"],
                             seed=opts["seed"],
                             profile=opts['profile'],
                             availableMemoryProportion=globalAMP,
                             stable_norm=opts["stable_norm"],
                             internalExchangeOptimisationTarget=opts[
                                 "internal_exchange_optimisation_target"
                             ],
                             limitVertexState=opts.get("limitVertexState", True))

    if opts['use_popdist']:
        ipu_options = popdist.tensorflow.set_ipu_config(ipu_options, opts['shards'], configure_device=False)

    if opts['compile_only']:
        ipu_options = ipu.utils.set_ipu_connection_type(
            ipu_options, ipu.utils.DeviceConnectionType.NEVER, ipu_version=2, enable_remote_buffers=True)

    ipu.utils.configure_ipu_system(ipu_options)
    train_sess = tf.Session(graph=train_graph, config=sess_config, target=sess_target)

    return GraphOps(train_graph, train_sess, train_init, [train], placeholders, train_iterator, outfeed, train_saver,
                    profile_report)


def training_step(train, e, learning_rate):
    # Run Training
    start = time.time()
    _ = train.session.run(train.ops, feed_dict={train.placeholders['learning_rate']: learning_rate})
    batch_time = (time.time() - start)
    if not os.environ.get('TF_POPLAR_FLAGS') or '--use_synthetic_data' not in os.environ.get('TF_POPLAR_FLAGS'):
        loss, cross_ent, accuracy, lr_out = train.session.run(train.outfeed)
        loss = np.mean(loss)
        accuracy = 100.0 * np.mean(accuracy)
        lr = lr_out.flatten()[-1]
    else:
        loss, accuracy, lr = 0, 0, 0
    return loss, accuracy, batch_time, lr


def train_process(model, LR_Class, opts):
    # --------------- OPTIONS --------------------
    epochs = opts["epochs"]
    iterations_per_epoch = DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES'] // opts["total_batch_size"]

    if not opts['iterations']:
        iterations = DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES']*epochs // opts["total_batch_size"]
        log_freq = iterations_per_epoch // opts['logs_per_epoch']
    else:
        iterations = opts['iterations']
        log_freq = opts['log_freq']

    if log_freq < opts['batches_per_step']:
        iterations_per_step = log_freq
    else:
        iterations_per_step = log_freq // int(round(log_freq / opts['batches_per_step']))

    iterations_per_valid = iterations_per_epoch
    iterations_per_ckpt = iterations_per_epoch // opts['ckpts_per_epoch'] if opts['ckpts_per_epoch'] else np.inf

    LR = LR_Class(opts, iterations)

    batch_accs = deque(maxlen=iterations_per_epoch // iterations_per_step)
    batch_losses = deque(maxlen=iterations_per_epoch // iterations_per_step)
    batch_times = deque(maxlen=iterations_per_epoch // iterations_per_step)
    start_all = None
    validation_points = []
    ckpts = []

    # -------------- BUILD TRAINING GRAPH ----------------
    train_iterations = (iterations_per_step if opts['pipeline'] else
                        iterations_per_step * opts["gradient_accumulation_count"])
    train = training_graph(model, opts, train_iterations)
    train.session.run(train.init)
    train.session.run(train.iterator.initializer)

    # -------------- SAVE AND RESTORE --------------
    if opts.get('init_path'):
        train.saver.restore(train.session, opts['init_path'])

    if opts.get('restoring'):
        if opts['distributed_worker_index'] == 0:
            filename_pattern = re.compile('(.*ckpt-(\d+)).index')
            patterns = map(lambda x: filename_pattern.match(x), os.listdir(opts['logs_path']))  # apply regex
            filtered = filter(lambda x: x is not None, patterns)  # remove patterns that don't match regex
            tuples = list(map(lambda x: (int(x.group(2)), os.path.join(opts['logs_path'], x.group(1))),
                              filtered))  # create a tuple for easier sorting
            filenames = sorted(tuples, key=lambda x: x[0])  # sort
            latest_checkpoint = filenames[-1]
            logging.print_to_file_and_screen(
                "Restoring training from latest checkpoint: {}".format(latest_checkpoint[1]), opts)
            i = int(latest_checkpoint[0])
            train.saver.restore(train.session, latest_checkpoint[1])

            # restore list of saved checkpoints
            for j, f in filenames:
                epoch = float(opts["total_batch_size"] * j) / DATASET_CONSTANTS[opts['dataset']]['NUM_IMAGES']
                if j != 0:
                    ckpts.append((j, epoch, False, f))

                _j = j - iterations_per_step
                valid_this_step = (opts['validation'] and
                                   ((_j // iterations_per_valid) < ((_j + iterations_per_step) // iterations_per_valid) or
                                    (_j == 0) or
                                    ((_j + (2 * iterations_per_step)) >= iterations)))
                if valid_this_step:
                    validation_points.append((j, epoch, j == 0, f))
        else:
            i = 0

        if opts['use_popdist']:
            # only instance 0 accesses the disk to restore the checkpoints, so the value of the most recent iteration
            # is only known to this instance. We use horovod to synchronise the iteration value across the other
            # instances. The same happens to the variable values restored from the latest checkpoint.
            with tf.Graph().as_default(), tf.Session():
                local_tensor = tf.constant(i)
                root_tensor = hvd.broadcast(local_tensor, root_rank=0)
                i = root_tensor.eval()
            broadcast_ops = []
            with train.graph.as_default():
                for var in tf.global_variables():
                    broadcast_ops.append(var.assign(hvd.broadcast(var, root_rank=0)))
                train.session.run(broadcast_ops)
    else:
        i = 0

    if opts['ckpts_per_epoch'] and opts['distributed_worker_index'] == 0:
        filepath = train.saver.save(train.session, opts["checkpoint_path"], global_step=0)
        print("Saved checkpoint to {}".format(filepath))

    # ------------- TRAINING LOOP ----------------

    print_format = (
        "step: {step:6d}, iteration: {iteration:6d}, epoch: {epoch:6.2f}, lr: {lr:6.4g}, loss: {loss_avg:6.3f}, top-1 accuracy: {train_acc_avg:6.3f}%"
        ", img/sec: {img_per_sec:6.2f}, time: {it_time:8.6f}, total_time: {total_time:8.1f}")

    step = 0
    start_all = time.time()

    while i < iterations:

        if not opts['pipeline']:
            step += opts["gradient_accumulation_count"]
        else:
            step += 1
        log_this_step = ((i // log_freq) < ((i + iterations_per_step) // log_freq) or
                         (i == 0) or
                         ((i + (2 * iterations_per_step)) >= iterations))
        ckpt_this_step = (opts["ckpts_per_epoch"] and
                          ((i // iterations_per_ckpt) < ((i + iterations_per_step) // iterations_per_ckpt) or
                          (i == 0) or
                          ((i + (2 * iterations_per_step)) >= iterations)))
        valid_this_step = (opts['validation'] and
                           ((i // iterations_per_valid) < ((i + iterations_per_step) // iterations_per_valid) or
                           (i == 0) or
                           ((i + (2 * iterations_per_step)) >= iterations)))

        epoch = float(opts["total_batch_size"] * (i + iterations_per_step)) / DATASET_CONSTANTS[opts['dataset']][
            'NUM_IMAGES']

        # Run Training
        try:
            batch_loss, batch_acc, batch_time, current_lr = training_step(train, i + 1, LR.feed_dict_lr(i))
            if opts['pipeline']:
                current_lr *= opts["lr_scale"]
        except tf.errors.OpError as e:
            raise tf.errors.ResourceExhaustedError(e.node_def, e.op, e.message)

        batch_time /= iterations_per_step

        # Calculate Stats
        batch_accs.append([batch_acc])
        batch_losses.append([batch_loss])

        if i != 0:
            batch_times.append([batch_time])

        # Print loss
        if log_this_step:
            train_acc = np.mean(batch_accs)
            train_loss = np.mean(batch_losses)

            if len(batch_times) != 0:
                avg_batch_time = np.mean(batch_times)
            else:
                avg_batch_time = batch_time

            # flush times every time it is reported
            batch_times.clear()

            total_time = time.time() - start_all

            stats = OrderedDict([
                ('step', step),
                ('iteration', i + iterations_per_step),
                ('epoch', epoch),
                ('lr', current_lr),
                ('loss_batch', batch_loss),
                ('loss_avg', train_loss),
                ('train_acc_batch', batch_acc),
                ('train_acc_avg', train_acc),
                ('it_time', avg_batch_time),
                ('img_per_sec', opts['total_batch_size'] / avg_batch_time),
                ('total_time', total_time),
            ])

            logging.print_to_file_and_screen(print_format.format(**stats), opts)
            logging.write_to_csv(stats, i == 0, True, opts)

        # only instance 0 writes checkpoints to disk
        if ckpt_this_step and (opts['distributed_worker_index'] == 0 or opts['ckpt_all_instances']):
            filepath = train.saver.save(train.session, opts["checkpoint_path"], global_step=i + iterations_per_step)
            print("Saved checkpoint to {}".format(filepath))
            ckpts.append((i + iterations_per_step, epoch, i == 0, filepath))

        # Eval
        # only instance 0 loads checkpoints from disk during validation
        if valid_this_step and opts['validation'] and opts['distributed_worker_index'] == 0:
            validation_points.append((i + iterations_per_step, epoch, i == 0, filepath))

        # ------------ COLLECT PROFILE -----------
        # only instance 0 writes the tf profile to disk
        if opts["profile"] and i == 0 and opts['distributed_worker_index'] == 0:
            from gcprofile import save_tf_report
            save_tf_report(train.session.run(train.profile))
        i += iterations_per_step

    # only instance 0 loads checkpoints form disk during weight averaging
    if (opts['weight_avg_N'] or opts['weight_avg_exp']) and opts['distributed_worker_index'] == 0:
        _ckpts = ckpts[:-1]  # discard final checkpoint as the final 2 are practically identical
        final_iteration, final_epoch = _ckpts[-1][:2]
        if opts['weight_avg_N']:
            for N in opts['weight_avg_N']:
                V = average_ckpts(
                    [c[3] for c in _ckpts if round(c[1], 1) >= round(final_epoch, 1) - N],
                    mode='mean')
                filename = os.path.join(opts["checkpoint_path"], "weight_avg_N_{}".format(N))
                save_ckpt(V, ckpts[-1][3], filename)
                validation_points.append((final_iteration, final_epoch, False, filename))

        if opts['weight_avg_exp']:
            for d in opts['weight_avg_exp']:
                V = average_ckpts(list(zip(*ckpts))[3], mode='exponential', decay=d)
                filename = os.path.join(opts["checkpoint_path"], "weight_avg_exp_{}".format(d))
                save_ckpt(V, ckpts[-1][3], filename)
                validation_points.append((final_iteration, final_epoch, False, filename))

    # ------------ VALIDATION ------------
    if len(validation_points) > 0 and opts['validation']:
        # -------------- BUILD VALIDATION GRAPH ----------------
        valid = validation.initialise_validation(model, opts)

        # ------------ RUN VALIDATION ------------
        for iteration, epoch, first_run, filepath in validation_points:
            stats = validation.validation_run(valid, filepath, iteration, epoch, first_run, opts)

    # --------------- CLEANUP ----------------
    train.session.close()


def add_main_arguments(parser, required=True):
    group = parser.add_argument_group('Main')
    group.add_argument('--model', type=str.lower, default='resnet', help="Choose model")
    group.add_argument('--lr-schedule', default='stepped',
                       help="Learning rate schedule function. Default: stepped")
    group.add_argument('--restore-path', type=str,
                       help='path to training log folder of run to restore')
    group.add_argument('--help', action='store_true', help='Show help information')

    return parser


def set_main_defaults(opts):
    opts['training'] = True
    if opts.get('restore_path'):
        opts['restoring'] = True
    opts['summary_str'] = "\n"


def add_training_arguments(parser):
    tr_group = parser.add_argument_group('Training')
    tr_group.add_argument('--batch-size', type=int,
                          help="Set batch-size for training graph")
    tr_group.add_argument('--gradient-accumulation-count', type=int, default=1,
                          help="""Number of gradients to accumulate before doing a weight update.
                                When using pipelining this is the number of times each pipeline stage
                                will be executed.""")
    tr_group.add_argument('--base-learning-rate', type=float,
                          help="Base learning rate exponent (2**N). blr = lr /  bs")
    tr_group.add_argument('--abs-learning-rate', type=float,
                          help="Absolute learning rate, if value not specified the base learning rate is used.")
    tr_group.add_argument('--epochs', type=float,
                          help="Number of training epochs")
    tr_group.add_argument('--iterations', type=int, default=None,
                          help="Force a fixed number of training iterations to be run rather than epochs.")
    tr_group.add_argument('--weight-decay', type=float,
                          help="Value for weight decay bias, setting to 0 removes weight decay.")
    tr_group.add_argument('--loss-scaling', type=float, default=128,
                          help="Loss scaling factor")
    tr_group.add_argument('--label-smoothing', type=float, default=0,
                          help="Label smoothing factor (Default=0 => no smoothing)")

    tr_group.add_argument('--ckpts-per-epoch', type=int, default=1,
                          help="Checkpoints per epoch")
    tr_group.add_argument('--ckpt-all-instances', type=bool, default=False,
                          help="""Allow all instances to create a checkpoint.
                                By default only instance 0 does checkpointing""")
    tr_group.add_argument('--no-validation', action="store_false", dest='validation',
                          help="Do not do any validation runs.")
    tr_group.set_defaults(validation=True)
    tr_group.add_argument('--shards', type=int, default=1,
                          help="Number of IPU shards for training graph")
    tr_group.add_argument('--replicas', type=int, default=1,
                          help="Replicate graph N times to increase batch to batch-size*N")
    tr_group.add_argument('--max-cross-replica-buffer-size', type=int, default=10 * 1024 * 1024,
                          help="""The maximum number of bytes that can be waiting before a cross
                                replica sum op is scheduled. [Default=10*1024*1024]""")
    tr_group.add_argument('--pipeline', action="store_true",
                          help="""Enables pipelining. Must also set --shards > 1
                                and --gradient-accumulation-count > --shards.""")
    tr_group.add_argument('--pipeline-splits', nargs='+', type=str, default=None,
                          help="Strings for splitting pipelines. E.g. b2/0/relu b3/0/relu")
    tr_group.add_argument('--pipeline-schedule', type=str, default="Interleaved",
                          choices=pipeline_schedule_options,
                          help="Pipelining schedule. Choose between 'Interleaved', 'Grouped' and 'Sequential'.")
    tr_group.add_argument('--optimiser', type=str, default="SGD", choices=['SGD', 'RMSprop', 'momentum', 'LARS'],
                          help="Optimiser")
    tr_group.add_argument('--momentum', type=float, default=0.9,
                          help="Momentum coefficient")
    tr_group.add_argument('--rmsprop-decay', type=float,
                          help="RMSprop decay coefficient")
    tr_group.add_argument('--rmsprop-base-decay-exp', type=float,
                          help="Linearly scale RMSprop decay coefficient as 1-(total_batch_size*2**rmsprop_base_decay_exp) ")
    tr_group.add_argument('--rmsprop-epsilon', type=float, default=0.001,
                          help="RMSprop epsilon coefficient")
    tr_group.add_argument('--variable-filter', nargs='+', type=str, default=[],
                          help="Filter which variables to include in training")
    tr_group.add_argument('--init-path', type=str,
                          help="Path to checkpoint to initialise from")

    tr_group.add_argument('--distributed', action="store_true",
                          help="Use distributed multi-worker training")

    tr_group.add_argument('--stable-norm', action="store_true",
                          help="Use stable implementation of normalization functions")

    tr_group.add_argument('--weight-avg-N', nargs='+', type=int, default=None,
                          help="Number of checkpoints to average over")
    tr_group.add_argument('--weight-avg-exp', nargs='+', type=float, default=None,
                          help="Decay factor for averaging weights")

    tr_group.add_argument('--lars-epsilon', type=float, default=0.0,
                          help='Optional epsilon coefficient for LARS optimiser')
    tr_group.add_argument('--lars-skip-list', type=str, nargs='+', default=['beta', 'gamma', 'bias'],
                          help='Variables to not update via the LARS optimiser')
    tr_group.add_argument('--lars-weight-decay', type=float,
                          help='Weight decay within LARS optimiser, defaults to value of --weight-decay')
    tr_group.add_argument('--lars-eeta', type=float, default=0.001, help='Eeta value for LARS optimiser')

    return parser


def set_training_defaults(opts):
    opts['total_batch_size'] = opts['batch_size'] * opts['gradient_accumulation_count'] * opts['replicas'] * opts[
        'distributed_worker_count']
    opts['summary_str'] += "Training\n"
    opts['summary_str'] += " Batch Size: {total_batch_size}\n"
    if opts['pipeline']:
        opts['summary_str'] += "  Pipelined over {shards} stages\n"
    elif opts['shards'] > 1:
        opts['summary_str'] += " Training Shards: {shards}\n"
    if opts['gradient_accumulation_count'] > 1:
        opts['summary_str'] += "  Gradients accumulated over {gradient_accumulation_count} fwds/bwds passes \n"
    if opts['replicas'] > 1:
        opts['summary_str'] += "  Training on {replicas} replicas \n"
    opts['summary_str'] += (" Base Learning Rate: 2**{base_learning_rate}\n"
                            " Weight Decay: {weight_decay}\n"
                            " Loss Scaling: {loss_scaling}\n")
    if opts["iterations"]:
        opts['summary_str'] += " Iterations: {iterations}\n"
    else:
        opts['summary_str'] += " Epochs: {epochs}\n"

    if opts['abs_learning_rate'] is not None:
        opts['base_learning_rate'] = math.log(opts['abs_learning_rate'] / opts['total_batch_size'], 2.0)

    if opts['rmsprop_base_decay_exp'] is not None:
        if opts['rmsprop_decay'] is None:
            opts['rmsprop_decay'] = 1 - ((2 ** opts['rmsprop_base_decay_exp']) * opts['total_batch_size'])
        else:
            print("'rmsprop_base_decay_exp' ignored as `rmsprop_decay' is already specified.")

    # lr_scale is used to scale down the LR to account for loss scaling
    # lr_scale==loss_scaling iff the update is not divided by a linear function
    # of the gradients otherwise lr_scale = 1
    opts['lr_scale'] = opts['loss_scaling']
    opts['grad_scale'] = 1.0
    if opts['optimiser'] == 'SGD':
        opts['summary_str'] += "SGD\n"
    elif opts['optimiser'] == 'momentum':
        opts['summary_str'] += ("SGD with Momentum\n"
                                " Momentum: {momentum}\n")
    elif opts['optimiser'] == 'RMSprop':
        opts['summary_str'] += ("RMSprop\n"
                                " Momentum: {momentum}\n"
                                " Decay: {rmsprop_decay}\n"
                                " Epsilon: {rmsprop_epsilon}\n")
        opts['lr_scale'] = 1.0
        opts['grad_scale'] = opts['loss_scaling']
    elif opts['optimiser'] == 'LARS':
        opts['summary_str'] += ("LARS\n"
                                " Momentum: {momentum}\n"
                                " Epsilon: {lars_epsilon}\n"
                                " Skip List: {lars_skip_list}\n")
        opts['lr_scale'] = 1.0
        opts['grad_scale'] = opts['loss_scaling']
        if not opts['lars_weight_decay']:
            opts['lars_weight_decay'] = opts['weight_decay']


def add_ipu_arguments(parser):
    group = parser.add_argument_group('IPU')
    group.add_argument('--precision', type=str, default="16.16", choices=["16.16", "16.32", "32.32"],
                       help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    group.add_argument('--enable-half-partials', action="store_true", default=False,
                       help="Use half (float16) partials for both convolutions and matmuls.")
    group.add_argument('--no-stochastic-rounding', action="store_true",
                       help="Disable Stochastic Rounding")
    group.add_argument('--batches-per-step', type=int, default=1000,
                       help="Maximum number of batches to perform on the device before returning to the host.")
    group.add_argument('--select-ipu', type=str, default="AUTO",
                       help="Select IPU either: AUTO or IPU ID")
    group.add_argument('--sharding-exclude-filter', nargs='+', type=str, default=['0/p'],
                       help="Excluded strings for splitting edges (only a substring must match)")
    group.add_argument('--sharding-include-filter', nargs='+', type=str, default=[''],
                       help="Included strings for splitting edges (only a substring must match)")
    group.add_argument('--fp-exceptions', action="store_true",
                       help="Turn on floating point exceptions")
    group.add_argument('--xla-recompute', action="store_true",
                       help="Allow recomputation of activations to reduce memory usage")
    group.add_argument('--seed', default=None, help="Seed for randomizing training")
    group.add_argument('--gc-profile', type=str, dest='profile', nargs='?', action='store', default=None,
                       const="IPU_PROFILE", choices=["DEVICE_PROFILE", "IPU_PROFILE", "TILE_PROFILE", "NO_PROFILE"],
                       help="Generate profile reprts")
    group.add_argument('--dataset-benchmark', action="store_true", default=False, help="Benchmark dataset")
    group.add_argument('--available-memory-proportion', default=None, nargs='+',
                       help="Proportion of memory which is available for convolutions. Use a value of less than 0.6 "
                            "to reduce memory usage.")
    group.add_argument('--disable-variable-offloading', action="store_true",
                       help="Disable offloading of variables to remote memory. This may increase live memory usage")
    group.add_argument('--enable-conv-dithering', action="store_true", default=False,
                       help="Enable dithering of the convolution start tile to improve tile memory balance")
    group.add_argument('--internal-exchange-optimisation-target', type=str, default=None,
                       choices=["balanced", "cycles", "memory"],
                       help="""The optimisation approach for internal exchanges.""")
    group.add_argument('--compile-only', action="store_true", default=False,
                       help="Configure TensorFlow to only compile the graph. This will not acquire any IPUs and thus "
                       "facilitates profiling without using hardware resources.")
    return parser


def set_distribution_defaults(opts):
    if opts['distributed'] and opts['use_popdist']:
        raise ValueError("Cannot use popdist with --distributed")

    if opts['distributed']:
        # Read the cluster config from the `TF_CONFIG` environment variable
        cluster = tf.distribute.cluster_resolver.TFConfigClusterResolver()

        # Allow `mpirun` to override the task index
        cluster.task_id = os.getenv("OMPI_COMM_WORLD_RANK")
        cluster.task_type = "worker"

        opts['distributed_worker_count'] = cluster.cluster_spec().num_tasks("worker")
        opts['distributed_worker_index'] = cluster.task_id
        opts['distributed_cluster'] = cluster.cluster_spec().as_dict()

        opts['summary_str'] += 'Distribution\n'
        opts['summary_str'] += ' Worker count: {distributed_worker_count}\n'
        opts['summary_str'] += ' Worker index: {distributed_worker_index}\n'
        opts['summary_str'] += ' Cluster: {distributed_cluster}\n'
    elif opts['use_popdist']:
        opts['distributed_worker_count'] = popdist.getNumInstances()
        opts['distributed_worker_index'] = popdist.getInstanceIndex()
        opts['distributed_cluster'] = None

        opts['summary_str'] += 'Popdist\n'
        opts['summary_str'] += ' Process count: {distributed_worker_count}\n'
        opts['summary_str'] += ' Process index: {distributed_worker_index}\n'
    else:
        opts['distributed_worker_count'] = 1
        opts['distributed_worker_index'] = 0
        opts['distributed_cluster'] = None


def set_ipu_defaults(opts):
    opts['summary_str'] += "Using Infeeds\n Max Batches Per Step: {batches_per_step}\n"
    opts['summary_str'] += 'Device\n'
    opts['summary_str'] += ' Precision: {}{}\n'.format(opts['precision'],
                                                       '_noSR' if opts['no_stochastic_rounding'] else '')
    opts['summary_str'] += ' IPU\n'
    opts['poplar_version'] = os.popen('popc --version').read()
    opts['summary_str'] += ' {poplar_version}'
    opts['select_ipu'] = -1 if opts['select_ipu'].lower() == 'auto' else int(opts['select_ipu'])

    opts['hostname'] = gethostname()
    opts['datetime'] = str(datetime.datetime.now())

    if opts['seed']:
        # Seed the various random sources
        seed = int(opts['seed'])
        opts['seed_specified'] = True
        set_seeds(seed)
        opts['seed'] = seed
    else:
        opts['seed_specified'] = False

    opts['summary_str'] += (' {hostname}\n'
                            ' {datetime}\n')


def set_seeds(seed):
    random.seed(seed)
    # Set other seeds to different values for extra safety.
    # The new seeds are defined indirectly by the main seed,
    # since they are generated by the seeded random function.
    tf.set_random_seed(random.randint(0, 2**32 - 1))
    np.random.seed(random.randint(0, 2**32 - 1))
    reset_ipu_seed(random.randint(-2**16, 2**16 - 1))


def create_parser(model, lr_schedule, parser):
    parser = model.add_arguments(parser)
    parser = dataset.add_arguments(parser)
    parser = add_training_arguments(parser)
    parser = lr_schedule.add_arguments(parser)
    parser = add_ipu_arguments(parser)
    parser = logging.add_arguments(parser)
    return parser


def set_defaults(model, LR, opts):
    set_main_defaults(opts)
    dataset.set_defaults(opts)
    model.set_defaults(opts)
    set_distribution_defaults(opts)
    set_training_defaults(opts)
    LR.set_defaults(opts)
    validation.set_validation_defaults(opts)
    set_ipu_defaults(opts)
    logging.set_defaults(opts)


def set_profiling_opts(opts):
    if opts["profile"]:
        opts["summary_str"] += 'A profiling session is run, the report will be retrieved for the first iteration'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN Training in TensorFlow', add_help=False)
    parser = add_main_arguments(parser)
    parser = configurations.add_arguments(parser, config_file)
    args, unknown = parser.parse_known_args()
    args = configurations.parse_config(args, parser, config_file, known_args_only=True)
    print(args)
    args = vars(args)

    try:
        model = importlib.import_module("Models." + args['model'])
    except ImportError:
        raise ValueError('Models/{}.py not found'.format(args['model']))

    try:
        lr_schedule = importlib.import_module("LR_Schedules." + args['lr_schedule'])
    except ImportError:
        raise ValueError("LR_Schedules/{}.py not found".format(args['lr_schedule']))

    # Large number of deprecation warnings that cannot be resolved yet.
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = create_parser(model, lr_schedule, parser)
    opts = parser.parse_args()
    opts = configurations.parse_config(opts, parser, config_file)
    opts = vars(opts)
    print(opts)

    if opts['help']:
        parser.print_help()
    else:
        if opts['pipeline'] and opts['shards'] == 1:
            raise ValueError("--pipeline can only be used if --shards > 1")
        amps = opts['available_memory_proportion']
        if amps and len(amps) > 1:
            if not opts['pipeline']:
                raise ValueError('--available-memory-proportion should only have one value unless using pipelining')
            if len(amps) != int(opts['shards']) * 2:
                raise ValueError(
                    '--available-memory-proportion should have either one value or 2*shards values specified')

        if popdist.isPopdistEnvSet():
            opts['use_popdist'] = True
            opts['replicas'] = popdist.getNumLocalReplicas()
            opts['total_replicas'] = popdist.getNumTotalReplicas()
            if not opts['compile_only']:
                opts['select_ipu'] = str(popdist.getDeviceId())
        else:
            opts['use_popdist'] = False
            opts['total_replicas'] = opts['replicas']

        opts["command"] = ' '.join(sys.argv)
        set_defaults(model, lr_schedule, opts)

        if opts['dataset'] == 'imagenet':
            if opts['image_size'] is None:
                opts['image_size'] = 224
            if opts['image_size'] != 224:
                opts['name'] += '_{}x{}'.format(opts['image_size'], opts['image_size'])
            opts['summary_str'] += "Image Size: {}x{}\n".format(opts['image_size'], opts['image_size'])
        elif 'cifar' in opts['dataset']:
            if opts['image_size'] is not None and opts['image_size'] != 32:
                raise ValueError('--image-size not supported for CIFAR sized datasets')
            opts['image_size'] = 32
        set_profiling_opts(opts)
        logging.print_to_file_and_screen("Command line: " + opts["command"], opts)
        logging.print_to_file_and_screen(opts["summary_str"].format(**opts), opts)
        opts["summary_str"] = ""
        logging.print_to_file_and_screen(opts, opts)
        if opts['dataset_benchmark']:
            dataset = dataset.data(opts, is_training=True)
            benchmark_op = ipu.dataset_benchmark.dataset_benchmark(dataset, opts['epochs'], 512)
            with tf.Session() as sess:
                json_string = sess.run(benchmark_op)
            json_string = json.loads(json_string[0].decode('utf-8'))
            for i in range(len(json_string['epochs'])):
                json_string['epochs'][i]['elements_per_second'] *= opts['batch_size']
                json_string['epochs'][i]['elements_processed'] *= opts['batch_size']
            mean_throughput = np.mean([epoch['elements_per_second'] for epoch in json_string['epochs']])
            print(f'mean throughput: {mean_throughput} imgs/sec')
        else:
            train_process(model, lr_schedule.LearningRate, opts)
