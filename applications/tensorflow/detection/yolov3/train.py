#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.


import argparse
import json
import os
import re
import time
from collections import deque
from contextlib import ExitStack
from datetime import datetime
from queue import Queue
from threading import Thread

import numpy as np

import core.utils as utils
import cv2
import ipu_utils
import log
import popdist
import popdist.tensorflow
import tensorflow as tf
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from ipu_optimizer import AdamWeightDecayOptimizer, IPUOptimizer, MomentumOptimizer
from ipu_utils import stages_constructor
from log import logger
from tensorflow.python import ipu
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu import internal_ops, ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops, \
    pipelining_ops, scopes
from tensorflow.python.ipu.horovod import ipu_multi_replica_strategy

threads = 2
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
cv2.setNumThreads(threads)

tf.disable_v2_behavior()


class YoloTrain(object):
    """Train yolov3 model
    """

    def __init__(self, opts):
        """Create a training class
        The constructor init all needed parameters
        """
        self.opts = opts
        self.learn_rate_init = opts["train"]["learn_rate_init"]
        self.learn_rate_end = opts["train"]["learn_rate_end"]
        self.epochs = opts["train"]["epochs"]
        self.warmup_epochs = opts["train"]["warmup_epochs"]
        self.initial_weight = opts["train"]["initial_weight"]
        self.moving_avg_decay = opts["yolo"]["moving_avg_decay"]
        self.trainset = Dataset("train", self.opts)
        self.steps_per_epoch = len(
            self.trainset)/opts["train"]["total_replicas"]
        self.precision = tf.float16 if opts["yolo"]["precision"] == "fp16" else tf.float32
        self.model = YOLOV3(opts["train"]["bn_trainable"], opts)
        self.batch_size = opts["train"]["batch_size"]
        self.data_threads_number = opts["train"]["data_threads_number"]
        self.loss_scaling = opts["train"]["loss_scaling"]
        self.repeat_count = opts["train"]["repeat_count"]
        self.for_speed_test = opts["train"]["for_speed_test"]

    def optimize_func(self, giou_loss, conf_loss, prob_loss, lr):
        self.loss = giou_loss + conf_loss + prob_loss
        self.loss = self.loss*self.loss_scaling
        if self.opts["train"]["freeze_pretrain"]:
            # with freeze_pretrain option, we only train new added parameters
            restored_variables = get_restore_variables(self.opts["train"]["load_type"])
            var_list = [var for var in tf.trainable_variables() if var not in restored_variables]
            logger.info("variables will be trained:")
            for var in var_list:
                logger.info(var.name)
        else:
            var_list = tf.trainable_variables()

        if self.opts["train"]["optimizer"] == "adamw":
            # adamw uses a update scaled by it's second momentum
            # so gradients getting larger won't affect it's update
            optimizer = AdamWeightDecayOptimizer(lr,
                                                 use_moving_avg=opts["yolo"]["use_moving_avg"],
                                                 moving_avg_decay=opts["yolo"]["moving_avg_decay"],
                                                 darknet_gn=opts["yolo"]["darknet_gn"],
                                                 upsample_gn=opts["yolo"]["upsample_gn"]
                                                 )
        elif self.opts["train"]["optimizer"] == "momentum":
            optimizer = MomentumOptimizer(lr,
                                          use_moving_avg=opts["yolo"]["use_moving_avg"],
                                          moving_avg_decay=opts["yolo"]["moving_avg_decay"],
                                          loss_scaling=self.loss_scaling,
                                          momentum=0.9,
                                          backbone_gn=opts["yolo"]["backbone_gn"],
                                          upsample_gn=opts["yolo"]["upsample_gn"]
                                          )
        else:
            raise Exception("unexpected optimizer config")

        return pipelining_ops.OptimizerFunctionOutput(IPUOptimizer(optimizer,
                                                                   sharded=False,
                                                                   replicas=opts["train"]["total_replicas"],
                                                                   gradient_accumulation_count=opts[
                                                                       "train"]["pipeline_depth"],
                                                                   pipelining=True,
                                                                   var_list=var_list), self.loss,
                                                      )

    def get_lr(self, global_step):
        with tf.name_scope("learn_rate"):
            # warmup_steps plus 0.9 to avoid nan
            warmup_steps = tf.constant(self.warmup_epochs * self.steps_per_epoch+0.9,
                                       dtype=tf.float32, name="warmup_steps")
            train_steps = tf.constant((self.epochs) * self.steps_per_epoch,
                                      dtype=tf.float32, name="train_steps")
            global_step = tf.cast(global_step, tf.float32)

            true_val = self.learn_rate_init * \
                tf.cast(global_step, tf.float32)/warmup_steps
            true_val = tf.cast(true_val, dtype=tf.float16)

            angle = (global_step - warmup_steps) / \
                (train_steps - warmup_steps) * np.pi
            false_val = self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * \
                (1 + tf.cos(angle))
            false_val = tf.cast(false_val, dtype=tf.float16)
            self.learn_rate = tf.cond(
                pred=global_step < warmup_steps,
                true_fn=lambda: true_val,
                false_fn=lambda: false_val)
        return self.learn_rate

    def build_pretrain_pipeline_stages(self, model, opts):
        """Build pipeline stages according to "pipeline_stages" in config file
        """
        # pipeline_depth need to be times of stage_number*2
        # this is constrained by sdk
        assert opts["train"]["pipeline_depth"] % (
            len(opts["train"]["pipeline_stages"])*2) == 0

        # list of list
        # outter list stands for stages
        # inner list stands for conv layers of each stage
        stage_layer_list = []
        conv_layers = []
        # build layer functions for backbone and upsample
        conv_layers.extend(model.build_backbone())
        conv_layers.extend(model.build_upsample())
        # keep number of backbone and upsample layers for checking
        backbone_layers = 0
        upsample_layers = 0
        for stage in opts["train"]["pipeline_stages"]:
            func_list = []
            for layers in stage:
                # layers should be in the format of [backbone]|[upsample]_x[layer_number]
                if "backbone" in layers or "upsample" in layers:
                    layer_number = int(layers.split("_x")[-1])
                    if "backbone" in layers:
                        backbone_layers += layer_number
                    if "upsample" in layers:
                        upsample_layers += layer_number
                    func_list.extend(conv_layers[:layer_number])
                    conv_layers = conv_layers[layer_number:]
            stage_layer_list.append(func_list)
        # last layer of darknet53 is classification layer, so it have 52 conv layers
        assert backbone_layers == 52
        # there is 25 conv layers if we count upsmaple as a conv layer
        assert upsample_layers == 25
        # decoding layer and loss layer is always put on last IPU
        stage_layer_list[-1].append(model.decode_boxes)
        stage_layer_list[-1].append(model.compute_loss)
        # we append learning rate calculation at last stage

        def lr_wrapper(global_step):
            return {"lr": self.get_lr(global_step)}
        stage_layer_list[-1].append(lr_wrapper)

        computational_stages = stages_constructor(
            stage_layer_list,
            ["global_step", ],
            ["giou_loss", "conf_loss", "prob_loss", "lr"])

        return computational_stages

    def model_func(self, model, opts, global_step_holder, infeed_queue, outfeed_queue):
        computational_stages = self.build_pretrain_pipeline_stages(
            model, opts)

        options = [ipu.pipelining_ops.PipelineStageOptions(
            matmul_options={
                "availableMemoryProportion": str(0.2),
                "partialsType": "half"
            },
            convolution_options={
                "partialsType": "half"}
        )] * len(opts["train"]["device_mapping"])

        # we write this wrapper because self.optimizer_func has "self" as it's parameter
        # it will cause an error when cal ipu_compiler.compile
        def optimizer_wrapper(giou_loss, conf_loss, prob_loss, lr):
            return self.optimize_func(giou_loss, conf_loss, prob_loss, lr)

        pipeline_op = pipelining_ops.pipeline(
            computational_stages=computational_stages,
            gradient_accumulation_count=opts["train"]["pipeline_depth"],
            repeat_count=self.repeat_count,
            optimizer_function=optimizer_wrapper,
            inputs=[global_step_holder],
            forward_propagation_stages_poplar_options=options,
            backward_propagation_stages_poplar_options=options,
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            offload_activations=False,
            offload_gradient_accumulation_buffers=False,
            offload_weight_update_variables=False,
            device_mapping=opts["train"]["device_mapping"],
            name="Pipeline")
        return pipeline_op

    def get_loader_and_saver(self):
        variables = get_restore_variables(self.opts["train"]["load_type"])
        if len(variables) > 0:
            loader = tf.train.Saver(variables)
        else:
            loader = None
        saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.opts["train"]["ckpt_num"])
        return loader, saver

    def get_dataset_on_the_fly(self, stop_flag, data_threads):
        precision_np = np.float16 if self.precision == tf.float16 else np.float32

        input_fields = ("input_data",
                        "label_sbbox",
                        "label_mbbox",
                        "label_lbbox",
                        "true_sbbox",
                        "true_mbbox",
                        "true_lbbox")

        input_shapes = []
        for data in self.trainset:
            for part in data:
                input_shapes.append(tf.TensorShape(part.shape))
            break

        def data_generator():
            # buffer dataset to avoid blocked by preprocessing
            # Create the shared queue and launch threads
            # we simply set queue size as the data length for one sess.run
            data_queue = Queue(
                maxsize=self.opts["train"]["pipeline_depth"]*self.opts["train"]["replicas"]*self.opts["train"]["repeat_count"])
            for i in range(self.data_threads_number):
                thread_producer = Thread(target=data_producer, args=(
                    data_queue, self.trainset, precision_np, input_shapes, input_fields, stop_flag, self.for_speed_test, i))
                thread_producer.start()
                data_threads.append(thread_producer)
            while True:
                yield data_queue.get()

        ds = tf.data.TFRecordDataset.from_generator(
            data_generator,
            output_types=(dict(zip(input_fields, [self.precision]*len(input_fields)))),
            output_shapes=dict(zip(input_fields, input_shapes))
        )
        if self.opts["distributed_worker_count"] > 1:
            ds.shard(num_shards=opts["distributed_worker_count"],
                     index=opts["distributed_worker_index"])
        # prefetch data for one sess.run
        ds = ds.prefetch(self.opts["train"]["pipeline_depth"]*self.opts["train"]
                         ["replicas"]*self.opts["train"]["repeat_count"])
        return ds

    def train(self):
        # Configure the IPU options.
        ipu_options = ipu_utils.get_ipu_config(
            ipu_id=self.opts["select_ipu"],
            num_ipus_required=len(
                self.opts["train"]["device_mapping"])*self.opts["train"]["replicas"],
            fp_exceptions=False,
            stochastic_rounding=True,
            xla_recompute=True,
            available_memory_proportion=0.2,
            max_cross_replica_buffer_size=16*1024*1024,
            scheduler_selection="Clustering",
            compile_only=False,
            partials_type="half"
        )
        # config replication strategy
        if self.opts["use_popdist"]:
            strategy = create_popdist_strategy()
            ipu_options = strategy.update_ipu_config(ipu_options)
            ipu_options = popdist.tensorflow.set_ipu_config(
                ipu_options, len(self.opts["train"]["device_mapping"]), configure_device=False)
        ipu_options.configure_ipu_system()

        self.sess = tf.Session(config=tf.ConfigProto())

        stop_flag = []
        data_threads = []
        ds = self.get_dataset_on_the_fly(stop_flag, data_threads)

        global_step_holder = tf.placeholder(dtype=tf.int32, shape=())

        # we write this wrapper because self.model_func has "self" as it's parameter
        # it will cause an error when cal ipu_compiler.compile
        def model_wrapper():
            self.model_func(self.model, self.opts,
                            global_step_holder, self.infeed_queue, self.outfeed_queue)

        with ExitStack() as stack:
            if self.opts["use_popdist"]:
                stack.enter_context(strategy.scope())
            self.infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
                ds,
                replication_factor=self.opts["train"]["replicas"],
                feed_name="infeed")
            self.outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
                replication_factor=self.opts["train"]["replicas"],
                feed_name="outfeed")

            with ipu.scopes.ipu_scope("/device:IPU:0"):
                if self.opts["use_popdist"]:
                    def distributed_per_replica_func():
                        return ipu_compiler.compile(model_wrapper, inputs=[])
                    compiled_model = strategy.experimental_run_v2(
                        distributed_per_replica_func, args=[])
                else:
                    compiled_model = ipu_compiler.compile(model_wrapper, inputs=[])
            # The outfeed dequeue has to happen after the outfeed enqueue(after calling compile)
            dequeue_outfeed = self.outfeed_queue.dequeue()

            if self.opts["use_popdist"]:
                # Take the mean of all the outputs across the distributed workers
                dequeue_outfeed = [strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, v) for v in dequeue_outfeed]

            with tf.name_scope("loader_and_saver"):
                self.loader, self.saver = self.get_loader_and_saver()

        self.sess.run(self.infeed_queue.initializer)
        self.sess.run(tf.global_variables_initializer())

        begin_epoch = 0

        if self.opts["train"]["load_type"] == "resume":
            # resume a half-trained run
            ckpts = []
            if os.path.exists("./checkpoint"):
                ckpts = sorted([path for path in os.listdir(
                    "./checkpoint") if "meta" in path])
            if len(ckpts) == 0:
                logger.info("fail to resume, not find any ckpt")
                return
            ckpt_path = "./checkpoint/"+ckpts[-1].replace(".meta", "")
            logger.info("=> Resume training from: %s ... " % ckpt_path)
            self.loader.restore(self.sess, ckpt_path)
            begin_epoch = int(
                re.search("epoch=([0-9]+)", ckpt_path).groups()[0])
        elif self.opts["train"]["load_type"] in ["yolov3", "darknet53", "phase1"]:
            # if load some pretrained ckpt
            if self.initial_weight and os.path.exists(self.initial_weight+".meta"):
                logger.info("=> Restoring weights from: %s ... " %
                            self.initial_weight)
                self.loader.restore(self.sess, self.initial_weight)
            else:
                raise Exception("can't find ckpt to load")

        elif self.opts["train"]["load_type"] == "empty":
            logger.info("=> no checkpoint to load !!!")
            logger.info("=> Now it starts to train YOLOV3 from scratch ...")
        else:
            raise Exception("'load_type' is not one of expected values: yolov3, darknet53, phase1, resume, empty")

        total_epochs = self.epochs
        total_batch_size = self.opts["train"]["pipeline_depth"] * \
            self.batch_size * \
            self.opts["train"]["replicas"] * \
            self.opts["distributed_worker_count"]
        samples_per_interaction = total_batch_size * self.repeat_count
        samples_per_epoch = len(self.trainset)*self.batch_size
        interactions_per_epoch = samples_per_epoch//samples_per_interaction
        if self.for_speed_test:
            interactions_per_epoch = 30
            total_epochs = 1
        steps_per_epoch = interactions_per_epoch*self.repeat_count
        logger.info("total epochs: {}".format(total_epochs))
        logger.info("steps_per_epoch: {}".format(steps_per_epoch))
        moving_loss = deque(maxlen=30)

        if self.opts["distributed_worker_index"] == 0:
            # we only write logs to tensorboard on main worker
            summary_writer = tf.summary.FileWriter(
                "./tf_log/"+datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), session=self.sess)

        train_begin_time = time.time()
        for epoch in range(begin_epoch, total_epochs):
            logger.info("epoch {}:".format(epoch+1))

            start_time = time.time()
            for interaction_count in range(interactions_per_epoch):
                global_step = epoch*steps_per_epoch+interaction_count*self.repeat_count
                self.sess.run(compiled_model, feed_dict={
                              global_step_holder: global_step})
                result = self.sess.run(dequeue_outfeed)

                if self.opts["distributed_worker_index"] == 0:
                    giou_loss = np.mean(result[0])
                    conf_loss = np.mean(result[1])
                    prob_loss = np.mean(result[2])
                    lr = np.mean(result[3])
                    total_loss = giou_loss + conf_loss + prob_loss
                    moving_loss.append(total_loss)
                    end_time = time.time()
                    duration = end_time - start_time
                    start_time = time.time()
                    total_samples = global_step*total_batch_size
                    logger.info("epoch:{}, global_steps:{}, total_samples:{}, lr:{:.3e}, \
 moving_total_loss:{:.2f}, duration:{:.2f}, samples/s:{:.2f},\
 total_time:{:.2f}".format(
                        epoch+1,
                        global_step,
                        total_samples,
                        lr,
                        np.mean(moving_loss),
                        duration,
                        samples_per_interaction/duration,
                        time.time()-train_begin_time))

                    train_summary = tf.Summary()
                    train_summary.value.add(tag="giou_loss", simple_value=giou_loss)
                    train_summary.value.add(tag="conf_loss", simple_value=conf_loss)
                    train_summary.value.add(tag="prob_loss", simple_value=prob_loss)
                    train_summary.value.add(tag="total_loss", simple_value=total_loss)
                    train_summary.value.add(tag="lr", simple_value=lr)
                    train_summary.value.add(
                        tag="samples_per_sec", simple_value=samples_per_interaction/duration)
                    summary_writer.add_summary(train_summary, total_samples)
                    summary_writer.flush()
            if (not self.for_speed_test) and (epoch % self.opts["train"]["epochs_per_ckpt"] == 0 or epoch == total_epochs-1):
                if self.opts["distributed_worker_index"] == 0:
                    ckpt_loss = np.mean(moving_loss)
                else:
                    # if not call save on all instances, there will be a all-reduce error
                    # but call save on all workers is pointless
                    # so only ckpt saved at worker 0 will have a name with loss value
                    ckpt_loss = 0.0
                ckpt_file = "./checkpoint/yolov3-{}-epoch={}-moving_total_loss={:.4f}.ckpt".format(
                    datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), epoch+1, ckpt_loss)

                logger.info("saving to: "+ckpt_file)
                model_path = self.saver.save(self.sess, ckpt_file,
                                             global_step=global_step)
                if self.opts["distributed_worker_index"] == 0:
                    log.save_model_statistics(model_path, summary_writer, global_step*total_batch_size)
        # tell threads to stop
        stop_flag.append(0)
        for data_thread in data_threads:
            data_thread.join()
        self.sess.close()


def create_popdist_strategy():
    """Creates a distribution strategy with popdist.
    We use the Horovod-based IPUMultiReplicaStrategy. Horovod is used for the initial
    broadcast of the weights and when reductions are requested on the host.
    Imports are placed here so they are only done when required, as Horovod
    might not always be available.
    """

    from tensorflow.python.ipu import horovod as hvd
    from tensorflow.python.ipu.horovod import ipu_multi_replica_strategy

    hvd.init()

    # We add the IPU cross replica reductions explicitly in the IPUOptimizer,
    # so disable them in the IPUMultiReplicaStrategy.
    return ipu_multi_replica_strategy.IPUMultiReplicaStrategy(
        add_ipu_cross_replica_reductions=False)


def get_restore_variables(load_type):
    def contain_words(name, words):
        for word in words:
            if word in name:
                return True
        return False

    def get_darknet53_vars():
        darknet53_vars = []
        for var in tf.global_variables():
            # there's 52 convolutional layers in darknet53 batckbone(0-51)
            # conv52 is the first layer beyond darknet53
            if var.name == "conv52/weight:0":
                break
            darknet53_vars.append(var)
        return darknet53_vars

    def get_yolov3_vars():
        vars = []
        for var in tf.global_variables():
            # if we load from yolov3 model pretrain on coco&gpu
            # we won't want the head of the network, moving average and optimizer parameters
            if contain_words(var.name,
                             ["adam", "ExponentialMovingAverage", "conv_sbbox", "conv_mbbox", "conv_lbbox", "GroupNorm"]):
                continue
            vars.append(var)
        return vars

    def get_phase1_vars():
        vars = []
        for var in tf.global_variables():
            # if it's second training stage
            # we will load all parameters bug those from optimizer, step counter, and moving average
            if contain_words(var.name, ["ExponentialMovingAverage", "adam", "local_step"]):
                continue
            vars.append(var)
        return vars
    if load_type == "yolov3":
        return get_yolov3_vars()
    elif load_type == "darknet53":
        return get_darknet53_vars()
    elif load_type == "phase1":
        return get_phase1_vars()
    elif load_type == "resume":
        return tf.global_variables()
    elif load_type == "empty":
        return []
    else:
        raise Exception("load_type is not one of expected values: yolov3, darknet53, phase1, resume, empty")


def data_producer(out_queue, trainset, precision_np, shapes, fields, stop_flag, for_speed_test, thread_id):
    # A thread that produces data
    while True:
        if len(stop_flag) > 0:
            break
        for data in trainset:
            if len(stop_flag) > 0:
                break
            assert len(data) == 7
            result = []
            for i in range(len(data)):
                arr = data[i]
                arr = arr.astype(precision_np)
                result.append(arr)
            try:
                if for_speed_test:
                    # when trying to reach maximum possible speed
                    # we use one sample to pretend we have load all data into memory
                    while True:
                        if len(stop_flag) > 0:
                            break
                        try:
                            out_queue.put(dict(zip(fields, result)),
                                          block=True, timeout=2)
                        except:
                            pass
                else:
                    out_queue.put(dict(zip(fields, result)),
                                  block=True, timeout=2)
                    if out_queue.qsize() < out_queue.max_size/3:
                        logger.info("data queue storate low, current_len: {}".format(len(out_queue)))
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="yolov3 training in TensorFlow", add_help=False)
    parser.add_argument("--config", type=str, default="config/config.json",
                        help="json config file for yolov3.")
    arguments = parser.parse_args()
    with open(arguments.config) as f:
        opts = json.load(f)

    if popdist.isPopdistEnvSet():
        opts["use_popdist"] = True
        opts["train"]["replicas"] = popdist.getNumLocalReplicas()
        opts["train"]["total_replicas"] = popdist.getNumTotalReplicas()
        opts["select_ipu"] = popdist.getDeviceId(
            len(opts["train"]["device_mapping"]))
        opts["distributed_worker_count"] = int(
            popdist.getNumTotalReplicas() / popdist.getNumLocalReplicas())
        opts["distributed_worker_index"] = int(
            popdist.getReplicaIndexOffset() / popdist.getNumLocalReplicas())
        opts["use_popdist"] = True

    else:
        opts["use_popdist"] = False
        opts["train"]["total_replicas"] = opts["train"]["replicas"]
        opts["select_ipu"] = -1
        opts["distributed_worker_count"] = 1
        opts["distributed_worker_index"] = 0
        opts["use_popdist"] = False

    # for each instance will have difference seed, so data will be shuffled differently
    np.random.seed(opts["distributed_worker_index"])
    logger.info(opts)
    YoloTrain(opts).train()
    if opts["use_popdist"]:
        hvd.shundown()
