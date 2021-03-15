# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tqdm import tqdm
import os
import math
import time
import torch
import poptorch
import logging
import popdist
import horovod.torch as hvd

from poptorch.optim import SGD, RMSprop, AdamW
from lr_schedule import WarmUpLRDecorator, PeriodicLRDecorator
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from train_utils import parse_arguments
from validate import test, validate_checkpoints, create_validation_opts
import weight_avg
import sys
sys.path.append('..')
import models
import utils
import datasets


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model, replicas=1, label_smoothing=0.0):
        super().__init__()
        self.model = model
        self.label_smoothing = label_smoothing
        self.loss = torch.nn.NLLLoss(reduction="mean")
        self.replicas = replicas


    def forward(self, args, loss_inputs=None):
        output = self.model(args)
        # Calculate loss in full precision
        output = output.float()
        if loss_inputs is None:
            return output
        else:
            log_preds = torch.nn.functional.log_softmax(output, dim=1)
            # NOTE: after popART batch + replica reduction dispatch returning loss/final loss hack can be removed
            returning_loss = final_loss = (1.0-self.label_smoothing) * self.loss(log_preds, loss_inputs)
            final_loss = final_loss * self.replicas
            if self.label_smoothing > 0.0:
                # cross entropy between uniform distribution and output distribution
                smoothing_loss = - torch.mean(log_preds) * self.label_smoothing
                final_loss = final_loss + smoothing_loss
                returning_loss = returning_loss + smoothing_loss
            poptorch.identity_loss(final_loss, reduction='mean')
            return output, returning_loss


def train(training_model, training_data, opts, lr_scheduler, epochs, optimizer, validation_function=None):
    old_lr = lr_scheduler.get_last_lr()[0]
    iterations_per_epoch = len(training_data)
    num_instances = opts.popdist_size if opts.use_popdist else 1
    metrics = utils.Metrics(running_mean_length=iterations_per_epoch, distributed=opts.use_popdist)
    # Determine loss scaling change points
    num_loss_scaling_steps = int(math.log2(opts.loss_scaling // opts.initial_loss_scaling)) + 1
    loss_scaling_steps = {i * (opts.epoch // num_loss_scaling_steps) + 1: opts.initial_loss_scaling * (2 ** i) for i in range(num_loss_scaling_steps)}
    new_loss_scaling = old_loss_scaling = opts.initial_loss_scaling
    for epoch in epochs:
        logging.info(f"Epoch {epoch}/{opts.epoch}")
        if opts.disable_metrics:
            bar = training_data
        else:
            bar = tqdm(training_data, total=iterations_per_epoch)
        epoch_start_time = time.time()
        total_sample = 0
        if epoch in loss_scaling_steps.keys():
            new_loss_scaling = loss_scaling_steps[epoch]
        for batch_idx, (input_data, labels) in enumerate(bar):
            preds, losses = training_model(input_data, labels)
            epoch_num = epoch - 1 + float(batch_idx+1) / iterations_per_epoch
            if not opts.disable_metrics:
                with torch.no_grad():
                    mean_loss = torch.mean(losses).item()
                    acc = utils.accuracy(preds, labels)
                metrics.save_value("accuracy", acc)
                metrics.save_value("loss", mean_loss)
                aggregated_loss = metrics.get_running_mean("loss")
                aggregated_acc = metrics.get_running_mean("accuracy")
                bar.set_description(f"Loss:{aggregated_loss:0.4f} | Accuracy:{aggregated_acc:0.2f}%")
            total_sample += input_data.size()[0] * num_instances

            if not opts.disable_metrics and ((batch_idx + 1) % (iterations_per_epoch // opts.logs_per_epoch) == 0):
                elapsed_time = metrics.get_elapsed_time()
                num_batches = metrics.get_count()
                if validation_function is not None and (epoch % opts.validation_frequency == 0) and (not opts.use_popdist or opts.popdist_rank == 0):
                    training_model.detachFromDevice()
                    validation_accuracy = validation_function()
                    model.train()
                    training_model.attachToDevice()
                    logging.info(f"Validation Accuracy: {validation_accuracy:0.2f}%")
                else:
                    validation_accuracy = 0.0
                # save metrics
                result_dict = {"loss_avg": metrics.get_running_mean("loss"),
                               "loss_batch": metrics.get_value("loss"),
                               "epoch": epoch_num,
                               "iteration": batch_idx+1+(epoch-1)*iterations_per_epoch,
                               "train_accuracy_avg": metrics.get_running_mean("accuracy"),
                               "train_accuracy_batch": metrics.get_value("accuracy"),
                               "learning_rate": old_lr,
                               "loss_scaling": old_loss_scaling,
                               "train_img_per_sec": (num_batches * input_data.size()[0] / elapsed_time),
                               "latency_sec": elapsed_time / (num_batches * num_instances),
                               "validation_accuracy": validation_accuracy}
                utils.Logger.log_train_results(result_dict)

            # lr schedule
            lr_scheduler.step(epoch_num)
            new_lr = lr_scheduler.get_last_lr()[0]
            if new_lr != old_lr or new_loss_scaling != old_loss_scaling:
                if new_loss_scaling != old_loss_scaling:
                    optimizer.loss_scaling = new_loss_scaling
                    if opts.optimizer == 'sgd':
                        optimizer.param_groups[0]["velocity_scaling"] = new_loss_scaling / opts.loss_velocity_scaling_ratio
                        optimizer.param_groups[1]["velocity_scaling"] = new_loss_scaling / opts.loss_velocity_scaling_ratio
                training_model.setOptimizer(optimizer)
                old_lr = new_lr
                old_loss_scaling = new_loss_scaling
                if opts.lr_schedule == "step":
                    logging.info(f"Learning rate is changed to {new_lr}")

        epoch_end_time = time.time()
        if not opts.disable_metrics:
            aggregated_acc = metrics.get_running_mean("accuracy")
            logging.info(f"Epoch {epoch}: Train accuracy is {aggregated_acc:0.2f}%")
        elapsed_time = epoch_end_time-epoch_start_time
        # sync metrics
        if opts.use_popdist:
            total_sample = utils.sync_metrics(total_sample)
            elapsed_time = utils.sync_metrics(elapsed_time)
        epoch_throughput = total_sample / elapsed_time
        logging.info(f"Throughput of the epoch:{epoch_throughput:0.1f} img/sec")
        # save, in case of multiple processes save the weights only from the first instance.
        if not opts.checkpoint_path == "" and (not opts.use_popdist or opts.popdist_rank == 0):
            if not os.path.exists(opts.checkpoint_path):
                os.makedirs(opts.checkpoint_path)
            filename = f"{opts.model}_{opts.data}_{epoch}.pt"
            save_path = os.path.join(opts.checkpoint_path, filename)
            state = training_model.model.model.state_dict()
            optimizer_state = optimizer.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state,
                'optimizer_state_dict': optimizer_state,
                'loss': aggregated_loss,
                'train_accuracy': aggregated_acc,
                'opts': opts
            }, save_path)


def create_model_opts(opts):
    if opts.use_popdist:
        model_opts = popdist.poptorch.Options(ipus_per_replica=len(opts.pipeline_splits) + 1)
    else:
        model_opts = poptorch.Options()
        model_opts.replicationFactor(opts.replicas)
    model_opts.deviceIterations(opts.device_iterations)
    # Set mean reduction
    model_opts.Training.accumulationReductionType(poptorch.ReductionType.Mean)
    model_opts.Training.gradientAccumulation(opts.gradient_accumulation)
    if opts.seed is not None:
        model_opts.randomSeed(opts.seed)
    return model_opts


def convert_to_ipu_model(model, opts, optimizer):
    model_opts = create_model_opts(opts)
    model_opts = utils.train_settings(opts, model_opts)
    replica_count = opts.replicas * (opts.popdist_size if opts.use_popdist else 1)
    model_with_loss = TrainingModelWithLoss(model, replicas=replica_count, label_smoothing=opts.label_smoothing)
    training_model = poptorch.trainingModel(model_with_loss, model_opts, optimizer=optimizer)
    return training_model


def get_optimizer(opts, model):
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {'params': regularized_params, 'weight_decay': opts.weight_decay},
        {'params': non_regularized_params, 'weight_decay': 0}
    ]


    if opts.optimizer == 'sgd':
        optimizer = SGD(params, lr=opts.lr, momentum=opts.momentum, loss_scaling=opts.initial_loss_scaling, velocity_scaling=opts.initial_loss_scaling / opts.loss_velocity_scaling_ratio)
    elif opts.optimizer == 'adamw':
        optimizer = AdamW(params, lr=opts.lr, loss_scaling=opts.initial_loss_scaling)
    elif opts.optimizer == 'rmsprop':
        optimizer = RMSprop(params, lr=opts.lr, alpha=opts.rmsprop_decay, momentum=opts.momentum, loss_scaling=opts.initial_loss_scaling)

    # Make optimizers distributed
    if opts.use_popdist:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    return optimizer


def get_validation_function(opts, model):
    if run_opts.validation_mode == "none":
        return None
    inference_model_opts = create_validation_opts(opts)
    test_data = datasets.get_data(opts, inference_model_opts, train=False, async_dataloader=True)
    inference_model = poptorch.inferenceModel(model, inference_model_opts)

    def validation_func():
        model.eval()
        if inference_model._executable:
            inference_model.attachToDevice()
        val_acc = test(inference_model, test_data, opts)
        inference_model.detachFromDevice()
        return val_acc
    return validation_func


def get_lr_scheduler(opts, optimizer):
    if opts.lr_schedule == "step":
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=opts.lr_epoch_decay, gamma=opts.lr_decay)
    elif opts.lr_schedule == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=opts.epoch)
    elif opts.lr_schedule == "exponential":
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=opts.lr_decay)

    if opts.lr_scheduler_freq > 0:
        lr_scheduler = PeriodicLRDecorator(optimizer=optimizer, lr_scheduler=lr_scheduler, period=1./opts.lr_scheduler_freq)

    if opts.warmup_epoch > 0:
        lr_scheduler = WarmUpLRDecorator(optimizer=optimizer, lr_scheduler=lr_scheduler, warmup_epoch=opts.warmup_epoch)

    return lr_scheduler


if __name__ == '__main__':
    run_opts = parse_arguments()

    logging.info("Loading the data")
    model_opts = create_model_opts(run_opts)
    train_data = datasets.get_data(run_opts, model_opts, train=True, async_dataloader=True)

    logging.info("Initialize the model")
    model = models.get_model(run_opts, datasets.datasets_info[run_opts.data], pretrained=False)
    model.train()

    optimizer = get_optimizer(run_opts, model)
    lr_scheduler = get_lr_scheduler(run_opts, optimizer)
    training_model = convert_to_ipu_model(model, run_opts, optimizer)
    training_validation_func = get_validation_function(run_opts, model) if run_opts.validation_mode == "during" else None
    train(training_model, train_data, run_opts, lr_scheduler, range(1, run_opts.epoch+1), optimizer, training_validation_func)

    # Validation and weight averaging runs on single process
    if not run_opts.use_popdist or run_opts.popdist_rank == 0:
        if run_opts.weight_avg_strategy != 'none':
            average_fn = weight_avg.create_average_fn(run_opts)
            weight_avg.average_model_weights(run_opts.checkpoint_path, average_fn, run_opts.weight_avg_N)

        if run_opts.validation_mode == "after":
            if run_opts.checkpoint_path == "":
                training_model.destroy()
                val_func = get_validation_function(run_opts, model)
                acc = val_func()
                result_dict = {"validation_epoch": run_opts.epoch,
                               "validation_iteration": run_opts.logs_per_epoch * run_opts.epoch,
                               "validation_accuracy": acc}
                utils.Logger.log_validate_results(result_dict)
            else:
                training_model.destroy()
                checkpoint_files = [os.path.join(run_opts.checkpoint_path, file_name) for file_name in os.listdir(run_opts.checkpoint_path) if file_name.endswith(".pt")]
                validate_checkpoints(checkpoint_files)
