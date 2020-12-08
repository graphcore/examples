# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tqdm import tqdm
import os
import time
import torch
import poptorch
import popart
import logging

from poptorch.optim import SGD, RMSprop, AdamW
from lr_schedule import WarmUpLRDecorator, PeriodicLRDecorator
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from train_utils import parse_arguments, accuracy
from validate import test, validate_checkpoints
import weight_avg
import sys
sys.path.append('..')
import models
import utils
import data


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss_scaling=1.0, label_smoothing=0.0, reduction='sum'):
        super().__init__()
        self.model = model
        self.loss_scaling = loss_scaling
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        if self.label_smoothing > 0.0:
            self.label_smoothing_loss = torch.nn.NLLLoss(reduction=reduction)
            self.reduction_op = torch.sum if reduction == 'sum' else torch.mean


    def forward(self, args, loss_inputs=None):
        output = self.model(args)
        if loss_inputs is None:
            return output
        else:
            # Calculate loss in full precision
            output = output.float()
            if self.label_smoothing > 0.0:
                # cross entropy between uniform distribution and output distribution
                log_preds = torch.nn.functional.log_softmax(output, dim=1)
                smoothing_loss = self.reduction_op(-log_preds.mean(dim=1), dim=0) * self.label_smoothing
                classification_loss = (1.0-self.label_smoothing) * self.label_smoothing_loss(log_preds, loss_inputs)
                final_loss = smoothing_loss + classification_loss
            else:
                final_loss = self.loss(output, loss_inputs)
            return output, poptorch.identity_loss(final_loss * self.loss_scaling, reduction='none')


def train(training_model, training_data, opts, lr_scheduler, epochs, optimizer):
    old_lr = lr_scheduler.get_last_lr()[0]
    iterations_per_epoch = len(training_data)
    for epoch in epochs:
        logging.info(f"Epoch {epoch}/{opts.epoch}")
        bar = tqdm(training_data, total=iterations_per_epoch)
        sum_loss = 0.0
        sum_acc = 0.0
        sum_batch_loss = 0.0
        sum_batch_acc = 0.0
        last_batch = -1
        start_batch_time = start_epoch_time = time.time()
        total_sample = 0
        for batch_idx, (input_data, labels) in enumerate(bar):
            preds, losses = training_model(input_data, labels)
            epoch_num = epoch - 1 + float(batch_idx+1) / iterations_per_epoch
            if not opts.disable_metrics:
                with torch.no_grad():
                    # Convert to full precision for CPU execute.
                    losses = losses.float()
                    preds = preds.float()
                    mean_loss = torch.mean(losses).item()
                    acc = accuracy(preds, labels)
                sum_acc += acc
                sum_loss += mean_loss
                sum_batch_loss += mean_loss
                sum_batch_acc += acc
                aggregated_loss = sum_loss / (batch_idx+1)
                aggregated_accuracy = sum_acc / (batch_idx+1)
                bar.set_description(f"Loss:{aggregated_loss:0.4f} | Accuracy:{aggregated_accuracy:0.2f}%")
            total_sample += input_data.size()[0]

            if not opts.disable_metrics and ((batch_idx + 1) % (iterations_per_epoch // opts.logs_per_epoch) == 0):
                # save metrics
                result_dict = {"loss_avg": aggregated_loss,
                               "loss_batch": sum_batch_loss / (batch_idx - last_batch),
                               "epoch": epoch_num,
                               "iteration": batch_idx+1+(epoch-1)*iterations_per_epoch,
                               "train_accuracy_avg": aggregated_accuracy,
                               "train_accuracy_batch": sum_batch_acc / (batch_idx - last_batch),
                               "learning_rate": old_lr * (opts.replicas * opts.gradient_accumulation if opts.reduction == 'sum' else 1.0),
                               "train_img_per_sec": ((batch_idx-last_batch) * input_data.size()[0] / (time.time()-start_batch_time)),
                               "latency_sec": (time.time()-start_batch_time) / (batch_idx-last_batch)}
                utils.Logger.log_train_results(result_dict)
                sum_batch_loss = 0.0
                sum_batch_acc = 0.0
                last_batch = batch_idx
                start_batch_time = time.time()

            # lr schedule
            lr_scheduler.step(epoch_num)
            new_lr = lr_scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                training_model.setOptimizer(optimizer)
                old_lr = new_lr
                if opts.lr_schedule == "step":
                    logging.info(f"Learning rate is changed to {new_lr}")

        end_time = time.time()
        if not opts.disable_metrics:
            logging.info(f"Epoch {epoch}: Train accuracy is {aggregated_accuracy:0.2f}%")
        epoch_throughput = total_sample / (end_time-start_epoch_time)
        logging.info(f"Throughput of the epoch:{epoch_throughput:0.1f} img/sec")
        # save
        if not opts.checkpoint_path == "":
            if not os.path.exists(opts.checkpoint_path):
                os.makedirs(opts.checkpoint_path)
            filename = f"{opts.model}_{opts.data}_{epoch}.pt"
            save_path = os.path.join(opts.checkpoint_path, filename)
            training_model.copyWeightsToHost()
            state = training_model.model.model.state_dict()
            optimizer_state = optimizer.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state,
                'optimizer_state_dict': optimizer_state,
                'loss': aggregated_loss,
                'train_accuracy': aggregated_accuracy,
                'opts': opts
            }, save_path)


def create_model_opts(opts):
    model_opts = poptorch.Options().deviceIterations(opts.device_iterations)
    model_opts.Training.gradientAccumulation(opts.gradient_accumulation)
    model_opts.replicationFactor(opts.replicas)
    return model_opts


def convert_to_ipu_model(model, opts, optimizer):
    model_opts = create_model_opts(opts)
    # PopART settings
    model_opts.Popart.set("enableStochasticRounding", opts.enable_stochastic_rounding)
    if opts.data == "synthetic":
        model_opts.Popart.set("syntheticDataMode", int(popart.SyntheticDataMode.RandomNormal))
    if opts.half_partial:
        model_opts.Popart.set("partialsTypeMatMuls", "half")
        model_opts.Popart.set("convolutionOptions", {'partialsType': 'half'})

    if opts.enable_pipeline_recompute and len(opts.pipeline_splits) > 0:
        model_opts.Popart.set("autoRecomputation", int(popart.RecomputationType.Pipeline))

    # disable prefetch to save memory
    if opts.replicas > 1:
        model_opts.Popart.set("enablePrefetchDatastreams", False)
    model_opts.Popart.set("disableGradAccumulationTensorStreams", True)

    num_stages = len(opts.pipeline_splits)+1
    if len(opts.available_memory_proportion) == 1:
        model_opts.setAvailableMemoryProportion({f'IPU{i}': opts.available_memory_proportion[0] for i in range(num_stages)})
    elif len(opts.available_memory_proportion) > 1:
            model_opts.setAvailableMemoryProportion({f'IPU{i}': amp for i, amp in enumerate(opts.available_memory_proportion)})

    if opts.reduction == 'mean':
        model_opts.Popart.set('accumulationReductionType', int(popart.ReductionType.Mean))

    if opts.disable_metrics:
        # if not interested in accurate metrics, return only subset of the predictions
        model_opts.anchorMode(poptorch.AnchorMode.Final)
    else:
        model_opts.anchorMode(poptorch.AnchorMode.All)

    # Scale the loss to be the same as bs=1 on a single IPU training.
    loss_scaling = 1.0 / opts.batch_size if opts.reduction == 'sum' else 1.0
    model_with_loss = TrainingModelWithLoss(model, loss_scaling=loss_scaling, label_smoothing=opts.label_smoothing, reduction=opts.reduction)
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
        optimizer = SGD(params, lr=opts.lr, momentum=opts.momentum, weight_decay=0, loss_scaling=opts.loss_scaling, velocity_scaling=opts.loss_scaling)
    elif opts.optimizer == 'adamw':
        optimizer = AdamW(params, lr=opts.lr, weight_decay=0, loss_scaling=opts.loss_scaling)
    elif opts.optimizer == 'rmsprop':
        optimizer = RMSprop(params, lr=opts.lr, alpha=opts.rmsprop_decay, momentum=opts.momentum, weight_decay=0, loss_scaling=opts.loss_scaling)

    return optimizer


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

    # Scale learning rate and weight decay
    if run_opts.reduction == 'sum':
        run_opts.lr = run_opts.lr / (run_opts.replicas * run_opts.gradient_accumulation)
        run_opts.weight_decay = run_opts.weight_decay * (run_opts.replicas * run_opts.gradient_accumulation)


    logging.info("Loading the data")
    model_opts = create_model_opts(run_opts)
    train_data = data.get_data(run_opts, model_opts, train=True, async_dataloader=True)
    if not run_opts.no_validation:
        inference_model_opts = poptorch.Options().deviceIterations(max(model_opts.device_iterations, 1+len(run_opts.pipeline_splits)))
        inference_model_opts.replicationFactor(run_opts.replicas)
        test_data = data.get_data(run_opts, inference_model_opts, train=False, async_dataloader=True)

    logging.info("Initialize the model")
    model = models.get_model(run_opts, data.datasets_info[run_opts.data], pretrained=False)
    model.train()

    optimizer = get_optimizer(run_opts, model)
    lr_scheduler = get_lr_scheduler(run_opts, optimizer)
    training_model = convert_to_ipu_model(model, run_opts, optimizer)
    train(training_model, train_data, run_opts, lr_scheduler, range(1, run_opts.epoch+1), optimizer)

    if run_opts.weight_avg_strategy != 'none':
        average_fn = weight_avg.create_average_fn(run_opts)
        weight_avg.average_model_weights(run_opts.checkpoint_path, average_fn)

    if not run_opts.no_validation:
        if run_opts.checkpoint_path == "":
            training_model.destroy()
            model.eval()
            inference_model = poptorch.inferenceModel(model, inference_model_opts)
            acc = test(inference_model, test_data, run_opts)
            result_dict = {"validation_epoch": run_opts.epoch,
                           "validation_iteration": run_opts.logs_per_epoch * run_opts.epoch,
                           "validation_accuracy": acc}
            utils.Logger.log_validate_results(result_dict)
            test_data.terminate()
        else:
            training_model.destroy()
            checkpoint_files = [os.path.join(run_opts.checkpoint_path, file_name) for file_name in os.listdir(run_opts.checkpoint_path) if file_name.endswith(".pt")]

            validate_checkpoints(checkpoint_files, test_data)
