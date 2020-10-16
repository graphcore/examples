# Copyright 2020 Graphcore Ltd.
from tqdm import tqdm
import os
import time
import torch
import poptorch

import logging
import copy

from torch.optim import Adam
from poptorch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from dataloader import AsyncDataLoader
from lr_schedule import WarmupMultiStepLR
from train_utils import parse_arguments, accuracy
from data import get_data, datasets_info
from validate import test, validate_checkpoints
import sys
sys.path.append('..')
import models  # noqa: E402
import utils  # noqa: E402


# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss_scaling=1.0):
        super().__init__()
        self.model = model
        self.loss_scaling = loss_scaling
        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, args, loss_inputs=None):
        output = self.model(args)
        if loss_inputs is None:
            return output
        else:
            # Calculate loss in full precision
            output = output.float()
            loss = self.loss(output, loss_inputs) * self.loss_scaling
            return output, loss


def train(training_model, training_data, opts, lr_scheduler, epochs, optimizer):
    nr_batches = len(training_data)
    try:
        for epoch in epochs:
            logging.info("Epoch {0}/{1}".format(epoch, opts.epoch))
            bar = tqdm(training_data)
            sum_loss = 0.0
            sum_acc = 0.0
            start_time = time.time()
            total_sample = 0
            for batch_idx, (data, labels) in enumerate(bar):
                preds, losses = training_model(data, labels)
                if not opts.disable_metrics:
                    with torch.no_grad():
                        # Convert to full precision for CPU execute.
                        losses = losses.float()
                        preds = preds.float()
                        mean_loss = torch.mean(losses).item()
                        acc = accuracy(preds, labels)
                    sum_acc += acc
                    sum_loss += mean_loss
                    aggregated_loss = sum_loss / (batch_idx+1)
                    aggregated_accuracy = sum_acc / (batch_idx+1)
                    bar.set_description("Loss:{:0.4f} | Accuracy:{:0.2f}%".format(aggregated_loss, aggregated_accuracy))

                total_sample += data.size()[0]

            end_time = time.time()
            if not opts.disable_metrics:
                print("Epoch {}: Train accuracy is {:0.2f}%".format(epoch, aggregated_accuracy))
            print("Throughput of the epoch:{:0.1f} img/sec".format(total_sample / (end_time-start_time)))
            # save
            if not opts.checkpoint_path == "":
                if not os.path.exists(opts.checkpoint_path):
                    os.makedirs(opts.checkpoint_path)
                filename = "{0}_{1}_{2}.pt".format(opts.model, opts.data, epoch)
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

            # lr schedule
            if not(lr_scheduler is None):
                lr_scheduler.step()
                new_optimizer = copy.copy(optimizer)
                training_model.setOptimizer(new_optimizer)
                logging.info(f"Learning rate is changed to {lr_scheduler.get_last_lr()}")
    finally:
        # kill the process which fetch the data
        if isinstance(training_data, AsyncDataLoader):
            training_data.stop_data_fetch()


def create_model_opts(opts):
    model_opts = poptorch.Options().deviceIterations(opts.device_iteration)
    model_opts.Training.gradientAccumulation(opts.gradient_accumulation)
    model_opts.replicationFactor(opts.replicas)
    return model_opts


def convert_to_ipu_model(model, opts, optimizer):
    model_opts = create_model_opts(opts)
    # PopART settings
    if opts.enable_stochastic_rounding:
        model_opts.Popart.set("enableStochasticRounding", True)
    if opts.data == "synthetic":
        model_opts.Popart.set("syntheticDataMode", 2)
    if opts.half_partial:
        model_opts.Popart.set("partialsTypeMatMuls", "half")
        model_opts.Popart.set("convolutionOptions", {'partialsType': 'half'})

    if opts.enable_pipeline_recompute and len(opts.pipeline_splits) > 0:
        model_opts.Popart.set("autoRecomputation", 3)

    # disable prefetch to save memory
    if opts.replicas > 1:
        model_opts.Popart.set("enablePrefetchDatastreams", False)
    model_opts.Popart.set("disableGradAccumulationTensorStreams", True)

    num_stages = len(opts.pipeline_splits)+1
    if len(opts.available_memory_proportion) == 1:
        model_opts.setAvailableMemoryProportion({f'IPU{i}': opts.available_memory_proportion[0] for i in range(num_stages)})
    elif len(opts.available_memory_proportion) > 1:
            model_opts.setAvailableMemoryProportion({f'IPU{i}': amp for i, amp in enumerate(opts.available_memory_proportion)})

    # Scale the loss to be the same as bs=1 on a single IPU training.
    loss_scaling_factor = (1.0 / opts.batch_size)
    model_with_loss = TrainingModelWithLoss(model, loss_scaling_factor)
    training_model = poptorch.trainingModel(model_with_loss, model_opts, optimizer=optimizer)
    return training_model


def get_optimizer(opts, model):
    if opts.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=opts.lr, momentum=opts.momentum, loss_scaling=opts.loss_scaling, velocity_scaling=opts.loss_scaling)
    else:
        optimizer = Adam(model.parameters(), lr=opts.lr)

    lr_scheduler = None
    if opts.lr_schedule == "step":
        if opts.warmup_epoch > 0:
            lr_scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=opts.lr_epoch_decay, lr=opts.lr, warmup_epoch=opts.warmup_epoch, gamma=opts.lr_decay)
        else:
            lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=opts.lr_epoch_decay, gamma=opts.lr_decay)
    return optimizer, lr_scheduler


if __name__ == '__main__':
    run_opts = parse_arguments()
    utils.setup_logging_folder(run_opts)
    if run_opts.batch_size == 1 and run_opts.normlayer == "batch":
        logging.warning("The BatchNorm with batch size of 1 may cause instability during the inference.")
    # Scale learning rate
    run_opts.lr = run_opts.lr / (run_opts.replicas * run_opts.gradient_accumulation)

    logging.info("Loading the data")
    model_opts = create_model_opts(run_opts)
    train_data, test_data = get_data(run_opts, model_opts)

    logging.info("Initialize the model")
    model = models.get_model(run_opts, datasets_info[run_opts.data], pretrained=False)
    model.train()

    optimizer, lr_scheduler = get_optimizer(run_opts, model)
    training_model = convert_to_ipu_model(model, run_opts, optimizer)
    train(training_model, train_data, run_opts, lr_scheduler, range(1, run_opts.epoch+1), optimizer)

    if not run_opts.no_validation:
        if run_opts.checkpoint_path == "":
            training_model.copyWeightsToHost()
            del training_model
            inference_model_opts = poptorch.Options().deviceIterations(run_opts.device_iteration)
            inference_model_opts.replicationFactor(run_opts.replicas)
            model.eval()
            inference_model = poptorch.inferenceModel(model, inference_model_opts)
            test(inference_model, test_data, run_opts)
        else:
            del training_model
            checkpoint_files = [os.path.join(run_opts.checkpoint_path, file_name) for file_name in os.listdir(run_opts.checkpoint_path)]
            validate_checkpoints(checkpoint_files, test_data)
