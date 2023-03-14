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
import numpy as np

from datetime import datetime
from poptorch.optim import SGD, RMSprop, AdamW
from lr_schedule import WarmUpLRDecorator, PeriodicLRDecorator
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from train_utils import parse_arguments
from validate import test, validate_checkpoints, create_validation_opts
import weight_avg
import sys
import import_helper
import models
import models.loss
import utils
import datasets
import datasets.augmentations as augmentations


def train(training_model, training_data, args, lr_scheduler, epochs, optimizer, validation_function=None):
    training_start_time = datetime.now()
    logging.info(f"Training the model. Start: {str(training_start_time)}")

    # A generic container used by the train function to set and update the host-side training state.
    class TrainingState(): pass
    state = TrainingState()

    state.iterations_per_epoch = len(training_data)
    metrics = utils.Metrics(running_mean_length=state.iterations_per_epoch, distributed=args.use_popdist)
    state.old_lr = lr_scheduler.get_last_lr()[0]
    state.num_instances = args.popdist_size if args.use_popdist else 1

    # Determine the loss scaling change points.
    num_loss_scaling_steps = int(math.log2(args.loss_scaling // args.initial_loss_scaling)) + 1
    loss_scaling_steps = {i * (args.epoch // num_loss_scaling_steps) + 1: args.initial_loss_scaling * (2 ** i) for i in range(num_loss_scaling_steps)}
    state.new_loss_scaling = state.old_loss_scaling = args.initial_loss_scaling

    if args.mixup_enabled or args.cutmix_enabled:
        augmentation_generator = np.random.default_rng(args.seed)

    for state.epoch in epochs:
        state.epoch_start_time = time.perf_counter()
        state.epoch_sample_size = 0
        logging.info(f"Epoch {state.epoch}/{args.epoch + args.fine_tune_epoch}")

        bar = tqdm(training_data, total=state.iterations_per_epoch)

        if state.epoch in loss_scaling_steps.keys():
            state.new_loss_scaling = loss_scaling_steps[state.epoch]

        # Beginning of the epoch.
        for state.batch_idx, (input_data, labels) in enumerate(bar):
            state.epoch_progress = (state.epoch - 1) + (state.batch_idx + 1) / state.iterations_per_epoch
            state.epoch_sample_size += labels.size()[0]

            if args.compile_only:
                _ = training_model.compile(input_data, labels)
                logging.info("Graph compilation complete, --compile-only was set, exiting.")
                sys.exit(0)

            loss, sublosses, metric_values = training_model(input_data, labels)
            if args.profile:
                # Profile report is only generated for one iteration.
                sys.exit(0)

            running_mean_loss, running_mean_acc = handle_metrics(
                metrics,
                loss,
                sublosses,
                metric_values[0],  # First item in the tuple is accuracy
                state,
                bar,
                validation_function,
                training_model,
                args,
            )

            update_lr(lr_scheduler, optimizer, training_model, state, args)

        # End of the epoch.
        if not args.checkpoint_output_dir == "":
            model_state = models.get_model_state_dict(training_model)
            optimizer_state = optimizer.state_dict()
            
            if args.use_popdist:
                popdist.execute_on_instances(
                    {0},
                    persist_checkpoint,
                    model_state,
                    optimizer_state,
                    args.model,
                    args.checkpoint_output_dir,
                    state,
                    running_mean_loss,
                    running_mean_acc,
                    args,
                )
            else:
                persist_checkpoint(
                    model_state,
                    optimizer_state,
                    args.model,
                    args.checkpoint_output_dir,
                    state,
                    running_mean_loss,
                    running_mean_acc,
                    args,
                )

    training_end_time = datetime.now()
    total_training_time = training_end_time - training_start_time
    logging.info(f"Finished training. Time: {str(training_end_time)}. It took: {str(total_training_time)}")


def get_augmented_samples(args, input_data, random_generator):
    # Mixup coefficients are sampled on the host, cutmix coefficients are
    # sampled on the device.
    batch_and_mixup_coefficients = [input_data]
    if args.mixup_enabled:
        mixup_coeffs = augmentations.sample_mixup_coefficients(
            alpha=args.mixup_alpha,
            global_batch_size=args.micro_batch_size * args.gradient_accumulation * args.replicas * args.device_iterations,
            np_type=np.float16 if args.precision[:3] == "16." else np.float,
            random_generator=random_generator,
        )
        batch_and_mixup_coefficients.append(mixup_coeffs)
    return tuple(batch_and_mixup_coefficients)


def handle_metrics(metrics, loss, sublosses, accuracy, state, bar, validation_function, training_model, args):
    def mean_metric(metric):
        # Each replica returns a sum of values. We need to normalize by
        # gradient accumulation and device iteration.
        with torch.no_grad():
            normalization_term = args.device_iterations * args.gradient_accumulation
            return torch.mean(metric).item() / normalization_term

    metrics.save_value("loss", mean_metric(loss))
    metrics.save_value("accuracy", mean_metric(accuracy))
    metrics.save_value("classification_loss", mean_metric(sublosses[0]))
    metrics.save_value("smoothing_loss", mean_metric(sublosses[1]) if len(sublosses) > 1 else 0.0)

    if is_time_to_log_metrics(state, args):
        # Log detailed metrics to wandb and/or stdout.
        current_throughput = compute_throughput(args, state)

        metric_names = ["loss", "accuracy", "classification_loss", "smoothing_loss"]
        mean_values, running_mean_values = metrics.compute_mean_values(
            names=metric_names,
            running_mean_names=metric_names,
        )

        mean_loss = mean_values[0]
        mean_acc = mean_values[1]
        mean_classification_loss = mean_values[2]
        mean_smoothing_loss = mean_values[3]
        running_mean_loss = running_mean_values[0]
        running_mean_acc = running_mean_values[1]
        running_mean_classification_loss = running_mean_values[2]
        running_mean_smoothing_loss = running_mean_values[3]

        if validation_function is not None and (state.epoch % args.validation_frequency == 0):
            training_model.detachFromDevice()
            validation_accuracy = validation_function()
            training_model.attachToDevice()
            logging.info(f"Validation Accuracy: {validation_accuracy:0.2f}%")
        else:
            validation_accuracy = 0.0

        if not args.use_popdist or args.popdist_rank == 0:
            log_data = {
                "loss_avg": running_mean_loss,
                "loss_batch": mean_loss,
                "epoch": state.epoch_progress,
                "iteration": state.epoch_progress * state.iterations_per_epoch * state.num_instances,
                "train_accuracy_avg": running_mean_acc,
                "train_accuracy_batch": mean_acc,
                "learning_rate": state.old_lr,
                "train_img_per_sec": current_throughput,
                "validation_accuracy": validation_accuracy,
                "classification_loss_batch": mean_classification_loss,
                "classification_loss_avg": running_mean_classification_loss,
                "smoothing_loss_batch:": mean_smoothing_loss,
                "smoothing_loss_avg:": running_mean_smoothing_loss,
            }
            if not args.auto_loss_scaling:
                log_data["loss_scaling"] = state.old_loss_scaling
            if args.wandb_weight_histogram:
                utils.Logger.log_model_histogram(models.get_nested_model(training_model))
            utils.Logger.log_train_results(log_data)
            bar.set_description(f"Loss:{running_mean_loss:0.4f} | Accuracy:{running_mean_acc:0.2f}%")
    else:
        # Only update the progress bar.
        metric_names = ["loss", "accuracy"]
        _, running_mean_values = metrics.compute_mean_values(
            names=[],
            running_mean_names=metric_names,
        )
        running_mean_loss = running_mean_values[0]
        running_mean_acc = running_mean_values[1]
        if not args.use_popdist or args.popdist_rank == 0:
            bar.set_description(f"Loss:{running_mean_loss:0.4f} | Accuracy:{running_mean_acc:0.2f}%")

    if is_end_of_epoch(state):
        metrics.reset_values()
        if not args.use_popdist or args.popdist_rank == 0:
            logging.info(f"Epoch {state.epoch}")
            # Standardised metric reporting
            logging.info(f"loss: {running_mean_loss:0.4f},")
            logging.info(f"accuracy: {running_mean_acc:0.2f} %")
            logging.info(f"throughput: {current_throughput:0.1f} samples/sec")

    return running_mean_loss, running_mean_acc


def is_time_to_log_metrics(state, args):
    # This can happen at the middle of the epoch (depending on logs_per_epoch)
    # and it always happens at the end of the epoch.
    return (state.batch_idx + 1) % (state.iterations_per_epoch // args.logs_per_epoch) == 0


def is_end_of_epoch(state):
    return (state.batch_idx + 1) % state.iterations_per_epoch == 0


def compute_throughput(args, state):
    epoch_elapsed_time = time.perf_counter() - state.epoch_start_time
    sample_size = state.epoch_sample_size
    if args.use_popdist:
        epoch_elapsed_time, sample_size = utils.synchronize_throughput_values(
            epoch_elapsed_time,
            sample_size,
        )
    return sample_size / epoch_elapsed_time


def update_lr(lr_scheduler, optimizer, training_model, state, args):
    lr_scheduler.step(state.epoch_progress)
    state.new_lr = lr_scheduler.get_last_lr()[0]
    if state.new_lr != state.old_lr or state.new_loss_scaling != state.old_loss_scaling:
        if state.new_loss_scaling != state.old_loss_scaling:
            optimizer.loss_scaling = state.new_loss_scaling
            if args.optimizer == 'sgd_combined':
                optimizer.param_groups[0]["velocity_scaling"] = state.new_loss_scaling
                optimizer.param_groups[1]["velocity_scaling"] = state.new_loss_scaling
        training_model.setOptimizer(optimizer)
        state.old_lr = state.new_lr
        state.old_loss_scaling = state.new_loss_scaling
        if args.lr_schedule == "step":
            logging.info(f"Learning rate is changed to {state.new_lr}")


def persist_checkpoint(model_state, optimizer_state, model_name, checkpoint_output_path, state, running_mean_loss, running_mean_acc, args):

    # Save the weights in the first process only.
    if not os.path.exists(checkpoint_output_path):
        os.makedirs(checkpoint_output_path)

    save_path = os.path.join(checkpoint_output_path, f"{model_name}_{args.data}_{state.epoch}.pt")
    save_data = {
        'epoch': state.epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': running_mean_loss,
        'train_accuracy': running_mean_acc,
        'args': args,
    }
    torch.save(save_data, save_path)

    print(f"Checkpoint saved to: {save_path}")

def create_training_opts(args):
    ipus_per_replica = len(args.pipeline_splits) + 1
    total_replicas = args.replicas
    if args.use_popdist:
        opts = popdist.poptorch.Options(ipus_per_replica = ipus_per_replica)
        total_replicas = popdist.getNumTotalReplicas()
    else:
        opts = poptorch.Options()
        opts.replicationFactor(args.replicas)
    logging.info("Total replicas: " + str(total_replicas))
    logging.info("Global batch size: " + str(total_replicas * args.micro_batch_size * args.gradient_accumulation))
    logging.info("Number of IPUs required: " + str(total_replicas * ipus_per_replica))
    opts = utils.train_settings(args, opts)
    opts.deviceIterations(args.device_iterations)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
    opts.Training.gradientAccumulation(args.gradient_accumulation)

    if args.seed is not None:
        opts.randomSeed(args.seed)

    return opts


def convert_to_ipu_model(model, args, optimizer):
    opts = create_training_opts(args)
    training_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
    return training_model


def get_optimizer(args, model):
    regularized_params = []
    non_regularized_params = []

    # Filter biases and norm parameters.
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {'params': regularized_params, 'weight_decay': args.weight_decay},
        {'params': non_regularized_params, 'weight_decay': 0}
    ]

    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = SGD(params, lr=args.lr, momentum=args.momentum, loss_scaling=args.initial_loss_scaling, use_combined_accum=False)
    elif args.optimizer == 'sgd_combined':
        optimizer = SGD(params, lr=args.lr, momentum=args.momentum, loss_scaling=args.initial_loss_scaling, velocity_scaling=args.initial_loss_scaling, use_combined_accum=True)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(params, lr=args.lr, loss_scaling=args.initial_loss_scaling, eps=args.optimizer_eps)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(params, lr=args.lr, alpha=args.rmsprop_decay, momentum=args.momentum, loss_scaling=args.initial_loss_scaling, eps=args.optimizer_eps)
    elif args.optimizer == 'rmsprop_tf':
        optimizer = RMSprop(params, lr=args.lr, alpha=args.rmsprop_decay, momentum=args.momentum, loss_scaling=args.initial_loss_scaling, eps=args.optimizer_eps, use_tf_variant=True)
    return optimizer


def get_validation_function(args, model):
    class ValidationFunction:
        def __init__(self, func, iterations_per_epoch):
            self.func = func
            self.validation_iterations_per_epoch = iterations_per_epoch

    if args.validation_mode == "none":
        return None

    if args.mixup_enabled or args.cutmix_enabled:
        assert isinstance(model, augmentations.AugmentationModel)
        model = model.model

    opts = create_validation_opts(args, use_popdist=args.use_popdist)
    test_data = datasets.get_data(args, opts, train=False, async_dataloader=True, return_remaining=True)
    inference_model = poptorch.inferenceModel(model, opts)

    def validation_func():
        model.eval()
        if inference_model._executable:
            inference_model.attachToDevice()
        val_acc = test(inference_model, test_data)
        inference_model.detachFromDevice()
        model.train()
        return val_acc
    return ValidationFunction(validation_func, len(test_data))


def get_lr_scheduler(args, optimizer, step_per_epoch, start_epoch=0):
    scheduler_freq = args.lr_scheduler_freq if args.lr_scheduler_freq > 0.0 else step_per_epoch
    scheduler_last_epoch = (scheduler_freq * start_epoch) - 1
    if args.lr_schedule == "step":
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[step*scheduler_freq for step in args.lr_epoch_decay], gamma=args.lr_decay, last_epoch=scheduler_last_epoch)
    elif args.lr_schedule == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch*scheduler_freq, last_epoch=scheduler_last_epoch)
    elif args.lr_schedule == "exponential":
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=args.lr_decay, last_epoch=scheduler_last_epoch)

    lr_scheduler = PeriodicLRDecorator(optimizer=optimizer, lr_scheduler=lr_scheduler, period=1./scheduler_freq)
    lr_scheduler = WarmUpLRDecorator(optimizer=optimizer, lr_scheduler=lr_scheduler, warmup_epoch=args.warmup_epoch)
    return lr_scheduler


def fine_tune(args):
    logging.info("Fine-tuning the model after half resolution training")
    args.half_res_training = False
    args.mixup_enabled = False
    args.cutmix_enabled = False
    args.optimizer = 'sgd'
    args.momentum = 0.0
    args.warmup_epoch = 0
    args.lr = args.fine_tune_lr
    args.lr_schedule = 'cosine'
    args.lr_scheduler_freq = 0
    args.micro_batch_size = args.fine_tune_micro_batch_size
    args.gradient_accumulation = args.fine_tune_gradient_accumulation
    opts = create_training_opts(args)

    train_data = datasets.get_data(args, opts, train=True, fine_tuning=True, async_dataloader=True)
    model_fine_tune = models.get_model(args, datasets.datasets_info[args.data], pretrained=False, use_mixup=args.mixup_enabled, use_cutmix=args.cutmix_enabled, with_loss=True, inference_mode=False)

    if not args.use_popdist or args.popdist_rank == 0:
        avg_checkpoint_file = os.path.join(args.checkpoint_input_dir, f"{args.model}_{args.data}_{args.epoch}_averaged.pt")
        avg_checkpoint = torch.load(avg_checkpoint_file)
        models.load_model_state_dict(model_fine_tune, avg_checkpoint['model_state_dict'])

    if args.use_popdist:
        hvd.broadcast_parameters(models.get_model_state_dict(model_fine_tune), root_rank=0)

    model_fine_tune.train()
    nested_model = model_fine_tune.nested_model[0]

    # Freeze relevant parameters.
    for param_name, param in nested_model.named_parameters():
        param_name = param_name.replace('.', '/')
        if param_name.startswith(args.fine_tune_first_trainable_layer):
            break
        logging.info(f"Freezing parameter {param_name}")
        param.requires_grad = False

    # Make relevant dropout and batch norm layers eval.
    for module_name, module in nested_model.named_modules():
        module_name = module_name.replace('.', '/')
        if module_name.startswith(args.fine_tune_first_trainable_layer):
            break
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, torch.nn.modules.dropout._DropoutNd):
            logging.info(f"Setting module {module_name} to eval mode")
            module.eval()

    optimizer = get_optimizer(args, model_fine_tune)
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_data))
    training_model = convert_to_ipu_model(model_fine_tune, args, optimizer)
    train(training_model, train_data, args, lr_scheduler, range(args.epoch + 1, args.epoch + 1 + args.fine_tune_epoch), optimizer)
    train_data.terminate()
    return model_fine_tune, training_model


if __name__ == '__main__':
    args = parse_arguments()
    opts = create_training_opts(args)
    train_data = datasets.get_data(args, opts, train=True, async_dataloader=True)

    model = models.get_model(args, datasets.datasets_info[args.data], pretrained=False, use_mixup=args.mixup_enabled, use_cutmix=args.cutmix_enabled, with_loss=True, inference_mode=False)
    if args.use_popdist:
        hvd.broadcast_parameters(models.get_model_state_dict(model), root_rank=0)

    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_data))
    training_model = convert_to_ipu_model(model, args, optimizer)

    if args.validation_mode == "during":
        training_validation_func = get_validation_function(args, model).func
    else:
        training_validation_func = None

    train(training_model, train_data, args, lr_scheduler, range(1, args.epoch + 1), optimizer, training_validation_func)
    train_data.terminate()

    if args.weight_avg_strategy != 'none' and (not args.use_popdist or args.popdist_rank == 0):
        average_fn = weight_avg.create_average_fn(args)
        weight_avg.average_model_weights(args.checkpoint_input_dir, args.checkpoint_output_dir, average_fn, args.weight_avg_N)

    if args.half_res_training:
        training_model.destroy()
        model, training_model = fine_tune(args)

    if args.validation_mode == "after":
        training_model.destroy()
        if args.checkpoint_input_dir == "":
            validation_function = get_validation_function(args, model)
            val_accuracy = validation_function.func()
            if not args.use_popdist or args.popdist_rank == 0:
                log_data = {
                    "validation_epoch": args.epoch + args.fine_tune_epoch,
                    "validation_iteration": (args.epoch + args.fine_tune_epoch) * validation_function.validation_iterations_per_epoch,
                    "validation_accuracy": val_accuracy,
                }
                utils.Logger.log_validate_results(log_data)
        else:
            checkpoint_files = [os.path.join(args.checkpoint_input_dir, file_name) for file_name in os.listdir(args.checkpoint_input_dir) if file_name.endswith(".pt")]
            if args.use_popdist:
                popdist.execute_on_instances({0}, validate_checkpoints, checkpoint_files)
            else:
                validate_checkpoints(checkpoint_files)
