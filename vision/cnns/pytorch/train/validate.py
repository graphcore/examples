# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import argparse
import torch
import poptorch
import logging
import popdist
from tqdm import tqdm
import import_helper
import models
import utils
import datasets


class DataPadder:
    def __init__(self, global_batch_size, model):
        self.global_batch_size = global_batch_size
        self.model = model

    def _extend(self, tensor):
        if tensor.size()[0] == self.global_batch_size:
            return tensor
        elif self.global_batch_size is None:
            # The batches created by the dataset instead of dataloader
            self.global_batch_size = tensor.size()[0]
            return tensor
        else:
            full_batch_size = list(tensor.size())
            full_batch_size[0] = self.global_batch_size
            returning_tensor = torch.zeros(tuple(full_batch_size), dtype=tensor.dtype)
            returning_tensor[:tensor.size()[0]] = tensor
            return returning_tensor


    def __call__(self, inputs):
        if isinstance(inputs, torch.Tensor):
            original_batch_size = inputs.size()[0]
            inputs = self._extend(inputs)
        else:
            original_batch_size = inputs[0].size()[0]
            inputs = tuple([self._extend(tensor) for tensor in list(inputs)])
        output = self.model(inputs)

        if isinstance(output, torch.Tensor):
            return output[:original_batch_size]
        else:
            return tuple([tensor[:original_batch_size] for tensor in output])


def test(inference_model, test_data):
    logging.info("Testing the model")
    nr_batches = len(test_data)
    bar = tqdm(test_data, total=nr_batches)
    sum_acc = 0.0
    sample_count = 0
    inference_model = DataPadder(test_data.combinedBatchSize, inference_model)
    for input_data, labels in bar:
        sample_count += labels.size()[0]
        output = inference_model(input_data)
        output = output.float()
        with torch.no_grad():
            sum_acc += utils.accuracy(output, labels) * labels.size()[0]
            aggregated_accuracy = sum_acc / sample_count
        bar.set_description(f"Accuracy:{aggregated_accuracy:0.2f}%")
    acc = sum_acc / sample_count
    logging.info(f"Accuracy on test set: {acc:0.2f}%")
    return acc


def load_checkpoint_weights(inference_model, model, file_path):
    checkpoint = torch.load(file_path)
    args = checkpoint['args']
    logging.info(f"Restore the {args.model} model to epoch {checkpoint['epoch']} on {args.data} dataset(train loss: {checkpoint['loss']}, train accuracy: {checkpoint['train_accuracy']}%)")
    models.load_model_state_dict(model, checkpoint['model_state_dict'])
    inference_model.copyWeightsToDevice()


def create_validation_opts(args, use_popdist):
    if use_popdist:
        opts = popdist.poptorch.Options(ipus_per_replica=len(args.pipeline_splits) + 1)
    else:
        opts = poptorch.Options()
        opts.replicationFactor(args.replicas)
        opts.Distributed.disable()
    opts = utils.inference_settings(args, opts)
    opts.deviceIterations(max(args.device_iterations, 1+len(args.pipeline_splits)))
    opts.outputMode(poptorch.OutputMode.All)
    return opts


def validate_checkpoints(checkpoint_list, test_data=None):
    checkpoint = torch.load(checkpoint_list[0])
    args = checkpoint['args']
    utils.Logger.setup_logging_folder(args)

    # make sure the order is ascending
    def ckpt_key(ckpt):
        return int(ckpt.split('_')[-1].split('.')[0])
    try:
        checkpoint_list = sorted(checkpoint_list, key=ckpt_key)
    except:
        logging.warn("Checkpoint names are changed, which may cause inconsistent order in evaluation.")

    # Validate in a single instance
    opts = create_validation_opts(args, use_popdist=False)
    args.use_popdist = False
    args.popdist_size = 1

    if test_data is None:
        test_data = datasets.get_data(args, opts, train=False, async_dataloader=True, return_remaining=True)
    model = models.get_model(args, datasets.datasets_info[args.data], pretrained=False, inference_mode=True)
    # Load the weights of the first checkpoint for the model
    models.load_model_state_dict(model, checkpoint['model_state_dict'])

    inference_model = poptorch.inferenceModel(model, opts)
    for checkpoint in checkpoint_list:
        if inference_model.isCompiled():
            load_checkpoint_weights(inference_model, model, checkpoint)
        val_accuracy = test(inference_model, test_data)
        epoch_nr = torch.load(checkpoint)["epoch"]
        log_data = {
            "validation_epoch": epoch_nr,
            "validation_iteration": epoch_nr * len(test_data),
            "validation_accuracy": val_accuracy,
        }
        utils.Logger.log_validate_results(log_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run validation on a given checkpoint')
    parser.add_argument('--checkpoint-input-path', help="Path of checkpoint file or folder", required=True)
    args = parser.parse_args()
    if os.path.isdir(args.checkpoint_input_path):
        checkpoint_files = [os.path.join(args.checkpoint_input_path, file_name) for file_name in os.listdir(args.checkpoint_input_path) if file_name.endswith(".pt")]
    else:
        checkpoint_files = [args.checkpoint_input_path]
    validate_checkpoints(checkpoint_files)
