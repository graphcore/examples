# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import argparse
import torch
from tqdm import tqdm
import poptorch
import logging
import sys
sys.path.append('..')
import models
import utils
import datasets
import popdist


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


def test(inference_model, test_data, opts):
    nr_batches = len(test_data)
    bar = tqdm(test_data, total=nr_batches)
    sum_acc = 0.0
    sample_count = 0
    inference_model = DataPadder(test_data.combinedBatchSize, inference_model)
    with torch.no_grad():
        for idx, (input_data, labels) in enumerate(bar):
            sample_count += labels.size()[0]
            output = inference_model(input_data)
            output = output.float()
            sum_acc += utils.accuracy(output, labels) * labels.size()[0]
            aggregated_accuracy = sum_acc / sample_count
            bar.set_description(f"Accuracy:{aggregated_accuracy:0.2f}%")
    acc = sum_acc / sample_count
    logging.info(f"Accuracy on test set: {acc:0.2f}%")
    return acc


def load_checkpoint_weights(inference_model, file_path):
    checkpoint = torch.load(file_path)
    opts = checkpoint['opts']
    logging.info(f"Restore the {opts.model} model to epoch {checkpoint['epoch']} on {opts.data} dataset(Train loss:{checkpoint['loss']}, train accuracy:{checkpoint['train_accuracy']}%)")
    inference_model.model.load_state_dict(checkpoint['model_state_dict'])
    inference_model.copyWeightsToDevice()


def create_validation_opts(opts):
    if opts.use_popdist:
        model_opts = popdist.poptorch.Options(ipus_per_replica=len(opts.pipeline_splits) + 1)
    else:
        model_opts = poptorch.Options()
        model_opts.replicationFactor(opts.replicas)
    model_opts.deviceIterations(max(opts.device_iterations, 1+len(opts.pipeline_splits)))
    model_opts.anchorMode(poptorch.AnchorMode.All)
    return model_opts


def validate_checkpoints(checkpoint_list, test_data=None):
    checkpoint = torch.load(checkpoint_list[0])
    opts = checkpoint['opts']
    # Initialise popdist
    utils.handle_distributed_settings(opts)
    utils.Logger.setup_logging_folder(opts)

    # make sure the order is ascending
    def ckpt_key(ckpt):
        return int(ckpt.split('_')[-1].split('.')[0])
    try:
        checkpoint_list = sorted(checkpoint_list, key=ckpt_key)
    except:
        logging.warn("Checkpoint names are changed, which may cause inconsistent order in evaluation.")

    model_opts = create_validation_opts(opts)

    if test_data is None:
        logging.info("Loading the data")
        test_data = datasets.get_data(opts, model_opts, train=False, async_dataloader=True, return_remaining=True)

    logging.info("Create model")
    model = models.get_model(opts, datasets.datasets_info[opts.data], pretrained=False)
    model.eval()
    # Load the weights of the first checkpoint for the model
    model.load_state_dict(checkpoint['model_state_dict'])
    inference_model = poptorch.inferenceModel(model, model_opts)

    for checkpoint in checkpoint_list:
        if inference_model.isCompiled():
            load_checkpoint_weights(inference_model, checkpoint)
        acc = test(inference_model, test_data, opts)
        epoch_nr = torch.load(checkpoint)["epoch"]
        result_dict = {"validation_epoch": epoch_nr,
                       "validation_iteration": opts.logs_per_epoch * epoch_nr,
                       "validation_accuracy": acc}
        utils.Logger.log_validate_results(result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run validation on a given checkpoint')
    parser.add_argument('--checkpoint-path', help="Path of checkpoint file or folder", required=True)
    args = parser.parse_args()
    if os.path.isdir(args.checkpoint_path):
        checkpoint_files = [os.path.join(args.checkpoint_path, file_name) for file_name in os.listdir(args.checkpoint_path) if file_name.endswith(".pt")]
    else:
        checkpoint_files = [args.checkpoint_path]
    validate_checkpoints(checkpoint_files)
