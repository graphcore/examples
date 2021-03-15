# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import webdataset as wds
import torch
import json
import os
from torchvision import transforms


def identity(x):
    return x


def get_webdataset(opts, model_opts, train=True, half_precision=True, transform=identity, shuffle_buffer=5000):
    subset_name = 'train' if train else 'validation'
    with open(os.path.join(opts.imagenet_data_path, "metadata.json")) as metadata_file:
        metadata = json.load(metadata_file)
        dataset_size = metadata[f"{subset_name}_length"]
        data_format = metadata["format"]
        done_preprocess = metadata["transform_pipeline"]

    batch_size = opts.batch_size * opts.device_iterations * opts.replicas
    if train and hasattr(opts, "gradient_accumulation"):
        batch_size *= opts.gradient_accumulation

    chunks = [file_name[-10:-4] for file_name in os.listdir(opts.imagenet_data_path) if file_name.startswith(subset_name)]
    chunks.sort()  # sort the chunks so that they can be distributed properly
    chunk_str = distribute_chunks(chunks, model_opts.Distributed.processId, model_opts.Distributed.numProcesses)
    dataset = wds.Dataset(os.path.join(opts.imagenet_data_path, subset_name + "-{" + chunk_str + "}.tar"), length=dataset_size // (batch_size * model_opts.Distributed.numProcesses)).shuffle(shuffle_buffer)
    # Remove transformations, which are already done during preprocess.
    transform = match_preprocess(transform, done_preprocess)

    if data_format == "img":
        dataset = dataset.decode("pil").to_tuple("jpg;png", "cls").map_tuple(transform, identity)
    elif data_format == "tensor":
        dataset = dataset.decode("torch").to_tuple("pth", "cls").map_tuple(transform, identity)
    else:
        raise Exception(f"Data format {data_format} is not supported.")
    # Creating the batches
    dataset.pipeline.append(wds.filters.batched(batchsize=batch_size, partial=True,))
    return dataset


def distribute_chunks(chunks, instance_id, total_instance):
    start = (len(chunks) // total_instance) * instance_id
    end = (len(chunks) // total_instance) * (instance_id + 1) - 1
    # Handle uneven chunks
    if len(chunks) % total_instance > instance_id:
        start += instance_id
        end += instance_id
    return chunks[start] + ".." + chunks[end]


def match_preprocess(preprocess_pipeline, done_preprocess=[]):
    # Remove initial resizing
    if len(done_preprocess) > 0 and done_preprocess[0].startswith("Resize(size=256"):
        preprocess_pipeline.transforms.pop(0)
    return preprocess_pipeline


class DatasetRebatch:
    """
    Wrapper for DataLoader to hide multiple non-complete batches and combine them to full batches
    """
    def __init__(self, dataloader, batch_size):
        self.dataloader = dataloader
        self.batch_size = batch_size


    def __iter__(self):
        self.remaining = None
        self.iterable_dataloader = iter(self.dataloader)
        return self


    def __len__(self):
        return len(self.dataloader)


    def __next__(self):
        while True:
            tensor = next(self.iterable_dataloader)
            if tensor[0].size()[0] == self.batch_size:
                return tensor
            else:
                if self.remaining is None:
                    self.remaining = tensor
                else:
                    self.remaining = [torch.cat([buffer, current], dim=0) for buffer, current in zip(self.remaining, tensor)]
                    if self.remaining[0].size()[0] >= self.batch_size:
                        returning_tensor = [buffer[:self.batch_size] for buffer in self.remaining]
                        self.remaining = [buffer[self.batch_size:] for buffer in self.remaining]
                        return returning_tensor
