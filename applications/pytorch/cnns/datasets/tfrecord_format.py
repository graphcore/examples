# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import glob
import torch
from tfrecord.reader import tfrecord_loader
from PIL import Image
from io import BytesIO
import numpy as np
import multiprocessing
import sys
sys.path.append('..')
from datasets.webdataset_format import DistributeNode


file_pattern = {"train": "train-*", "validation": "validation-*"}
TFRECORD_KEYS = {"image/encoded": "byte", "image/class/label": "int", "image/colorspace": "byte", "image/format": "byte",
                 "image/object/bbox/xmin": "float", "image/object/bbox/ymin": "float",
                 "image/object/bbox/xmax": "float", "image/object/bbox/ymax": "float"}


class DecodeImage:
    def __init__(self, transform, use_bbox_info=False):
        self.transform = transform
        self.use_bbox_info = use_bbox_info

    def __call__(self, features):
        # Validate the image format
        colorspace = ''.join([chr(ch) for ch in features["image/colorspace"]])
        assert colorspace == "RGB"
        img_format = ''.join([chr(ch) for ch in features["image/format"]])
        assert img_format == "JPEG"
        file_jpgdata = BytesIO(features["image/encoded"])
        img = Image.open(file_jpgdata).convert("RGB")
        label = features["image/class/label"][0].tolist()
        # determine bboxes
        if self.use_bbox_info and len(features["image/object/bbox/xmin"]) > 0:
            x1 = np.min(features["image/object/bbox/xmin"])
            y1 = np.min(features["image/object/bbox/ymin"])
            x2 = np.max(features["image/object/bbox/xmax"])
            y2 = np.max(features["image/object/bbox/ymax"])
            if self.transform:
                img = self.transform((img, (x1, y1, x2, y2)))
        else:
            if self.transform:
                img = self.transform(img)
        return img, label


def get_tfrecord(opts, model_opts, train=True, transform=None, use_bbox_info=False):
    assert model_opts.Distributed.numProcesses == 1, "PopRun i not supported with TFRecord"
    dataset_pattern = file_pattern["train"] if train else file_pattern["validation"]
    chunks = glob.glob(os.path.join(opts.imagenet_data_path, dataset_pattern))
    file_shuffle = DistributeNode(None, model_opts.Distributed.processId, model_opts.Distributed.numProcesses, seed=opts.seed)
    buffer_size = 5 if train else 1
    dataset = TFRecordDataset(chunks, file_buffer_size=buffer_size, shuffle=train, transform=DecodeImage(transform, use_bbox_info=use_bbox_info and train), nodesplitter=file_shuffle)
    return dataset


class TFRecordDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 input_files,
                 file_buffer_size=5,
                 shuffle=True,
                 transform = None,
                 nodesplitter = None):
        self.files = input_files
        self.file_buffer_size = file_buffer_size
        self.shuffle = shuffle
        self.transform = transform
        self.nodesplitter = nodesplitter

    def reset(self):
        self.file_index = self.shard[0] * self.file_buffer_size
        self.data_index = 0

    def samples_per_file(self, filename):
        reader = tfrecord_loader(filename, None, list(TFRECORD_KEYS))
        count = 0
        for _ in reader:
            count += 1
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
        self.shard = worker_info.id, worker_info.num_workers
        self.reset()
        self.worker_id = worker_info.id
        if self.shuffle and self.nodesplitter:
            self.files = self.nodesplitter(self.files)
        self.load_data()
        return self


    def __next__(self):
        if self.data_index >= len(self.data):
            self.load_data()
        data = self.data[self.data_index]
        self.data_index += 1
        data = self.transform(data)
        return data


    def load_data(self):
        # This drops the remainder
        if self.file_index >= len(self.files):
            raise StopIteration
        self.data = []
        # Load multiple files into the data buffer at a time
        for _ in range(self.file_buffer_size):
            self.data += self.load_file()
            self.file_index += 1
            if self.file_index >= len(self.files):
                break
        self.file_index += (self.shard[-1] - 1) * self.file_buffer_size
        if self.shuffle:
            np.random.shuffle(self.data)
        self.data_index = 0

    def load_file(self):
        reader = tfrecord_loader(self.files[self.file_index],
                                 None,
                                 list(TFRECORD_KEYS))
        data = []
        for datum in reader:
            data.append(datum)
        return data
