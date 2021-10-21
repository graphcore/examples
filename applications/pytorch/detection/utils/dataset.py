# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import io
import re
import PIL
import yacs
import torch
import requests
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Any


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, cfg: yacs.config.CfgNode, transform: torchvision.transforms = None):
        self.mode = cfg.model.mode

        if self.mode == "train":
            dataset = cfg.dataset.train
        elif self.mode == "test":
            dataset = cfg.dataset.test

        self.input_channels = cfg.model.input_channels
        self.images_path = None
        self.labels_path = None
        self.img_data = None

        self.transform = transform

        self.maximum_synthetic_data = 10000

        if self.mode == "test_inference":
            url_sample_image = 'http://images.cocodataset.org/train2017/000000001072.jpg'
            img_data = requests.get(url_sample_image).content
            self.img_data = Image.open(io.BytesIO(img_data))
            height, width = self.img_data.size
            self.images_shapes = np.full((self.maximum_synthetic_data, 2), [height, width])
            self.labels = np.full((self.maximum_synthetic_data, 1, 5), [1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            path += cfg.dataset.name + "/" + dataset.file
            loaded = False
            if os.path.isfile((dataset.cache_path + '.npy')):
                images_path, labels_path = np.load(dataset.cache_path + '.npy')
                paths_pos = [m.start() for m in re.finditer(r"/", images_path[0])]
                if images_path[0][:paths_pos[-3]] == path[:path.rfind("/")]:
                    images_shapes = np.load(dataset.cache_path + '.shape.npy')
                    self.images_path, self.images_shapes = images_path.tolist(), images_shapes.tolist()
                    self.labels_path = labels_path.tolist()
                    loaded = True
            if not loaded:
                self.images_path, self.images_shapes = self.get_image_names_shapes(path)
                self.labels_path = self.get_labels_names()
                dir_pos = dataset.cache_path.rfind('/')
                if not os.path.isdir(dataset.cache_path[:dir_pos]):
                    os.mkdir(dataset.cache_path[:dir_pos])
                np.save(dataset.cache_path, (np.array(self.images_path), np.array(self.labels_path)))
                np.save(dataset.cache_path + '.shape', (np.array(self.images_shapes)))

            self.labels = self.get_labels()

            if dataset.cache_data:
                print("Not available")


    def get_image_names_shapes(self, path: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Verifies and returns the names and shapes of the images in a set path
        Parameters:
            path: a string containing the path to the images
        Return:
            images_path: a list of string with the paths to each individual image
            images_shapes: a list of int tuples with the shape information of each individual image
        """
        images_path = []
        if self.images_path is None:
            if os.path.isfile(path):
                files = []
                # If the reference to the files is local then we need the global direction
                if path.startswith('./'):
                    parent_global_location = os.path.abspath(path)[:len(path)]
                else:
                    end_last_word = path.rfind('/')
                    parent_global_location = os.path.abspath(path)[:end_last_word+1]
                with open(path, 'r') as lines:
                    lines = lines.read().splitlines()
                    files += [_file.replace('./', parent_global_location) if _file.startswith('./') else _file for _file in lines]
                images_path = sorted(files)
            else:
                raise Exception("File: " + path + " doesn't exist.")

            if len(images_path) == 0:
                raise Exception("No images found in: " + path)
        else:
            images_path = self.images_path

        return self.verify_images(images_path)


    def get_labels_names(self) -> List[str]:
        """
        Returns a list with each individual label path
        Return:
            labels_path: a list of string with the paths to each individual label
        """
        labels_path = []
        if self.labels_path is None:
            labels_path = [_file.replace('images', 'labels').replace(os.path.splitext(_file)[-1], '.txt') for _file in self.images_path]
            if not os.path.isfile(labels_path[0]):
                raise Exception("File: " + labels_path[0] + " doesn't exist.")
        else:
            labels_path = self.labels_path
        return labels_path


    def get_labels(self) -> List[np.array]:
        """
        Returns the values of the labels
        Return:
            labels: a list of np.array containing the values of all the labels per image
        """
        labels = []
        pbar = tqdm(self.labels_path, desc='Loading labels', total=len(self.labels_path))
        for label in pbar:
            try:
                label_value = []
                if os.path.isfile(label):
                    with open(label, 'r') as _file:
                        label_value = np.array([line.split() for line in _file.read().splitlines()], dtype=np.float32)  # labels
                if len(label_value) == 0:
                    label_value = np.zeros((0, 5), dtype=np.float32)
                labels.append(label_value)
            except Exception as e:
                labels.append(None)
                print('WARNING: %s' % (e))
        return labels

    def verify_images(self, images_path: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Verifies the images contained in the images_path location and returns their shapes and individual paths
        Parameters:
            path: a string containing the path to the images
        Return:
            images_path: a list of string with the paths to each individual image
            images_shapes: a list of int tuples with the shape information of each individual image
        """
        tmp_images_path = []
        tmp_images_shapes = []
        pbar = tqdm(images_path, desc='Verifying images', total=len(images_path))
        for image_path in pbar:
            try:
                image = Image.open(image_path)
                image.verify()
                height, width = image.size
                min_size = 10
                if height > min_size and width > min_size:
                    tmp_images_path.append(image_path)
                    tmp_images_shapes.append([height, width])
                else:
                    print("Warning!!! Invalid image, image smaller than " + str(min_size) + " pixels: " + image_path)
            except Exception as e:
                print("Warning!!! Invalid image: " + image_path + str(e))
        return tmp_images_path, tmp_images_shapes


    def __len__(self) -> int:
        """
        Returns the length of the dataset
        Return:
            length: length of the dataset
        """
        return len(self.images_path) if (self.mode != "test_inference") else self.maximum_synthetic_data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns the index image after applying the transformations specified
        during initialization
        Parameters:
            index: the position of the image in the dataset
        Return:
            image: transformed image in the index position
            labels: transformed labels in the index position
        """
        image = self.get_image(index)
        labels = self.labels[index]  # Format: [0] is the Class, [1, 2] is the Center x, y point and [3, 4] is the width and height of the box
        size = torch.as_tensor(image.size)

        if self.transform:
            transformed_image, transformed_labels = self.transform((image, labels))

        return transformed_image, transformed_labels, size, torch.as_tensor(index)

    def get_image(self, index: int) -> PIL.Image.Image:
        """
        Returns the index image of the dataset
        Parameters:
            index: the position of the image in the dataset
        Return:
            image: image in the index position
        """
        if self.mode == "test_inference":
            image = self.img_data
        else:
            image = Image.open(self.images_path[index])


        if self.input_channels == 3:
            return image.convert('RGB')
        elif self.input_channels == 1:
            return image.convert('LA')
        else:
            raise RuntimeError("Unsupported number of channels! Supported: [1, 3]")
