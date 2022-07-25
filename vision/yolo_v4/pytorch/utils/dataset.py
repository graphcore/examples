# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import io
import re
import PIL
import yacs
import torch
import random
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Any
from torchvision.transforms import Compose
from utils.preprocessing import HSV, ToNumpy, Pad, ToTensor, ResizeImage, Mosaic, RandomPerspective, HorizontalFlip, VerticalFlip


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, cfg: yacs.config.CfgNode, mode: str):
        self.cfg = cfg
        self.dataset = None
        self.mode = mode

        if self.mode == "train":
            self.dataset = self.cfg.dataset.train
        elif self.mode == "test" or self.mode == "test_inference":
            self.dataset = self.cfg.dataset.test
        else:
            raise Exception("Mode not supported.")

        self.images_path = None
        self.labels_path = None
        self.img_data = None

        # Change the data type of the dataloader depeding of the options
        if self.cfg.model.uint_io:
            image_type = "uint"
        elif not self.cfg.model.ipu or not self.cfg.model.half:
            image_type = "float"
        else:
            image_type = "half"

        # Create transforms
        self.resize_image = ResizeImage(self.cfg.model.image_size)
        self.mosaic = Mosaic(self.cfg.model.image_size, self.cfg.model.input_channels)
        # We create two perspective transforms, one for the case with mosaic, and one without.
        # The difference is only in the border_to_remove parameter.
        self.random_perspective_mosaic = RandomPerspective(self.cfg.dataset.train.degrees,
                                                           self.cfg.dataset.train.translate,
                                                           self.cfg.dataset.train.scale,
                                                           self.cfg.dataset.train.shear,
                                                           self.cfg.dataset.train.perspective,
                                                           border_to_remove=[self.cfg.model.image_size, self.cfg.model.image_size])
        self.random_perspective_augment = RandomPerspective(self.cfg.dataset.train.degrees,
                                                            self.cfg.dataset.train.translate,
                                                            self.cfg.dataset.train.scale,
                                                            self.cfg.dataset.train.shear,
                                                            self.cfg.dataset.train.perspective,
                                                            border_to_remove=[0, 0])
        self.hsv_augment = HSV(self.cfg.dataset.train.hsv_h_gain,
                               self.cfg.dataset.train.hsv_s_gain,
                               self.cfg.dataset.train.hsv_v_gain)
        self.horizontal_flip = HorizontalFlip()
        self.vertical_flip = VerticalFlip()
        self.pad = Pad(self.cfg.model.image_size)
        self.image_to_tensor = Compose([ToNumpy(), ToTensor(int(self.cfg.dataset.max_bbox_per_scale), image_type)])

        self.maximum_synthetic_data = 10000

        if self.mode == "test_inference":
            url_sample_image = 'http://images.cocodataset.org/train2017/000000001072.jpg'
            img_data = requests.get(url_sample_image).content
            self.img_data = Image.open(io.BytesIO(img_data))
            height, width = self.img_data.size
            self.images_shapes = np.full((self.maximum_synthetic_data, 2), [height, width])
            self.labels = np.full((self.maximum_synthetic_data, 1, 5), [1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            path += self.cfg.dataset.name + "/" + self.dataset.file
            loaded = False
            if os.path.isfile((self.dataset.cache_path + '.id.npy')):
                images_path, labels_path = np.load(self.dataset.cache_path + '.npy')
                images_id = np.load(self.dataset.cache_path + '.id.npy')
                paths_pos = [m.start() for m in re.finditer(r"/", images_path[0])]
                if images_path[0][:paths_pos[-3]] == path[:path.rfind("/")]:
                    images_shapes = np.load(self.dataset.cache_path + '.shape.npy')
                    self.images_path, self.images_shapes, self.images_id = images_path.tolist(), images_shapes.tolist(), images_id.tolist()
                    self.labels_path = labels_path.tolist()
                    loaded = True
            if not loaded:
                self.images_path, self.images_shapes, self.images_id = self.get_image_names_shapes(path)
                self.labels_path = self.get_labels_names()
                dir_pos = self.dataset.cache_path.rfind('/')
                if not os.path.isdir(self.dataset.cache_path[:dir_pos]):
                    os.mkdir(self.dataset.cache_path[:dir_pos])
                np.save(self.dataset.cache_path, (np.array(self.images_path), np.array(self.labels_path)))
                np.save(self.dataset.cache_path + '.id', (np.array(self.images_id)))
                np.save(self.dataset.cache_path + '.shape', (np.array(self.images_shapes)))

            self.labels = self.get_labels()

            if self.dataset.cache_data:
                print("Not available")

        if not self.dataset.data_aug:
            if self.cfg.dataset.mosaic:
                print("Warning: mosaic augmentation won\'t be applied to a dataset with disabled data_aug.")
            if self.cfg.dataset.color:
                print("Warning: color augmentation won\'t be applied to a dataset with disabled data_aug.")

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
                # When in training mode shift label from 0-79 to 1-80 because loss uses 0 to pad
                if self.mode == 'train':
                    label_value[:, 0] = label_value[:, 0] + 1
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
        tmp_images_id = []
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
                    tmp_images_id.append(int(Path(image_path).stem))
                else:
                    print("Warning!!! Invalid image, image smaller than " + str(min_size) + " pixels: " + image_path)
            except Exception as e:
                print("Warning!!! Invalid image: " + image_path + str(e))
        return tmp_images_path, tmp_images_shapes, tmp_images_id


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
        labels = self.labels[index].copy()  # Format: [0] is the Class, [1, 2] is the Center x, y point and [3, 4] is the width and height of the box
        size = torch.as_tensor(image.size)

        transformed_image, transformed_labels = self.resize_image((image, labels))

        if self.cfg.dataset.mosaic and self.dataset.data_aug:
            # sample other 3 images to stitch with the current image
            indices = np.random.randint(0, self.__len__(), 3)
            mosaic_candidates = [
                self.resize_image((self.get_image(i), self.labels[i].copy())) for i in indices
            ]
            mosaic_candidates = [(transformed_image, transformed_labels)] + mosaic_candidates
            transformed_image, transformed_labels = self.mosaic(tuple(mosaic_candidates))
            transformed_image, transformed_labels = self.random_perspective_mosaic((transformed_image, transformed_labels))

        else:
            # Pad if we don't do the mosaic
            transformed_image, transformed_labels = self.pad((transformed_image, transformed_labels))

        if self.dataset.data_aug:
            if not self.cfg.dataset.mosaic:
                transformed_image, transformed_labels = self.random_perspective_augment((transformed_image, transformed_labels))

            # HSV color augmentation
            if self.cfg.dataset.color:
                transformed_image, transformed_labels = self.hsv_augment((transformed_image, transformed_labels))

            # Vertical Flip
            if random.random() < self.cfg.dataset.train.flipud:
                transformed_image, transformed_labels = self.vertical_flip((transformed_image, transformed_labels))

            # Horizontal Flip
            if random.random() < self.cfg.dataset.train.fliplr:
                transformed_image, transformed_labels = self.horizontal_flip((transformed_image, transformed_labels))

        transformed_image, transformed_labels = self.image_to_tensor((transformed_image, transformed_labels))

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


        if self.cfg.model.input_channels == 3:
            return image.convert('RGB')
        elif self.cfg.model.input_channels == 1:
            return image.convert('LA')
        else:
            raise RuntimeError("Unsupported number of channels! Supported: [1, 3]")
