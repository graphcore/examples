# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import csv
import logging
import os
import torchvision


class ImageNetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        if "bbox_file" in kwargs.keys() and (not kwargs["bbox_file"] is None):
            bbox_file = kwargs["bbox_file"]
            self.use_bbox = True

        else:
            self.use_bbox = False
        if "bbox_file" in kwargs.keys():
            del kwargs["bbox_file"]
        super(ImageNetDataset, self).__init__(*args, **kwargs)
        if self.use_bbox:
            self.load_bboxes(bbox_file)


    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.use_bbox:
            _, bbox_key = os.path.split(path)
            if bbox_key in self.bboxes.keys():
                sample = (sample, self.bboxes[bbox_key])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def load_bboxes(self, file_path):
        self.bboxes = {}
        if os.path.exists(file_path):
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    file_name = row[0]
                    x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    self.bboxes[file_name] = (x1, y1, x2, y2)
        else:
            logging.warning("Bounding Box information hasn't found.")
