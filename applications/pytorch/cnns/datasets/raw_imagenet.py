# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import csv
import logging
import os
import torchvision


class ImageNetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, bbox_file=None, **kwargs):
        super(ImageNetDataset, self).__init__(*args, **kwargs)
        # assign bbox to the samples
        if bbox_file is not None:
            bboxes = self.load_bboxes(bbox_file)
        else:
            bboxes = {}
        for idx, (path, target) in enumerate(self.samples):
            self.samples[idx] = path, target, bboxes.get(target, None)


    def __getitem__(self, index: int):
        path, target, bbox = self.samples[index]
        with open(path, 'rb') as jpeg_file:
            img = jpeg_file.read()

        if self.transform is not None:
            sample = self.transform((img, bbox))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def load_bboxes(self, file_path):
        bboxes = {}
        if os.path.exists(file_path):
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    file_name = row[0]
                    x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    bboxes[file_name] = (x1, y1, x2, y2)
            return bboxes
        else:
            logging.warning("Bounding Box information hasn't found.")
