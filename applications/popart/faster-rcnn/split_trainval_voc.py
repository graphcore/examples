# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
import os
import shutil
import sys

voc_path = sys.argv[1]
voc_annolist_txt = os.path.join(voc_path, 'ImageSets/Main/trainval.txt')

with open(voc_annolist_txt) as f:
    lines = f.readlines()

file_names = [line.replace('\n', '.xml') for line in lines]

if os.path.exists(os.path.join(voc_path, 'Annotations_trainval')):
    raise RuntimeError(
        'The directory to be created already exists, check whether the validation set and training set have been merged')

os.mkdir(os.path.join(voc_path, 'Annotations_trainval'))

for file_name in file_names:
    src_path = os.path.join(voc_path, 'Annotations', file_name)
    dst_path = os.path.join(voc_path, 'Annotations_trainval', file_name)

    shutil.copy(src_path, dst_path)
