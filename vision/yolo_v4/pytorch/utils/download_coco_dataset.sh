# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


#!/bin/bash
# Download and unzip the images for test, validation and training
training_file='train2017.zip'
validation_file='val2017.zip'
test_file='test2017.zip'
url=http://images.cocodataset.org/zips/
directory='/localdata/datasets/coco/images' # unzip directory
for image_file in $training_file $validation_file $test_file; do
    curl -L $url$image_file -o $image_file && unzip -q $image_file -d $directory && rm $image_file
done
