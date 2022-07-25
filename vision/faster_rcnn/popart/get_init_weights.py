# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import torch
import os

src_weights_path = 'weights/resnet50-caffe.pth'
dst_weights_path = 'weights/GC_init_weights.pth'

assert not os.path.exists(dst_weights_path)
weights = torch.load(src_weights_path)

new_weights = {'resnet.'+_key: _val for _key, _val in weights.items()}

torch.save(weights, dst_weights_path)
