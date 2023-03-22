#!/bin/sh
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
poprun -vv --host $your_host \
      --vipu-partition=pod128       \
      --num-instances=2 \
      --num-replicas=8 \
      --num-ilds=2 \
      --ipus-per-replica=8  \
      --update-partition=yes  \
      --mpi-global-args="--mca oob_tcp_if_include "10.5.0.0/16" --mca btl_tcp_if_include 10.5.0.0/16"   \
      --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x IPUOF_VIPU_API_TIMEOUT=3600 -x POPLAR_LOG_LEVEL=WARN -x POPLAR_SDK_ENABLED -x POPLAR_ENGINE_OPTIONS"       \
      --vipu-server-timeout=3600 \
python train_swin.py \
--cfg SWIN_LARGE_224_22K_FINETUNE_1K_FP16_POD64 \
--output output_pod128 \
--data-path /path/to/imagenet1k \
--pretrained-model /path/to/swin_large_patch4_window7_224_22k.pth
