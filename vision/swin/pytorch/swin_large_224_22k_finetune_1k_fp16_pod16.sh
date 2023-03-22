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
python train_swin.py \
--cfg SWIN_LARGE_224_22K_FINETUNE_1K_FP16_POD16 \
--data-path /path/to/imagenet1k/ \
--output ./output/swin_large_224/ \
--precision 'half' \
--pretrained-model /path/to/swin_large_patch4_window7_224_22k.pth
