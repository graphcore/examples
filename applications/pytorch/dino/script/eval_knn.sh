# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
weights_path=$1

train_features="python extract_feature.py 
                --data_path ../data/imagenet1k/train
                --weights $weights_path
                --replic 8
                --di 128
                --output train.pth"

val_features="python extract_feature.py 
                --data_path ../data/imagenet1k/validation
                --weights $weights_path
                --replic 8
                --di 128
                --output val.pth"

knn_acc="python knn_accuracy.py
         --train train.pth
         --validation val.pth"

info="You can get vit weigths with extract_feature.py \n
      comment like: \n
      python extract_weights.py --weights dino_model.pth --output vit.pt
      vit.pt will as a parameter for eval_knn.sh
      "

if [ ! -f "$weights_path" ]; then
    echo $info
else
    eval $train_features
    eval $val_features
    eval $knn_acc
fi
