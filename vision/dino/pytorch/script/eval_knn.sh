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
arch=$1
# backbone arch in this weight checkpoint
weights_path=$2
# checkpoint path
knn_dir=./knn_accuracy
mkdir $knn_dir

python script/extract_weights.py \
	--arch $arch \
       	--weights $weights_path \
        --output $knn_dir/vit_model.pth

train_features="python script/extract_feature.py 
                --arch $arch
                --data_path ./data/imagenet1k/train
                --weights $knn_dir/vit_model.pth
                --replic 8
                --di 128
                --output $knn_dir/train.pth"

val_features="python script/extract_feature.py 
                --arch $arch
                --data_path ./data/imagenet1k/validation
                --weights $knn_dir/vit_model.pth
                --replic 8
                --di 128
                --output $knn_dir/val.pth"

knn_acc="python script/knn_accuracy.py
         --train $knn_dir/train.pth
         --validation $knn_dir/val.pth"

info="$2 not exists."

if [ ! -f "$weights_path" ]; then
    echo $info
else
    eval $train_features
    eval $val_features
    eval $knn_acc
fi
