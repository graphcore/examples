# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
config_path="./config/config_544_phase1.json"
python scripts/coco_weights_to_voc.py
python scripts/fp32_to_fp16.py
python train.py --config $config_path
# start phase2
config_path="./config/config_544_phase2.json"
phase1_ckpt=$(ls -1 ./checkpoint|grep '.meta'|tail -n 1|sed 's/.meta//')
python train.py --config $config_path --initial-weight ./checkpoint/$phase1_ckpt

ckpt_paths=$(ls -1 ./checkpoint|grep '.meta'|tail -n 20|sed 's/.meta//')

for ckpt_path in $ckpt_paths;
do
	echo evaluating checkpoint: $ckpt_path
	python evaluate.py --config $config_path --weight-file "./checkpoint/$ckpt_path"
	pushd mAP
		echo $ckpt_path |tee -a mAP.log
		python main.py -na |tee -a  mAP.log 
	popd
done
