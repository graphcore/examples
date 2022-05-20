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

# NO.1 compare grad
extract_name="python alignment.py 
                --pipeline 3 0 
                --device ipu 
                --extract_name
                --arch vit_mini"

grad_ipu="python alignment.py 
                --pipeline 3 0 
                --device ipu 
                --grad_compare
                --arch vit_mini"
grad_cpu="python alignment.py 
                --device cpu 
                --grad_compare
                --arch vit_mini"

# NO.2 alignment shard
shard_ipu="python alignment.py 
                --pipeline 3 0
                --device ipu
                --arch vit_mini"

shard_cpu="python alignment.py 
                --device cpu
                --arch vit_mini"

# NO.3 alignment pipeline 
pipeline_ipu="python alignment.py 
                --pipeline 1 2 2 2 2 2 1 0 
                --device ipu 
                --alignment_pipeline 
                --ga 64
                --arch vit_base"

pipeline_cpu="python alignment.py 
                --device cpu 
                --alignment_pipeline
                --ga 64
                --arch vit_base"

# compare grad and weights use little model which contains all the ops
eval $extract_name
eval $grad_ipu
eval $grad_cpu


echo "You can compare grad with grad_compare.py"