// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef  __TILE_MAPPING_COMMON_HPP__
#define  __TILE_MAPPING_COMMON_HPP__

#include <vector>
#include <tuple>

using SplitChannelInfo  = std::tuple<std::vector<size_t>, std::vector<size_t>, size_t>;
using SplitWorker1DInfo = std::pair<std::vector<int>, std::vector<int>>;
using SplitWorkerInfo   = std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>;
SplitChannelInfo  splitChannelByGroup(size_t   channel_cnt, 
                                      size_t   group_size, 
                                      size_t   num_tiles,
                                      size_t   tiles_per_ipu);

SplitWorker1DInfo split_worker_1dvector(int channel_cnt, 
                                       int inner_size, 
                                       int work_num);

SplitWorkerInfo  split_worker_vector(int channel_cnt, 
                                     int inner_size, 
                                     int work_num);

#endif