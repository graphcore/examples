// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <iostream>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

poplar::Tensor crop(poplar::Graph &graph, poplar::program::Sequence &prog,
                    const poplar::Tensor &input, float scale_factor,
                    const poplar::DebugContext &di = {});

poplar::Tensor crop_grads(poplar::Graph &graph, poplar::program::Sequence &prog,
                          const poplar::Tensor &grad_output, float scale_factor,
                          const poplar::DebugContext &di = {});
