// cppimport
// NOTE: the cppimport comment is required for the `make compile` command
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
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

// -------------- cppimport --------------
// clang-format off
/*
<%
cfg['sources'] = ['attention_mask.cpp',
                  'disable_attn_dropout_bwd_pattern.cpp',
                  'tied_gather_pattern.cpp',
                  'tied_gather.cpp',
                  'utils.cpp',
                  'workarounds/prevent_const_expr_folding_op.cpp']
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare']
cfg['libraries'] = ['popart', 'poplar', 'popops']
%>
*/
