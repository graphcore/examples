# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
#
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


from tensorflow.python import ipu
from tensorflow.python.ipu.config import StochasticRoundingBehaviour


def get_config(num_ipus=1, floating_point_behaviours=False, stochastic_rounding=True):
    config = ipu.config.IPUConfig()
    config.optimizations.merge_infeed_io_copies = True
    config.auto_select_ipus = num_ipus
    config.allow_recompute = True
    config.matmuls.poplar_options = {'partialsType': 'half', 'availableMemoryProportion': '0.2'}
    config.convolutions.poplar_options = {'partialsType': 'half', 'availableMemoryProportion': '0.2'}
    config.floating_point_behaviour.esr = StochasticRoundingBehaviour.from_bool(stochastic_rounding)
    if floating_point_behaviours:
        config.floating_point_behaviour.nanoo = True
        config.floating_point_behaviour.oflo = True
        config.floating_point_behaviour.inv = True
        config.floating_point_behaviour.div0 = True
    return config
