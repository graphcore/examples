# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import logging

import numpy as np
import popart

from .continuous_pipelining import ContinuousPipeliningOptimizerTransform
from .layer_scaling import LayerScalingOptimizerTransform

logger = logging.getLogger(__name__)


class BaseOptimizerFactory():
    def __init__(self, args, iteration, tensors=None):

        self.opt_type = args.optimizer

        if self.opt_type == "SGD":
            self.option_values = {
                "defaultLearningRate": args.learning_rate,
                "defaultMomentum": args.momentum,
                "defaultDampening": args.dampening or args.momentum,
                "defaultVelocityScaling": args.velocity_scaling,
                "lossScaling": args.loss_scaling,
            }
        elif "ADAM" in self.opt_type or "LAMB" in self.opt_type:
            self.option_values = {
                "defaultLearningRate": args.learning_rate,
                "defaultBeta1": args.beta1,
                "defaultBeta2": args.beta2,
                "lossScaling": args.loss_scaling,
                "maxWeightNorm": args.max_weight_norm if args.max_weight_norm is not None else np.finfo(np.float16).max
            }
        else:
            raise RuntimeError("Unknown opt_type in BaseOptimizerFactory")

        # Weight decay will be applied separately
        self.weight_decay = args.weight_decay

        self._options_created = False
        self._non_const_options = set()

        self.accl1_type = popart.DataType.FLOAT16 if args.use_half_optimizer_state else popart.DataType.FLOAT

        self.iteration = iteration

        self.tensors = tensors if tensors is not None else {}

        self.transforms = []

        if args.continuous_pipeline_optimizer_scaling and tensors is not None:
            self.transforms.append(
                ContinuousPipeliningOptimizerTransform(max(tensors.keys())))

        if args.squad_lr_scale != 1:
            self.transforms.append(
                LayerScalingOptimizerTransform(
                    name="Squad",
                    scale=args.squad_lr_scale))

    def _const_option(self, option):
        def default(name):
            return "default" + name[:1].capitalize() + name[1:]
        return not (option in self._non_const_options or default(option) in self._non_const_options)

    def _make_tuple_options(self, values):
        return {k: (v, self._const_option(k)) for k, v in values.items()}

    @property
    def optimizer_options(self):
        self._options_created = True
        # By default, options are const. They only become variable when they're scheduled in some way, at
        # which point their key should be appended to _non_const_options
        return self._make_tuple_options(self.option_values)

    @property
    def learning_rate(self):
        return self.option_values["defaultLearningRate"]

    def include_for_weight_decay(self, tensor_id):
        """ Do not include bias and norms for weight decay."""

        return self.weight_decay > 0 and not tensor_id.endswith(
            'B') and not tensor_id.endswith('Beta') and not tensor_id.endswith(
                'Gamma') and not tensor_id.endswith('Bias')

    def update_and_create(self, iteration):
        self.update(iteration)
        return self.create()

    def create(self):
        self.iteration.learning_rate = self.option_values["defaultLearningRate"]

        if self.opt_type == "SGD":
            optimizer = popart.SGD(self.optimizer_options)
        elif self.opt_type == "ADAM":
            optimizer = popart.Adam(self.optimizer_options,
                                    accl1_type=self.accl1_type,
                                    scaled_optimizer_state=self.accl1_type == popart.DataType.FLOAT16)
        elif self.opt_type == "ADAM_NO_BIAS":
            optimizer = popart.Adam(self.optimizer_options,
                                    mode=popart.AdamMode.AdamNoBias,
                                    accl1_type=self.accl1_type,
                                    scaled_optimizer_state=self.accl1_type == popart.DataType.FLOAT16)
        elif self.opt_type == "LAMB":
            optimizer = popart.Adam(self.optimizer_options,
                                    mode=popart.AdamMode.Lamb,
                                    accl1_type=self.accl1_type,
                                    scaled_optimizer_state=self.accl1_type == popart.DataType.FLOAT16)
        elif self.opt_type == "LAMB_NO_BIAS":
            optimizer = popart.Adam(self.optimizer_options,
                                    mode=popart.AdamMode.LambNoBias,
                                    accl1_type=self.accl1_type,
                                    scaled_optimizer_state=self.accl1_type == popart.DataType.FLOAT16)


        weight_decay_tensor_list = []

        for stage, tensors in self.tensors.items():
            for tensor_id in tensors:
                params = self.option_values.copy()

                if self.include_for_weight_decay(tensor_id):
                    params["weightDecay"] = self.weight_decay
                    weight_decay_tensor_list.append(tensor_id)
                else:
                    params["weightDecay"] = 0

                for transform in self.transforms:
                    params = transform(tensor_id, params, stage)

                specific_params = {
                    k: v for k, v in params.items() if k not in self.option_values
                }
                if specific_params:
                    p = self._make_tuple_options(specific_params)
                    optimizer.insertSpecific(tensor_id, p)

        if len(weight_decay_tensor_list) != 0:
            logger.debug(f" Weight decay of {self.weight_decay} applied to: {weight_decay_tensor_list}")

        return optimizer

    def should_update(self, iteration):
        raise NotImplementedError("This method should be overridden and not called directly")

    def update(self, iteration):
        raise NotImplementedError("This method should be overridden and not called directly")
