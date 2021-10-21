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

import logging
import json
import os

logger = logging.getLogger(__name__)


def set_poplar_engine_options(execution_profile, memory_profile, profile_dir, sync_replicas_independently, synthetic_data, tensorflow_progress_bar):
    # load the user defined json object
    if os.environ.get('POPLAR_ENGINE_OPTIONS'):
        poplar_engine_options = json.loads(os.environ.get('POPLAR_ENGINE_OPTIONS').encode())
    else:
        poplar_engine_options = {}

    if execution_profile or memory_profile:
        report_dict = {
            "autoReport.outputExecutionProfile": "true" if execution_profile else "false",
            "debug.instrument": "true" if execution_profile else "false",
            "debug.allowOutOfMemory": "true",
            "autoReport.outputSerializedGraph": "false",
            "debug.outputAllSymbols": "true",
            "autoReport.all": "true",
            "profiler.format": "v3",
            "autoReport.directory": profile_dir
        }

        # We update the report_dict with the info from the user
        report_dict.update(poplar_engine_options)
        # We then update the poplar engine options
        poplar_engine_options = report_dict

    if sync_replicas_independently:
        independent_sync_dict = {
            "target.syncReplicasIndependently": "true"
        }
        independent_sync_dict.update(poplar_engine_options)
        poplar_engine_options = independent_sync_dict

    if synthetic_data:
        synthetic_data_flag = " --use_synthetic_data --synthetic_data_initializer=random"
        if os.environ.get('TF_POPLAR_FLAGS'):
            os.environ['TF_POPLAR_FLAGS'] += synthetic_data_flag
        else:
            os.environ['TF_POPLAR_FLAGS'] = synthetic_data_flag

    if tensorflow_progress_bar and not tensorflow_progress_bar == 'false':
        progress_bar_string = " --show_progress_bar="+tensorflow_progress_bar
        if os.environ.get('TF_POPLAR_FLAGS'):
            os.environ['TF_POPLAR_FLAGS'] += progress_bar_string
        else:
            os.environ['TF_POPLAR_FLAGS'] = progress_bar_string

    if len(poplar_engine_options):
        # We then set the poplar engine options to be the ones we defined here
        os.environ['POPLAR_ENGINE_OPTIONS'] = json.dumps(poplar_engine_options)
