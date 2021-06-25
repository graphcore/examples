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

"""
This module implements a relative timer, this means that instead of computing
the time since the system epoch, this calculates the time from a more recent
start. This is useful since this timestamp can be represented with a 32 bit
float, rather than requiring a 64 bit float. The start of the timer is done at
the first import and is globally shared.
"""

import time

__start = time.time()


def now():
    return time.time() - __start


def get_start():
    return __start
