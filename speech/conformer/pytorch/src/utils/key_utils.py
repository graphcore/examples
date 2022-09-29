# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import string

class AishellKeyMapper:
    """
    Mapper for aishell keys of the form BAC009S0723W0395 to integers. This 
    is needed in order to be compatible with the poptorch dataloader that 
    requires 1D flat container of tensor (so list[str] is not allowed).

    The function enocde() removes characters [BAC00SW] from keys so that 
    an integer is left

    The function decode() adds these characters back to retrieve the original 
    key
    """

    @classmethod
    def encode(cls, input: str):
        _input = input[5:]
        return int("".join((k for k in _input if k not in string.ascii_uppercase)))

    @classmethod
    def decode(cls, input: int):
        _input = str(input)
        output = list("BAC00" + _input)
        output.insert(6, "S")
        output.insert(11, "W")
        return "".join(output)
