# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Copyright 1999-present Alibaba Group Holding Ltd.
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
#
# This file has been modified by Graphcore Ltd.
# It has been modified to disable running this module as a script.

import os
import sys
import random

import tempfile
from subprocess import call


def main(file, temporary=False):
    tf_os, tpath = tempfile.mkstemp(dir='~/DIN-V2-CODE')
    tf = open(tpath, 'w')

    fd = open(file, "r")
    for l in fd:
        print >> tf, l.strip("\n")
    tf.close()

    lines = open(tpath, 'r').readlines()
    random.shuffle(lines)
    if temporary:
        path, filename = os.path.split(os.path.realpath(file))
        fd = tempfile.TemporaryFile(prefix=filename + '.shuf', dir=path)
    else:
        fd = open(file + '.shuf', 'w')

    for l in lines:
        s = l.strip("\n")
        print >> fd, s

    if temporary:
        fd.seek(0)
    else:
        fd.close()

    os.remove(tpath)

    return fd
