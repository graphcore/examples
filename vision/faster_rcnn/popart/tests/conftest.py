# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di

import pytest
import os
import subprocess


def pytest_sessionstart(session):
    faster_rcnn_working_dic = os.path.join(os.path.dirname(__file__), '../')
