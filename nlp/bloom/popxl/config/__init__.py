# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os
from pathlib import Path

from .config import BloomConfig, Execution

CONFIG_DIR = Path(os.path.dirname(__file__))

del os, Path
