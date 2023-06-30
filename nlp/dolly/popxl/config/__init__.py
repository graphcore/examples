# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from .config import DollyConfig, Execution
import os
from pathlib import Path

CONFIG_DIR = Path(os.path.dirname(__file__))

del os, Path
