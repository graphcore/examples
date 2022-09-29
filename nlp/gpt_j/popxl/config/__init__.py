# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from .config import GPTJConfig, Execution
import os
from pathlib import Path

CONFIG_DIR = Path(os.path.dirname(__file__))

del os, Path
