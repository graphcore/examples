# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from .config import T5Config, Execution
import os
from pathlib import Path

CONFIG_DIR = Path(os.path.dirname(__file__))

del os, Path
