# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import sys
from pathlib import Path

root_folder = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, root_folder)
sys.path.insert(0, root_folder.split("vision")[0] + "utils/triton_server/")
