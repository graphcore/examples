# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import sys

if __name__ == '__main__':
    args_to_pytest = sys.argv[1:]
    pytest.main(args_to_pytest)
