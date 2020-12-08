# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""Custom layer functionality"""


def scale_1p2(x):
    """Simple scaling function for Lambda layers.

    Just multiplies the input by 1.2. Useful for extending the coverage of a
    tanh activation for targets in the range [-1,1].
    """
    return x*1.2
