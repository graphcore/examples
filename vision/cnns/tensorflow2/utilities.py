# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
from typing import List


def get_closest_divisor(a: int, b: int) -> int:
    i = 1
    min_d = math.inf
    c = 1
    while i * i < a:
        if (a % i == 0) and abs(i - b) < min_d:
            c = i
            min_d = abs(i - b)
        if (a % (a // i) == 0) and abs(a // i - b) < min_d:
            c = a // i
            min_d = abs(a // i - b)
        i += 1
    return c


def verify_params_present(
    params: List[str], expected_params: List[str], object_name: str, arg_name: str, all_or_any: str = "all"
):

    fn = all if all_or_any == "all" else any
    if not fn(param in params for param in expected_params):
        if all_or_any == "all":
            raise KeyError(
                f"Not all required params for {object_name} were specified in {arg_name}. "
                f"Found {params}, expected {expected_params}."
            )
        else:
            raise KeyError(
                f"None of the expected params for {object_name} were specified in {arg_name}. "
                f"Found {params}, expected one from {expected_params}."
            )
