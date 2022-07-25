# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

# coding=utf-8


def is_nearest_multiple(m, a, b):
    """
    Replacement for m % b == 0 operator when variable is incremented by value other than one at each iteration.
    m is assumed to be increased by a at each iteration. If not exactly divisible, this function returns True the first
    iteration returns True the first iteration m surpasses b.
    """
    return (a > b) or (m - m // b * b < a)
