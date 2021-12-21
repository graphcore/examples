# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import numpy as np


class StateManager:
    def __init__(self, seed=8888):
        current_state = np.random.get_state()
        np.random.seed(seed)
        self.state = np.random.get_state()
        np.random.set_state(current_state)

    def __call__(self, func):
        def new_func(*args, **kwargs):
            current_state = np.random.get_state()
            np.random.set_state(self.state)
            results = func(*args, **kwargs)
            self.state = np.random.get_state()
            np.random.set_state(current_state)
            return results

        return new_func
