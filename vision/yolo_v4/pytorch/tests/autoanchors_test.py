# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest

import torch
import numpy as np

from utils.config import get_cfg_defaults
from utils.dataset import Dataset
from utils.tools import AutoAnchors


@pytest.fixture(name="data_manager", scope="class")
def data_manager_fixture():
    """
    Manages the auto_anchor object creation, and the data
    required to set up the object. Also creates the data that
    the different functions need.
    """

    class DataManager:
        def __init__(self):
            self.gen = 1000
            self.cfg = get_cfg_defaults()
            mode = "test_inference"
            self.dataset = Dataset(None, self.cfg, mode)
            self.auto_anchors = AutoAnchors(self.dataset, self.cfg.model, self.gen)
            self.k_points = torch.ones((12, 2)) * 2.0
            self.wh = torch.ones((1000, 2)) * 2.0

    return DataManager()


class TestAutoAnchors:
    """
    Test the auto anchor computation functions
    """

    def test_best_ratio_metric(self, data_manager):
        best_ratio_in_dim, best = data_manager.auto_anchors.best_ratio_metric(data_manager.k_points, data_manager.wh)
        assert (best_ratio_in_dim == 1).all()
        assert (best == 1).all()

    def test_metric(self, data_manager):
        bpr, aat = data_manager.auto_anchors.metric(data_manager.k_points, data_manager.wh)
        assert bpr == 1.0
        assert aat == 12.0

    def test_fitness(self, data_manager):
        fitness = data_manager.auto_anchors.fitness(data_manager.k_points, data_manager.wh)
        assert fitness == 1.0
