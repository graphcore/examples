# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

from ogb.lsc import MAG240MDataset


class PreprocessMAG240Dataset(MAG240MDataset):


    def __init__(self, *args, pca_features_path=None):
        super().__init__(*args)
        self.pca_features_path = pca_features_path

    @property
    def all_pca_feat(self) -> np.ndarray:

        pca_features = np.load(self.pca_features_path)
        return pca_features

    @property
    def pca_feat(self) -> np.ndarray:
        pca_features = np.load(self.pca_features_path, mmap_mode='r')
        return pca_features
