# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import torch
import numpy as np
from nanodata.dataset import build_dataset
from nanodata.collate import collate_function
from utils import logger
from layer.anchor_target_layer_for_nanodata import AnchorTargetLayer


def get_data_loader(cfg):
    if cfg.TRAIN.PRESET_INDICES == '':
        local_preset_indices = None
        local_shuffle = cfg.NANO_DATA_CFG.DATA.SHUFFLE
    else:
        local_preset_indices = np.load(cfg.TRAIN.PRESET_INDICES)
        local_shuffle = False
        logger.log_str('using preset_indices:', cfg.TRAIN.PRESET_INDICES)

    # generate anchor target in cpu, put in dataset to speed up
    anchor_target_layer = AnchorTargetLayer(
        cfg.FEAT_STRIDE[0],
        cfg.ANCHOR_SCALES,
        cfg.ANCHOR_RATIOS,
        dtype=np.float32)

    cfg.NANO_DATA_CFG.DATA.TRAIN.input_size = cfg.INPUT_SIZE
    cfg.NANO_DATA_CFG.DATA.TRAIN.num_gtboxes = cfg.TRAIN.NUM_GT_BOXES
    specified_length = cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE * 2
    dataset_cfg = cfg.NANO_DATA_CFG.DATA.TRAIN
    dataset_cfg.img_path = os.path.join(cfg.DATA_DIR, dataset_cfg.img_path)
    dataset_cfg.ann_path = os.path.join(cfg.DATA_DIR, dataset_cfg.ann_path)
    train_dataset = build_dataset(dataset_cfg,
                                  "train",
                                  preset_indices=local_preset_indices,
                                  specified_length=specified_length,
                                  extra_layer=anchor_target_layer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=local_shuffle,
        num_workers=cfg.NANO_DATA_CFG.DATA.NUM_WORKERS,
        pin_memory=False,
        collate_fn=collate_function,
        drop_last=True,
    )

    return train_dataloader
