# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Max Bain
# This file has been modified by Graphcore

import poptorch
import torch
from configs import options
from torchvision import transforms

from .MSRVTT_dataset import MSRVTT
from .WebVid_dataset import WebVid


def init_transform_dict(
    input_res=224,
    center_crop=256,
    randcrop_scale=(0.5, 1.0),
    color_jitter=(0, 0, 0),
    norm_mean=(0.485, 0.456, 0.406),
    norm_std=(0.229, 0.224, 0.225),
):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        "training": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
                normalize,
            ]
        ),
        "inference": transforms.Compose(
            [
                transforms.Resize(center_crop),
                transforms.CenterCrop(center_crop),
                transforms.Resize(input_res),
                normalize,
            ]
        ),
    }
    return tsfm_dict


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, text_length, num_frames, split):
        super().__init__()
        self._length = 500000
        if split == "inference":
            self._length = 1000
        self.video = []
        self.attention_mask = []
        torch.manual_seed(0)
        self.video = torch.rand((num_frames, 3, 224, 224), dtype=torch.float16)
        self.input_text = "It is a unit test for frozen"
        self.attention_mask = torch.randint(0, 1, (text_length,))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return {"text": self.input_text, "attention_mask": self.attention_mask, "video": self.video}


class TextVideoDataLoader(poptorch.DataLoader):
    def __init__(
        self,
        dataset_name: str,
        text_params: dict,
        video_params: dict,
        IPU_options: dict,
        data_dir: str,
        metadata_dir=None,
        split="training",
        tsfm_params=None,
        tsfm_split=None,
        subsample=1,
        sliding_window_stride=-1,
        cut=None,
        reader="decord",
        batch_size=1,
        num_workers=1,
        shuffle=True,
    ):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)

        if tsfm_split is None:
            tsfm_split = split
        tsfms = tsfm_dict[tsfm_split]

        kwargs = dict(
            dataset_name=dataset_name,
            text_params=text_params,
            video_params=video_params,
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            split=split,
            tsfms=tsfms,
            cut=cut,
            subsample=subsample,
            sliding_window_stride=sliding_window_stride,
            reader=reader,
        )

        if dataset_name == "MSRVTT":
            dataset = MSRVTT(**kwargs)
        elif dataset_name == "WebVid":
            dataset = WebVid(**kwargs)
        elif dataset_name == "synthetic":
            dataset = SyntheticDataset(text_params.get("max_length", 32), video_params.get("num_frames", 1), split)
        else:
            raise NotImplementedError(f"Dataset: {dataset_name} not found.")

        self.n_samples = len(dataset)
        ipu_opts = options.get_opts(options=IPU_options, modelType=split)

        super().__init__(
            ipu_opts,
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True,
            persistent_workers=True,
            auto_distributed_partitioning=not isinstance(dataset, torch.utils.data.IterableDataset),
            worker_init_fn=None,
            async_options={"load_indefinitely": True},
        )
        self.dataset_name = dataset_name
