# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from datasets import load_dataset
from transformers import PerceiverFeatureExtractor
from torchvision.transforms import (
    Resize,
    Compose,
    ToTensor,
    CenterCrop,
    RandAugment,
    RandomResizedCrop,
    RandomHorizontalFlip,
)


from configs.hparams import SUPPORTED_DATASETS, DatasetArguments


class ApplyTransforms:
    """
    Functor that applies image transforms across a batch.
    """

    def __init__(self, transforms, dataset_name):
        self.transforms = transforms
        self.data_column_name = SUPPORTED_DATASETS[dataset_name]

    def __call__(self, example_batch):
        example_batch["pixel_values"] = [
            self.transforms(pil_img.convert("RGB")) for pil_img in example_batch[self.data_column_name]
        ]
        return example_batch


def get_dataset(mode: str, dataset_args: DatasetArguments, feature_extractor: PerceiverFeatureExtractor):
    """
    Returns a dataset with applied augmentations based on DatasetArguments.
    """

    assert mode in ["train", "eval", "test"]

    # load dataset
    if dataset_args.dataset_name == "cifar10":
        split = "train" if mode == "train" else "test"
    elif dataset_args.dataset_name == "imagenet-1k":
        split = "train" if mode == "train" else "validation"
    else:
        raise ValueError(f'Dataset "{dataset_args.dataset_name}" not supported.')

    # adjust how much of the dataset to use
    # (useful for debugging)
    split_proportion = dataset_args.train_split if mode == "train" else dataset_args.valid_split
    split = f"{split}[:{int(100*split_proportion)}%]"

    ds = load_dataset(data_dir=dataset_args.dataset_path, path=dataset_args.dataset_name, split=split)

    # define augmentations
    # training dataset
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            RandAugment(num_ops=4, magnitude=5),
            ToTensor(),
        ]
    )

    # test/validation datasets
    _test_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
        ]
    )

    # apply transforms
    transforms = _train_transforms if mode == "train" else _test_transforms
    ds.set_transform(ApplyTransforms(transforms, dataset_args.dataset_name))

    return ds
