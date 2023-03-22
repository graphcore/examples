# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 mlfoundations

# This file has been modified by Graphcore

import os
from collections import OrderedDict

import poptorch
import torch
from args import parse_args
from datasets import get_transforms, tokenize
from model import CLIP
from PIL import Image, ImageFile
from torchvision.datasets import CIFAR100
from tqdm import tqdm


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


ImageFile.LOAD_TRUNCATED_IMAGES = True


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, image_names, labels, transforms):
        super().__init__()
        self.images = image_names
        self.labels = labels
        self.transforms = transforms
        self.image_path = image_path

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, self.images[idx]))
        image = self.transforms(image)

        return image, self.labels[idx]

    def __len__(self):
        return len(self.labels)


def get_val_opts():
    # Config IPU
    opts = poptorch.Options()
    opts.deviceIterations(1)
    opts.replicationFactor(1)
    opts.Training.gradientAccumulation(1)
    opts.setAvailableMemoryProportion({"IPU0": 0.1})
    opts.outputMode(poptorch.OutputMode.Final)
    opts.TensorLocations.setOptimizerLocation(poptorch.TensorLocationSettings().useOnChipStorage(False))
    opts.randomSeed(42)
    opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))

    return opts


def get_val_dataloader(config):
    templates_dict = torch.load("datasets/text_templates.pt")

    if config.zeroshot_dataset == "imagenet":
        images = []
        labels = []
        imagenet_file = "data/imagenet1k/validation/val_official_clean.csv"
        with open(imagenet_file, "r", encoding="utf-8") as f:
            all_lines = f.read().splitlines()
            for line in tqdm(all_lines):
                img_name, label = line.split("\t")
                images.append(img_name)
                labels.append(int(label))

        dataset = ImagenetDataset(os.path.dirname(imagenet_file), images, labels, preprocess)
        classnames = templates_dict["imagenet_classnames"]
        templates = templates_dict["imagenet_templates"]
    else:
        # Download the dataset
        dataset = CIFAR100(root=os.path.expanduser("data/cifar100"), download=True, train=False, transform=preprocess)
        classnames = templates_dict["cifar100_classnames"]
        templates = templates_dict["cifar100_templates"]

    async_dataloader = True
    dataset_mode = poptorch.DataLoaderMode.AsyncRebatched if async_dataloader else poptorch.DataLoaderMode.Sync
    dataloader = poptorch.DataLoader(
        opts,
        dataset,
        batch_size=10,
        num_workers=4,
        shuffle=False,
        drop_last=True,
        persistent_workers=True,
        auto_distributed_partitioning=True,
        worker_init_fn=None,
        mode=dataset_mode,
        async_options={"load_indefinitely": True, "buffer_size": 8},
    )

    return dataloader, classnames, templates


def encode_text(text_inference, classnames, templates):
    text_inference.half()
    zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = [template.replace("{c}", classname) for template in templates]
        texts = tokenize(texts)
        class_embedding = text_inference(images=torch.zeros_like(torch.randn(1, 3, 224, 224)), texts=texts)

        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

    return zeroshot_weights


if __name__ == "__main__":
    opts = get_val_opts()

    # Config model
    config = parse_args()

    model = CLIP(config).eval()
    inference_model = poptorch.inferenceModel(model, options=opts)
    text_inference = poptorch.inferenceModel(model, options=opts)

    preprocess = get_transforms(is_train=False)

    if config.is_ipu_ckpt:
        inf_dict = torch.load(config.ckpt_file, map_location="cpu")

        inference_model.load_state_dict(inf_dict["model_state_dict"])
        text_inference.load_state_dict(inf_dict["model_state_dict"])
    else:
        # Reload checkpoint
        state_dict = torch.jit.load(config.ckpt_file, map_location="cpu")
        new_state_dict = OrderedDict()

        for k, v in state_dict.state_dict().items():
            if k in ["input_resolution", "context_length", "vocab_size"]:
                continue

            new_state_dict[k] = v
        new_state_dict["image_fea_queue"] = model.state_dict()["image_fea_queue"]
        new_state_dict["text_fea_queue"] = model.state_dict()["text_fea_queue"]

        inference_model.load_state_dict(new_state_dict)
        text_inference.load_state_dict(new_state_dict)

    dataloader, classnames, templates = get_val_dataloader(config)

    zeroshot_weights = encode_text(text_inference, classnames, templates)

    with torch.no_grad():
        n = 0
        top1, top5 = 0.0, 0.0
        for images, target in tqdm(dataloader):
            # Predict
            logits_per_image = inference_model(images, zeroshot_weights).type(torch.float32)

            # Measure accuracy
            acc1, acc5 = accuracy(logits_per_image, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    Recall_at_1 = top1 / n
    Recall_at_5 = top5 / n

    print(
        f"The sum of samples: {n}. \ntop1: {top1}, Recall_at_1: {Recall_at_1}. \ntop5: {top5}, Recall_at_5: {Recall_at_5}"
    )
