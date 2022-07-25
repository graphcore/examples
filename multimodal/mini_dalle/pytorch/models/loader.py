# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 lucidrains

# This file has been modified by Graphcore


from pathlib import Path
from random import randint, choice

import PIL
import torch
import poptorch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms as T
from models.tokenizer import SimpleTokenizer, YttmTokenizer


class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 bpe_path=None,
                 shuffle=False,
                 not_real_data=False,
                 image_dtype=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        self.bpe_path = bpe_path
        self.text_len = text_len
        self.not_real_data = not_real_data
        self.image_dtype = image_dtype
        if self.not_real_data:
            return
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose([
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        if self.not_real_data:
            return 120000
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        if self.not_real_data:
            tokenized_text = torch.randint(0, 4096, [self.text_len], dtype=torch.long)
            if self.image_dtype == torch.uint8:
                image_tensor = torch.randint(0, 255, [3, 256, 256], dtype=self.image_dtype)
            else:
                image_tensor = torch.rand([3, 256, 256], dtype=self.image_dtype)
            return tokenized_text, image_tensor

        if self.bpe_path is not None:
            klass = YttmTokenizer
            tokenizer = klass(self.bpe_path)
        else:
            tokenizer = SimpleTokenizer()

        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            img = PIL.Image.open(image_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_tensor = self.image_transform(img)
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        if self.image_dtype == torch.uint8:
            image_tensor = torch.clip((image_tensor*255).round_(), 0, 255).byte()
        elif self.image_dtype == torch.float16:
            image_tensor = image_tensor.half()

        # Success
        return tokenized_text, image_tensor


def get_data(configs, model_opts, image_size, train=True, async_dataloader=False):
    """
    A factory method to create a dataloader responsible for sending data
    to the IPU device. This build the appropriate dataset and wraps it in a dataloader.
    """
    if configs.byteio:
        image_dtype = torch.uint8
    elif configs.fp16:
        image_dtype = torch.float16
    else:
        image_dtype = torch.float32
    dataset = TextImageDataset(
        configs.input_folder,
        text_len=configs.text_seq_len,
        image_size=image_size,
        resize_ratio=1.0,
        truncate_captions=configs.truncate_captions,
        bpe_path=configs.bpe_path,
        shuffle=True,
        not_real_data=(configs.generated_data or configs.synthetic_data),
        image_dtype=image_dtype
    )

    assert len(dataset) > 0, 'dataset is empty'
    print(f'{len(dataset)} image-text pairs found for training')

    rebatched_worker_size = max(int(configs.gradient_accumulation/4), configs.batch_size)
    mode = poptorch.DataLoaderMode.AsyncRebatched if async_dataloader else poptorch.DataLoaderMode.Sync
    dataloader = poptorch.DataLoader(model_opts,
                                     dataset,
                                     batch_size=configs.batch_size if not(isinstance(
                                         dataset, IterableDataset)) else None,
                                     num_workers=configs.dataloader_workers,
                                     shuffle=train and not(isinstance(dataset, IterableDataset)),
                                     drop_last=not(isinstance(dataset, IterableDataset)),
                                     persistent_workers=True,
                                     auto_distributed_partitioning=not isinstance(
                                         dataset, IterableDataset),
                                     worker_init_fn=None,
                                     mode=mode,
                                     rebatched_worker_size=rebatched_worker_size,
                                     async_options={"early_preload": True, 'load_indefinitely': True})

    return dataloader
