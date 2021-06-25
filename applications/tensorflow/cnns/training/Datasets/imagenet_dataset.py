# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
from functools import partial
from Datasets.data import imagenet_preprocess
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


class ImageNetData:
    """Optimisation that splits the ImageNet dataset to allow better host CPU usage when
    a high processing throughput is required"""

    def __init__(self, opts, filenames):
        # defensive copy
        self.opts = opts.copy()
        self.n_files = len(filenames)
        self.n_cores = opts['pipeline_num_parallel']
        self.fnames = filenames
        self.n_splits = None
        self.split_fnames = None
        self.n_validation_images = 50000
        self.dataset_percentage_to_use = opts['dataset_percentage_to_use']


    def _get_subset(self, split_id, batch_size, is_training, datatype):
        """
        Splits the ImageNet dataset into subsets.
        inputs: split_id (Tensor(int)): the index into `self.split_fnames`.
        returns: the dataset indicated by a given `split_id`
        """
        # subdividing the filenames
        if self.n_splits > 1:
            predicates = [tf.equal(split_id, x) for x in range(self.n_splits)]
            # i=i is required to get lambdas with different return values
            lambdas = [(lambda i=i: self.split_fnames[i]) for i in range(self.n_splits)]
            subset = tf.case(list(zip(predicates, lambdas[:-1])), lambdas[-1], name='dataset_splitter')
        else:
            subset = self.fnames

        # Create records from the subset.
        dataset = tf.data.Dataset.from_tensor_slices(subset)

        # Interleave all the files belonging to this subset
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=self.n_files // self.n_splits,
                                     block_length=1, num_parallel_calls=self.n_files // self.n_splits)

        opts = self.opts

        if opts['train_with_valid_preprocessing'] and is_training:
            training_preprocessing = False
        else:
            training_preprocessing = is_training

        preprocess_fn = partial(imagenet_preprocess, is_training=training_preprocessing,
                                image_size=opts["image_size"],
                                dtype=datatype, seed=opts['seed'],
                                full_normalisation=opts['normalise_input'] if opts['hostside_norm'] else None)

        if not is_training:
            n_validation_useable = self.n_validation_images_useable(batch_size)
            n_to_take = n_validation_useable // self.n_splits
            dataset = dataset.take(n_to_take)
        if not opts['no_dataset_cache']:
            dataset = dataset.cache()

        if is_training:
            # Shuffles the records files within this (sub) dataset
            dataset = dataset.shuffle(self.n_files // self.n_splits)
            # in validation, we want to repeat the whole interleaved dataset and not each subset individually
            dataset = dataset.repeat()

        dataset = dataset.map(preprocess_fn, num_parallel_calls=batch_size)
        # Note: batching here (in the dataset that will be interleaved) means that elements will not be shuffled as
        # much. However, it brings a big speed boost.
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset


    def _align_dataset_size(self):
        num_of_files_to_use = len(self.fnames)
        if num_of_files_to_use < self.n_splits:
            raise ValueError(f"increase --dataset-percentage-to-use to have enough files to feed {self.n_splits} splits.")
        num_of_files_to_use = num_of_files_to_use - (num_of_files_to_use % self.n_splits)

        self.fnames = self.fnames[:num_of_files_to_use]
        self.n_files = len(self.fnames)


    def n_validation_images_useable(self, batch_size):
        """
        Calculates how many validation images will be used for a given batch_size.
        For example, in a normal ImageNet validation with batch size 4, you would use all 50,000 validation images
        because 4 divides 50,000. However, in this case, if you had n_splits of 16, you would have 3125 images in each
        split, and take 3124 images from each: meaning you would use (50,000 - 16) = 49,984 images.
        """
        n_splits = self._how_many_splits(batch_size, is_training=False)
        # This is a whole number because of how n_splits is constructed
        imgs_per_split = self.n_validation_images // n_splits
        # Images that will be dropped in the batching
        leftover_images_per_split = imgs_per_split % batch_size
        n_validation_useable = (imgs_per_split - leftover_images_per_split) * n_splits
        return n_validation_useable

    @staticmethod
    def even_split(arr, n_splits):
        """
        Splits an array evenly into n_splits by 'chunking'
        ImageNet records are split such that 'chunking' them results in the same number of files per split
        """
        if len(arr) % n_splits != 0:
            raise ValueError(f"Can't evenly split array of length {len(arr)} into {n_splits} splits.")
        n_per_split = len(arr) // n_splits
        return [arr[i * n_per_split: (i + 1) * n_per_split] for i in range(n_splits)]

    def _how_many_splits(self, batch_size, is_training):
        """
        Calculates how many splits the TFRecords dataset should be divided into
        """
        n_splits = next_pow_2(self.n_cores / batch_size)

        # condition to work if 'train_with_valid_preprocessing' is set
        if (not is_training) and (len(self.fnames) <= 128):
            # 50000 = (2 ** 4) * (5 ** 5), so 16 is the highest number of splits that will have an equal number
            # of images in each subset (important for the 'take' function)
            n_splits = min(n_splits, 16)
        return n_splits

    def get_dataset(self, batch_size, is_training, datatype):
        self.n_splits = self._how_many_splits(batch_size, is_training)

        self._align_dataset_size()
        assert self.n_files % self.n_splits == 0, "splits must evenly divide number of files"

        self.split_fnames = self.even_split(self.fnames, self.n_splits)

        # Each of these dummies will map to a dataset
        dummy_dataset = tf.data.Dataset.range(self.n_splits)
        # Each of the splits gets an equal share of the cores
        cores_per_split = self.n_cores // batch_size
        # num_parallel_calls cannot exceed the cycle length
        num_prll_calls = min(cores_per_split, self.n_splits)
        d = dummy_dataset.interleave(lambda x: self._get_subset(x, batch_size=batch_size, is_training=is_training,
                                                                datatype=datatype),
                                     cycle_length=self.n_splits, block_length=1,
                                     num_parallel_calls=num_prll_calls)

        if not is_training:
            # Take the right number of batches and repeat after the interleave (once the whole val-set has been seen)
            d = d.take(self.n_validation_images_useable(batch_size) // batch_size)
            d = d.repeat()

        print(f'Interleaving {self.n_splits} datasets (bs {batch_size}) with {num_prll_calls} parallel calls.')

        # Prefetching the dataset does not show ANY performance boost in testing
        # [using prefetch(...AUTOTUNE)) actually results in a memory leak]
        # d = d.prefetch(16)
        return d


def next_pow_2(x):
    ans = 1
    while ans < x:
        ans *= 2
    return ans


def accelerator_side_preprocessing(image, opts):
    # To speed up the data input, these steps can be done off-host
    from Datasets.imagenet_preprocessing import normalise_image
    image = normalise_image(image, full_normalisation=opts["normalise_input"])
    return image


def fused_accelerator_side_preprocessing(image, opts):
    # To speed up the data input, these steps can be done off-host
    from Datasets.imagenet_preprocessing import fused_normalise_image
    image = fused_normalise_image(image, full_normalisation=opts["normalise_input"])
    return image
