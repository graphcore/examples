# Copyright 2019 Graphcore Ltd.
# coding=utf-8
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf

_BINARISED_MNIST_TR = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat'
_BINARISED_MNIST_TEST = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat'


# noinspection PyPep8Naming
def download_dataset(dataset_name='mnist'):
    """
    Load MNIST dataset using keras convenience function

    Args:
        dataset_name (str): which of the keras datasets to download
        dtype (np.dtype): Type of numpy array

    Returns tuple[np.array[float]]:
        (train images, train labels), (test images, test labels)
    """
    if dataset_name == 'mnist':
        return tf.keras.datasets.mnist.load_data()
    elif dataset_name == 'binarised_mnist':
        return load_binarised_mnist_data()


def preprocess_np_inputs(an_array, datatype, flatten_images, normaliser=255.):
    """Flattens and normalises images"""
    preprocessed = an_array.astype(datatype)

    if flatten_images:
        # Convert each image to a vector
        preprocessed = flatten_2d_images(preprocessed)

    # Normalise [0, 255] -> [0, 1]
    preprocessed /= normaliser

    return preprocessed


def xy_array_combine(arrays, shuffle=True):
    """Cobines X and Y arrays into a single 2D numpy array, shuffles if required"""

    x_arr = np.reshape(arrays['x'], [arrays['x'].shape[0], -1])
    if arrays['y'].ndim == 1:
        y_arr = np.expand_dims(arrays['y'], 1)
    else:
        y_arr = arrays['y']
    arrays = np.concatenate((x_arr, y_arr), axis=1)
    if shuffle:
        shuffle_idx = np.random.permutation(arrays.shape[0])
        arrays = arrays[shuffle_idx]
    else:
        shuffle_idx = np.arange(arrays.shape[0])
    return arrays, shuffle_idx


def split_xy_array(arrays, sizes, shuffle_before=True, split_idx=None):
    """Splits examples in X, Y arrays at consistent indices"""
    x_arr_orig_shp = list(arrays['x'].shape[1:])
    if split_idx is not None:
        shuffle_before = False
    arrays, idx_order = xy_array_combine(arrays, shuffle=shuffle_before)
    if split_idx is None:
        split_arrs, [tr_idx, val_idx] = np.split(arrays, sizes), np.split(idx_order, sizes)
    else:
        tr_idx, val_idx = split_idx['train'], split_idx['validation']
        split_arrs = [arrays[tr_idx], arrays[val_idx]]
    return [{'x': np.reshape(s[:, :-1], [s.shape[0]] + x_arr_orig_shp), 'y': s[:, -1]} for s in split_arrs],\
        (tr_idx, val_idx)


def flatten_2d_images(img_array):
    """Reduces each 2D image in a 3D array to a single vector.
    NOTE: axis 0 assumed to represent different images"""
    return np.reshape(img_array, (img_array.shape[0], -1))


def load_binarised_mnist_data():
    """
    Load the dataset of pre-randomly-binarised MNIST, as originally released by Larochelle and Murray, 2011
    (http://www.dmi.usherb.ca/~larocheh/publications/aistats2011_nade.pdf).

    Returns:
        tuple[np.array]: train set and test set

    """
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    def load_from_url(url, set_name):
        # Load matlab file with data from Larochelle's website
        data_dir = f'binarised_mnist_{set_name}'
        path_local = tf.keras.utils.get_file(data_dir, url)

        with open(path_local, 'r') as f:
            lines = f.readlines()

        return lines_to_np_array(lines).reshape((-1, 28, 28)) * 255.  # Rescale back to [0, 1] in preprocess_np_data

    # Load all datasets from URL
    train = load_from_url(_BINARISED_MNIST_TR, 'train')
    test = load_from_url(_BINARISED_MNIST_TEST, 'test')

    # Not using labels - set to -1
    return (train, -1 * np.ones((train.shape[0],))), \
           (test, -1 * np.ones((test.shape[0],)))


def make_iterator_from_np(np_arrays, batch_size, shuffle=True, shuffle_buffer=50000, rand_binarise=True,
                          drop_remain=True, repeat=True):
    """
    Converts a numpy array to a tensorflow initialisable iterator over batches

    Args:
        np_arrays (np.array): dict of numpy arrays to split into batches. {'x': <input_array>, 'y': <outputs>}
        batch_size (int): how many samples in each batch
        shuffle (bool): whether to shuffle data
        shuffle_buffer (int): how many samples in buffer for shuffling, if `shuffle` is True
        rand_binarise (bool): whether to dynamically binarise data
        drop_remain (bool): whether to drop the remainder if the final batch size is not equal to main batch size
        repeat (bool): whether to repeat the data indefinitely
        one_hot (bool): whether to use one-hot, sparse vector representation of labels
        n_classes (int): only needed if one_hot is True

    Returns:
        tuple: iterator.get_next() operator, iterator.initializer operator
    """
    def _random_binarisation(i, x_batch, y_batch):
        runif = tf.random_uniform(tf.shape(x_batch), 0., 1., dtype=x_batch.dtype)
        return i, tf.cast(runif <= x_batch, x_batch.dtype), y_batch

    # Zip index of training examples, which will be shuffled consistently with data
    tf_data = tf.data.Dataset.from_tensor_slices((np.arange(0, np_arrays['x'].shape[0], dtype=np.int32),
                                                  np_arrays['x'],
                                                  np_arrays['y']))

    if repeat:
        if shuffle:
            tf_data = tf_data.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer))
        else:
            tf_data = tf_data.repeat()
    tf_data_batched = tf_data.batch(batch_size, drop_remainder=drop_remain)

    if rand_binarise:
        # Dynamic binarisation as introduced by Wu et al, 2016,
        # https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf
        tf_data_batched = tf_data_batched.map(_random_binarisation)

    tf_data_batched = tf_data_batched.prefetch(tf.contrib.data.AUTOTUNE)
    tf_data_iter = tf_data_batched.make_initializable_iterator()

    return tf_data_iter, tf_data_batched
