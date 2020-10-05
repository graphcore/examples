# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
import os
from . import imagenet_preprocessing
from functools import partial
from math import ceil

DATASET_CONSTANTS = {
    'imagenet': {
        'IMAGE_WIDTH': 224,
        'IMAGE_HEIGHT': 224,
        'NUM_CLASSES': 1000,
        'NUM_IMAGES': 1281167,
        'NUM_VALIDATION_IMAGES': 50000,
        'SHUFFLE_BUFFER': 10000,
        'FILENAMES': {
            'TRAIN': ['train-%05d-of-01024' % i for i in range(1024)],
            'TEST': ['validation-%05d-of-00128' % i for i in range(128)]
        }
    },
    'cifar-10': {
        'IMAGE_WIDTH': 32,
        'IMAGE_HEIGHT': 32,
        'NUM_CLASSES': 10,
        'NUM_IMAGES': 50000,
        'NUM_VALIDATION_IMAGES': 10000,
        'SHUFFLE_BUFFER': 50000,
        'RECORD_BYTES': (32 * 32 * 3) + 1,
        'FILENAMES': {
            'TRAIN': ['data_batch_{}.bin'.format(i) for i in range(1, 6)],
            'TEST': ['test_batch.bin']
        }
    },
    'cifar-100': {
        'IMAGE_WIDTH': 32,
        'IMAGE_HEIGHT': 32,
        'NUM_CLASSES': 100,
        'NUM_IMAGES': 50000,
        'NUM_VALIDATION_IMAGES': 10000,
        'SHUFFLE_BUFFER': 50000,
        'RECORD_BYTES': (32 * 32 * 3) + 2,
        'FILENAMES': {
            'TRAIN': ['train.bin'],
            'TEST': ['test.bin']
        }
    },
 }


def data(opts, is_training=True):
    from .imagenet_dataset import ImageNetData
    batch_size = opts["batch_size"]
    dtypes = opts["precision"].split('.')
    datatype = tf.float16 if dtypes[0] == '16' else tf.float32

    if opts['train_with_valid_preprocessing'] and is_training:
        training_preprocessing = False
    else:
        training_preprocessing = is_training

    if opts['synthetic_data']:
        dataset = synthetic_dataset(opts)
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    else:
        if is_training:
            filenames = DATASET_CONSTANTS[opts['dataset']]['FILENAMES']['TRAIN']
        else:
            filenames = DATASET_CONSTANTS[opts['dataset']]['FILENAMES']['TEST']
        filenames = list(map(lambda path: os.path.join(opts['data_dir'], path), filenames))

        cycle_length = 1 if opts['seed_specified'] else 4
        if opts["dataset"] == 'imagenet':
            if not opts['standard_imagenet'] and not opts['distributed']:
                dataset = ImageNetData(opts, filenames=filenames).get_dataset(batch_size=batch_size,
                                                                              is_training=training_preprocessing,
                                                                              datatype=datatype)
                return dataset
            else:
                preprocess_fn = partial(imagenet_preprocess, is_training=training_preprocessing,
                                        image_size=opts["image_size"],
                                        dtype=datatype, seed=opts['seed'],
                                        full_normalisation=None if opts['no_hostside_norm'] else opts['normalise_input'],)
                dataset_fn = tf.data.TFRecordDataset
                if is_training and opts['distributed']:
                    # Shuffle after sharding
                    dataset = tf.data.Dataset.list_files(filenames, shuffle=False)
                    dataset = dataset.shard(num_shards=opts['distributed_worker_count'], index=opts['distributed_worker_index'])
                    dataset = dataset.shuffle(ceil(len(filenames) / opts['distributed_worker_count']), seed=opts['seed'])
                else:
                    dataset = tf.data.Dataset.list_files(filenames, shuffle=is_training, seed=opts['seed'])
                dataset = dataset.interleave(dataset_fn, cycle_length=cycle_length,
                                             block_length=cycle_length, num_parallel_calls=cycle_length)
        elif 'cifar' in opts["dataset"]:
            preprocess_fn = partial(cifar_preprocess, is_training=training_preprocessing, dtype=datatype,
                                    dataset=opts['dataset'], seed=opts['seed'])
            dataset = tf.data.FixedLengthRecordDataset(filenames, DATASET_CONSTANTS[opts['dataset']]['RECORD_BYTES'])
            if is_training and opts['distributed']:
                dataset = dataset.shard(num_shards=opts['distributed_worker_count'], index=opts['distributed_worker_index'])
        else:
            raise ValueError("Unknown Dataset {}".format(opts["dataset"]))

        if is_training:
            if not opts['no_dataset_cache']:
                dataset = dataset.cache()
            shuffle_buffer = DATASET_CONSTANTS[opts['dataset']]['SHUFFLE_BUFFER']
            dataset = dataset.shuffle(shuffle_buffer, seed=opts['seed'])
        else:
            dataset = dataset.take(opts["validation_batches_per_step"] *
                                   opts["validation_iterations"]*opts["validation_total_batch_size"])
            if not opts['no_dataset_cache']:
                dataset = dataset.cache()
        dataset = dataset.repeat()

        # We can't get repeatable results with parallel calls
        if opts['seed_specified']:
            print("****\n\n"
                  "  Removing parallel pipelining for reproducibility as seed is specified.\n"
                  "  This will adversely affect performance.\n\n****")
        parallel_calls = 1 if opts['seed_specified'] else opts['pipeline_num_parallel']

        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                preprocess_fn,
                batch_size=parallel_calls,
                num_parallel_calls=parallel_calls,
                drop_remainder=True
            )
        ).unbatch().batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(16)

    return dataset


def synthetic_dataset(opts):
    """Returns dataset filled with random data."""
    # Synthetic input should be within [0, 255].

    height = opts['image_size']
    width = opts['image_size']
    num_classes = DATASET_CONSTANTS[opts['dataset']]['NUM_CLASSES']

    dtypes = opts["precision"].split('.')
    datatype = tf.float16 if dtypes[0] == '16' else tf.float32

    images = tf.truncated_normal(
        [height, width, 3],
        dtype=datatype,
        mean=127,
        stddev=60,
        name='synthetic_inputs')

    labels = tf.random_uniform(
        [],
        minval=0,
        maxval=num_classes - 1,
        dtype=tf.int32,
        name='synthetic_labels')

    return tf.data.Dataset.from_tensors((images, labels))


def cifar_preprocess(raw_record, is_training, dtype, dataset, seed):
    """FROM https://github.com/tensorflow/models/blob/master/official/resnet/cifar10_main.py"""
    """Parse CIFAR-10 image and label from a raw record."""
    # Convert bytes to a vector of uint8 that is record_bytes long.
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    label_byte = 1 if dataset == 'cifar-100' else 0

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(record_vector[label_byte], tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(record_vector[label_byte + 1:DATASET_CONSTANTS[dataset]['RECORD_BYTES']],
                             [3,
                              DATASET_CONSTANTS[dataset]['IMAGE_HEIGHT'],
                              DATASET_CONSTANTS[dataset]['IMAGE_WIDTH']])

    # Convert from [depth, height, width] to [height, width, depth], and cast as float32
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    # Subtract off the mean and divide by the variance of the pixels. Always returns tf.float32.
    image = tf.image.per_image_standardization(image)

    image = tf.cast(image, dtype)

    if is_training:
        shape = image.get_shape().as_list()
        padding = 4

        image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.pad(image,
                       [[padding, padding], [padding, padding], [0, 0]],
                       "CONSTANT")
        image = tf.random_crop(image, shape, seed=seed)

    return {
        "image": image,
        "label": label
    }


def imagenet_preprocess(raw_record, is_training, dtype, seed, image_size, full_normalisation):
    image, label = imagenet_preprocessing.parse_record(raw_record, is_training, dtype,
                                                       image_size,
                                                       full_normalisation,
                                                       seed,)

    label -= 1

    return {
        "image": image,
        "label": label
    }


def add_arguments(parser):
    group = parser.add_argument_group('Dataset')
    group.add_argument('--dataset', type=str.lower, choices=["imagenet", "cifar-10", "cifar-100"],
                       help="Chose which dataset to run on")
    group.add_argument('--data-dir', type=str, required=False,
                       help="path to data. ImageNet must be TFRecords. CIFAR-10/100 must be in binary format.")
    group.add_argument('--pipeline-num-parallel', type=int,
                       help="Number of images to process in parallel")
    group.add_argument('--synthetic-data', action="store_true",
                       help="Use synthetic data")
    group.add_argument('--no-dataset-cache', action="store_true",
                       help="Don't cache dataset to host RAM")
    group.add_argument('--normalise-input', action="store_true",
                       help='''Normalise inputs to zero mean and unit variance.
                           Default approach just translates [0, 255] image to zero mean. (ImageNet only)''')
    group.add_argument('--image-size', type=int,
                       help="Size of image (ImageNet only)")
    group.add_argument('--train-with-valid-preprocessing', action="store_true",
                       help="Use validation image preprocessing when training")
    group.add_argument('--no-hostside-norm', action='store_true',
                       help="Moves ImageNet image normalisation to the IPU")
    group.add_argument('--standard-imagenet', action='store_true',
                       help='Use the standard ImageNet preprocessing pipeline.')
    return parser


def set_defaults(opts):
    if opts['synthetic_data']:
        if opts['dataset'] is None:
            raise ValueError("Please specify the synthetic dataset using --dataset.")
    else:
        if opts['data_dir'] is None:
            try:
                opts['data_dir'] = os.environ['DATA_DIR']
            except KeyError:
                raise OSError("Cannot find Cifar/ImageNet data. "
                              "Either set the DATA_DIR environment variable or use the --data-dir option.")
        # If data-dir is set but not dataset then try to infer the dataset
        if opts['dataset'] is None:
            datadir = opts['data_dir'].lower()
            if 'imagenet' in datadir:
                opts['dataset'] = 'imagenet'
            elif 'cifar100' in datadir or 'cifar-100' in datadir:
                opts['dataset'] = 'cifar-100'
            elif 'cifar' in datadir:
                opts['dataset'] = 'cifar-10'
            else:
                raise ValueError("Cannot infer the dataset being used. Please specify using --dataset")

        first_training_file = DATASET_CONSTANTS[opts['dataset']]['FILENAMES']['TRAIN'][0]
        if not os.path.exists(os.path.join(opts['data_dir'], first_training_file)):
            # Search subdirectories for data
            default_dir = {'cifar-100': 'cifar-100-binary',
                           'cifar-10': 'cifar-10-batches-bin',
                           'imagenet': 'imagenet-data'}[opts['dataset']]
            data_dir = None

            for root, _, files in os.walk(opts['data_dir']):
                if os.path.basename(root) == default_dir and first_training_file in files:
                    data_dir = root
                    break

            if data_dir is None:
                raise ValueError('No {} dataset found. Searched in {} for {}'.format(opts['dataset'],
                                                                                     opts['data_dir'],
                                                                                     os.path.join(default_dir,
                                                                                                  first_training_file)))
            opts['data_dir'] = data_dir

    opts['summary_str'] += "{}\n".format(opts['dataset'])

    if opts['data_dir']:
        opts['data_dir'] = os.path.normpath(opts['data_dir'])

    if not opts['pipeline_num_parallel']:
        opts['pipeline_num_parallel'] = 48
    else:
        opts['summary_str'] += " Pipeline Num Parallel: {}\n".format(opts["pipeline_num_parallel"])

    if opts['synthetic_data']:
        opts['summary_str'] += " Synthetic Data\n"
