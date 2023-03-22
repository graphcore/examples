# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow import keras
from ipu_tensorflow_addons.keras.optimizers import AdamIpuOptimizer
import strategy_utils as su
import os
import argparse
import cv2
import numpy as np
import nif
import time
import datetime
import sys
import magic


if tf.__version__[0] != "2":
    raise ImportError("TensorFlow 2 is required")


def siren_init(n):
    s = np.sqrt(6 / n)
    return tf.keras.initializers.RandomUniform(minval=-s, maxval=s)


def sin_activation(x, use_fp16):
    if use_fp16:
        return tf.cast(tf.math.sin(tf.cast(x, dtype=tf.float16)), dtype=tf.float32)
    return tf.math.sin(x)


def create_model(input_dim, input_dtype, layer_size, num_layers, use_siren, fp16_sin, color_matrix):
    if use_siren:
        init_fn = siren_init(n=layer_size)
        act_fn = None
        concat_dim = 0
    else:
        init_fn = "glorot_uniform"
        act_fn = "relu"
        concat_dim = input_dim

    input = keras.Input(shape=(input_dim), dtype=input_dtype)
    half_size = num_layers // 2

    front_end = []
    for l in range(half_size - 1):
        front_end.append(keras.layers.Dense(layer_size, activation=act_fn, use_bias=False, kernel_initializer=init_fn))
    # Last layer of front-end reduces dimension so that after concatenating
    # the input the back end can have the same layer size:
    front_end.append(
        keras.layers.Dense(layer_size - concat_dim, activation=act_fn, use_bias=False, kernel_initializer=init_fn)
    )

    concat = tf.keras.layers.Concatenate(axis=1)

    back_end = []
    for l in range(half_size - 1):
        back_end.append(keras.layers.Dense(layer_size, activation=act_fn, use_bias=False, kernel_initializer=init_fn))
    # Penultimate layer has a bias:
    back_end.append(keras.layers.Dense(layer_size, activation=act_fn, kernel_initializer=init_fn))
    # Final layer maps back to colour-space coordinates:
    back_end.append(keras.layers.Dense(3, activation="linear"))

    # Functional model:
    x = input
    if use_siren:
        x = x * 30  # See SIREN paper (factor controls frequency distribution).

    for dense in front_end:
        x = dense(x)
        if use_siren:
            x = sin_activation(x, fp16_sin)

    if not use_siren:
        # Don't skip the input to back end of siren network:
        x = concat([x, input])

    for dense in back_end:
        x = dense(x)
        if use_siren:
            x = sin_activation(x, fp16_sin)

    output = x
    if color_matrix is not None:
        # NOTE: Regression target is in BGR format because that is how OpenCV loads images:
        color_matrix_init = keras.initializers.Constant(color_matrix)
        color_space_conversion = keras.layers.Dense(
            units=3,
            use_bias=False,
            activation=None,
            trainable=False,
            kernel_initializer=color_matrix_init,
            name="color_space_conversion",
        )
        bgr = color_space_conversion(x)
        output = bgr

    return keras.Model(inputs=input, outputs=output, name="nif_model")


def parse_args():
    parser = argparse.ArgumentParser("Neural Image Field (NIF) Generator")
    parser.add_argument(
        "--input", type=str, default="Mandrill_portrait_2_Berlin_Zoo.jpg", help="Input image file name."
    )
    parser.add_argument(
        "--blur", type=int, default=0, help="Size of Gaussian blur kernel applied to the input (0 to disable)."
    )
    parser.add_argument(
        "--model", type=str, default="./saved_model/", help="Output path to save the trained NIF model."
    )
    parser.add_argument("--learning-rate", type=float, default=0.001, help="The learning rate for ADAM.")
    parser.add_argument("--batch-size", type=int, default=512, help="The batch size.")
    parser.add_argument("--epochs", type=int, default=2000, help="Total number of epochs to train for.")
    parser.add_argument("--layer-size", type=int, default=256, help="Hidden size of the MLPs."),
    parser.add_argument(
        "--layer-count",
        type=int,
        default=6,
        help="Number of MLP layers. Should be multiple of 2 as the network is split into a front and back end (with the input concatenated to the back end as in NERF models).",
    )
    parser.add_argument(
        "--train-samples", type=int, default=1000000, help="The number of image samples used to train the NIF."
    )
    parser.add_argument(
        "--embedding-dimension", type=int, default=10, help="Dimension of the position embedding space for UV coords."
    )
    parser.add_argument(
        "--embedding-sigma",
        type=float,
        default=2.0,
        help="Base for the positional embedding (power base for Fourier features).",
    )
    parser.add_argument(
        "--no-position-embedding",
        action="store_true",
        help="Disable the position embedding and train directly on UV coords.",
    )
    parser.add_argument(
        "--deterministic-samples",
        action="store_true",
        help="Create training data from one uv sample per pixel (instead of randomly distributed).",
    )
    parser.add_argument(
        "--disable-psnr",
        action="store_true",
        help="Disable peak signal-to-noise ratio evaluation during training. (Note: evaluation of the PSNR launches a separate process on a separate device which may be undesirable.)",
    )
    parser.add_argument(
        "--replicas", type=int, default=1, help="Number of IPUs to replicate model over for data parallel training."
    )
    parser.add_argument(
        "--gradient-accumulation-count",
        type=int,
        default=1,
        help="Number of batches to process before applying a weight update.",
    )
    parser.add_argument(
        "--callback-period",
        type=int,
        default=10,
        help="Interval in epochs at which to log training stats and evaluate PSNR (if enabled).",
    )
    parser.add_argument("--siren", action="store_true", help="Use a SIREN MLP instead of an relu-MLP network.")
    parser.add_argument("--sin-fp16", action="store_true", help="If using SIREN use half precision sin function.")
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="If set the program will execute a single step then exit. This is useful if profiling using PopVision.",
    )
    parser.add_argument(
        "--no-ipu",
        action="store_true",
        help="Set this flag to disable IPU specific code paths (other IPU specific arguments will be ignored).",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Override the automatically generated name for the tensorboard logging path.",
    )
    parser.add_argument("--fp16", action="store_true", help="Train in fp16.")
    parser.add_argument("--loss-scale", type=float, default=32768, help="Loss scale (affects fp16 training only).")
    parser.add_argument("--mse", action="store_true", help="Use MSE loss instead of the default Huber loss.")
    parser.add_argument("--disable-stochastic-rounding", action="store_true", help="Disable stochastic rounding.")
    parser.add_argument(
        "--color-space",
        type=str,
        default="rgb",
        choices=["rgb", "yuv", "ycocg"],
        help="Force the network to predict in the specified color-space and convert to RGB with a static transform layer.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.fp16:
        tf.keras.mixed_precision.set_global_policy("float16")

    if args.layer_count % 2:
        raise ValueError("Layer count must be a multiple of 2.")

    # Load image (the extra flags enable loading of HDR images in EXR format):
    img = cv2.imread(args.input, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if args.blur > 0:
        img = cv2.GaussianBlur(img, (args.blur, args.blur), 0)
    height = img.shape[0]
    width = img.shape[1]
    input_mean = np.mean(np.mean(img, axis=0), axis=0)  # Mean BGR value
    input_min = img.min().astype(float)
    input_max = img.max().astype(float)
    print(
        f"Image loaded. Size: {img.shape} Type: {img.dtype} Mean value: {input_mean} Min/max: {input_min}/{input_max}"
    )
    rng = np.random.default_rng()

    if args.deterministic_samples:
        # Create one uv sample per pixel.
        train_uv = nif.deterministic_uv_samples(img.shape, dtype=np.float32)
        sample_count = train_uv.shape[0]
        print(f"Using deterministic samples: sample count: {sample_count}")
    else:
        # Using random uv-coords mimics some real-world use cases where we only have
        # access to a sparse set of samples and wen to do image reconstruction. It can
        # also be more efficient to approximate high resolution images as the number
        # of samples needed could be much fewer than the number of pixels:
        sample_count = args.train_samples
        print(f"Creating {sample_count} uniform image uv samples...")
        train_uv = nif.stochastic_uv_samples(rng, sample_count, img.shape, dtype=np.float32)

    # Decide whether to use an HDR transfer function based on file type:
    file_magic = magic.from_file(args.input)
    transfer_function = "linear"
    print(f"Input: {file_magic}")
    if "OpenEXR" in file_magic:
        print(f"HDR input detected: using log transfer function")
        transfer_function = "log"

    # Get BGR image values at the sample coordinates:
    _, file_extension = os.path.splitext(args.input)
    train_values, encode_params = nif.encode_samples(img, train_uv, transfer_function, "input_samples" + file_extension)
    print(f"Encode params: {encode_params}")

    max_encoded, mean_encoded = nif.value_stats(train_values)
    print(f"Max value after encode: {max_encoded} mean after encode {mean_encoded}")

    # A positional encoding is necessary for the network to
    # learn high frequency functions:
    if args.siren:
        args.no_position_embedding = True  # Mixing siren with position embedding just makes a mess.
        loss_fn = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mse_loss")
    else:
        if args.mse:
            loss_fn = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mse_loss")
        else:
            loss_fn = tf.keras.losses.Huber(delta=0.001, reduction="sum_over_batch_size", name="huber_loss")
    embedding_dimension = 0 if args.no_position_embedding else args.embedding_dimension
    if not args.no_position_embedding:
        t0 = time.time()
        train_uv = nif.uv_positional_encode(train_uv, embedding_dimension, args.embedding_sigma)
        t1 = time.time()
        print(f"UV encode time: {t1 - t0}")

    ds, steps_per_epoch, steps_per_exec = nif.make_train_data(train_uv, train_values, args)
    if args.deterministic_samples:
        ds = ds.cache().shuffle(buffer_size=sample_count)
    ds = ds.cache().repeat()

    strategy = su.get_train_strategy(args.no_ipu, args.replicas, args.fp16, args.disable_stochastic_rounding)

    color_matrix = nif.color_space_to_bgr_matrix(args.color_space)

    with strategy.scope():
        model = create_model(
            train_uv.shape[-1],
            train_uv.dtype,
            args.layer_size,
            args.layer_count,
            args.siren,
            args.sin_fp16,
            color_matrix,
        )
        if not args.no_ipu:
            # Use running mean for numerical stability:
            model.set_gradient_accumulation_options(
                gradient_accumulation_steps_per_replica=args.gradient_accumulation_count,
                gradient_accumulation_reduction_method="running_mean",
                dtype=tf.float32,
            )
        model.summary()

        # Explicitly keep Adam optimizer state in fp32 (otherwise they inherit the type of the master weights which might be fp16):
        opt = AdamIpuOptimizer(
            learning_rate=args.learning_rate, m_dtype="float32", v_dtype="float32", vhat_dtype="float32"
        )
        if args.fp16:
            # Loss scaling necessary with fp16 master weights:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=False, initial_scale=args.loss_scale)

        # Train the model:
        model.compile(loss=loss_fn, optimizer=opt, steps_per_execution=steps_per_exec)

        # Need to save some NIF metadata with the model also:
        saved_model_path = args.model
        metadata_file = nif.metadata_path_from_keras_model_path(saved_model_path)
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        nif.save_metadata(
            metadata_file,
            name=args.input,
            args=sys.argv,
            shape=img.shape,
            encode_params=encode_params,
            embedding_dim=embedding_dimension,
            embedding_sigma=args.embedding_sigma,
            model_path=args.model,
        )

        model_name_str = os.path.basename(os.path.normpath(args.model))
        time_string = datetime.datetime.now().isoformat()
        h5_model_path = os.path.join(saved_model_path, "assets.extra", "converted.hdf5")

        if args.tensorboard_dir is None:
            tb_logdir = f"./logs/run_{model_name_str}_{time_string}/"
        else:
            tb_logdir = os.path.join(args.tensorboard_dir, f"run_{model_name_str}_{time_string}")

        eval_callback = nif.EvalCallback(
            original_file=args.input,
            model_path=saved_model_path,
            period=args.callback_period,
            compute_psnr=not args.disable_psnr,
            no_ipu=args.no_ipu,
            log_dir=tb_logdir,
        )
        if args.single_step:
            callbacks = []
            epochs = 1
            steps_per_epoch = 1
            steps_per_exec = 1
        else:
            callbacks = [
                # Two save callbacks: one is for standard Keras format:
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=saved_model_path, save_weights_only=False, save_best_only=False
                ),
                # Second callback is for export to a format the Poplar ray-tracer can read:
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=h5_model_path, include_optimizer=False, save_weights_only=False, save_best_only=False
                ),
                eval_callback,
                tf.keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=5),
            ]
            epochs = args.epochs

        model.fit(ds, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=0, callbacks=callbacks)
        eval_callback.cleanup(epoch=args.epochs)
        print(f"Trained NIF.")
