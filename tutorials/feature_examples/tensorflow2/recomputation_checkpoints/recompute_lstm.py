# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python import ipu
import ipu_tensorflow_addons.keras.layers as ipu_layers
import argparse


def make_split_layer_model(args):
    """Split an LSTM into checkpointed sections."""

    def split_lstm(seq_len, checkpoints, inputs, num_units):
        lstm_layer = ipu_layers.PopnnLSTM(num_units, return_state=True)
        out_slices = []
        initial_state = None

        # The input is split with the intermediate states checkpointed
        for i in range(checkpoints + 1):
            begin = i * seq_len // (checkpoints + 1)
            end = (i + 1) * seq_len // (checkpoints + 1)
            seq_slice = tf.keras.layers.Lambda(lambda x, b=0, e=0: x[:, b:e, :], arguments={"b": begin, "e": end})(
                inputs
            )
            output, out_h, out_c = lstm_layer(seq_slice, initial_state=initial_state)

            # We don't need to checkpoint the final LSTM state.
            if i != checkpoints:
                out_h, out_c = ipu_layers.RecomputationCheckpoint()([out_h, out_c])
            # Set the initial state of the next LSTM section with the previous section's hidden state.
            initial_state = (out_h, out_c)
            out_slices.append(output)
        return tf.keras.layers.Concatenate(axis=1)(out_slices)

    input_layer = tf.keras.layers.Input(shape=(args.seq_len, 512))

    # We need to use a pipeline even if only a single IPU is being used.
    with tf.keras.ipu.PipelineStage(0):
        x = split_lstm(
            seq_len=args.seq_len,
            checkpoints=args.checkpoints,
            inputs=input_layer,
            num_units=256,
        )
    with tf.keras.ipu.PipelineStage(1):
        x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x)
    model.compile(tf.optimizers.SGD(0.001), loss="mse", steps_per_execution=4)

    # Checkpoints require Grouped or Sequential pipelining and
    # RecomputeAndBackpropagateInterleaved.
    # In this example we map both pipeline stages to the same IPU.
    pipeline_schedule = ipu.pipelining_ops.PipelineSchedule.Sequential
    mode = ipu.pipelining_ops.RecomputationMode.RecomputeAndBackpropagateInterleaved
    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=4,
        pipeline_schedule=pipeline_schedule,
        device_mapping=[0, 0],
        recomputation_mode=mode,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=int,
        default=7,
        help="How many recomputation checkpoints to use.",
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="The length of the LSTM sequence.")
    args = parser.parse_args()
    if args.seq_len % (args.checkpoints + 1) != 0:
        print(
            f"A sequence of length {args.seq_len} can't be evenly divided by "
            f"{args.checkpoints} checkpoints. This may result in extra IPU "
            f"code as function outlining won't be used as efficiently."
        )

    # Make dummy dataset for LSTM.
    constant_d = tf.constant(1.0, shape=[args.seq_len, 512])
    constant_l = tf.constant(0.2, shape=[args.seq_len])
    ds = tf.data.Dataset.from_tensors((constant_d, constant_l))
    ds = ds.repeat(64).batch(16, drop_remainder=True)

    # Configure the IPU with recompute turned on
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.allow_recompute = True
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        model = make_split_layer_model(args)
        model.set_infeed_queue_options(prefetch_depth=1)
        model.fit(ds)


if __name__ == "__main__":
    main()
