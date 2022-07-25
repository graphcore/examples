# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
import os
import tempfile

import tensorflow as tf
from absl import app, flags, logging
from ogb.graphproppred import Evaluator
import wandb

import utils
import xpu
from data_utils.data_generators import PackedBatchGenerator
from model import create_model

from data_utils.generated_graph_data import GeneratedGraphData
from utils import ThroughputCallback, get_optimizer, str_dtype_to_tf_dtype


flags.DEFINE_integer("micro_batch_size", 8, 'Compute batch size (if using packing this is measured in "packs per batch")')
flags.DEFINE_integer("global_batch_size", None, 'Global batch size (Includes batches across gradient accumulation factor and replicas)')

flags.DEFINE_integer("n_nodes_per_pack", 248, 'nodes per "pack"')
flags.DEFINE_integer("n_edges_per_pack", 512, 'edges per "pack"')
flags.DEFINE_integer("n_graphs_per_pack", 16, 'maximum number of graphs per "pack"')

flags.DEFINE_integer("epochs", 100, "maximum number of epochs to run for")

flags.DEFINE_float("lr", 2e-5, "learning rate")
flags.DEFINE_boolean("cosine_lr", False, "use a cosine lr decay")
flags.DEFINE_float("min_lr", 0, "minimum learning rate for the cosine scheduler")

flags.DEFINE_float("loss_scaling", 16, "loss scaling factor (to keep gradients representable in IEEE FP16)")
flags.DEFINE_enum("adam_m_dtype", "float16", ("float16", "float32"), "dtype for the m part of the adam optimizer")
flags.DEFINE_enum("adam_v_dtype", "float32", ("float16", "float32"), "dtype for the v part of the adam optimizer")

flags.DEFINE_enum("dataset_name", "ogbg-molhiv", ("ogbg-molhiv",), help="which dataset to use")

flags.DEFINE_boolean("generated_data", False, "Use randomly generated data instead of a real dataset.")
flags.DEFINE_integer("generated_data_n_nodes", 24, "nodes per graph for the randomly generated dataset")
flags.DEFINE_integer("generated_data_n_edges", 50, "edges per graph for the randomly generated dataset")
flags.DEFINE_integer("generated_batches_per_epoch", 2048, "Number of batches per epoch for the randomly generated dataset")

flags.DEFINE_boolean("do_training", True, "Run training on the dataset")
flags.DEFINE_boolean("do_validation", True, "Run validation on the dataset")
flags.DEFINE_boolean("do_test", True, "Run test on the dataset")

flags.DEFINE_boolean("execution_profile", False, "Create an execution profile in TensorBoard.")
flags.DEFINE_boolean("wandb", default=True, help="Enable logging to Weights & Biases")

flags.DEFINE_string("checkpoint_path", default=None, help="Path to checkpoint file if skipping training.")
flags.DEFINE_integer("checkpoint_every_n_epochs", default=1, help="Create checkpoints every N epochs.")


FLAGS = flags.FLAGS


class ScaledBinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, scaling_factor, dtype, **kwargs):
        super().__init__(**kwargs)
        self.scaling_factor = tf.cast(scaling_factor, dtype)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.scaling_factor * super().__call__(y_true, y_pred, sample_weight=sample_weight)


def cosine_lr(epoch, lr):
    completed_fraction = epoch / FLAGS.epochs
    cosine_decayed = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * completed_fraction))
    lr = max(cosine_decayed * FLAGS.lr, FLAGS.min_lr)
    if FLAGS.wandb:
        wandb.log({"lr": lr})
    return lr


def main(_):
    if FLAGS.wandb:
        wandb.init(entity="sw-apps",
                   project="GIN-test",
                   config=FLAGS.flag_values_dict())

    tf.keras.mixed_precision.set_global_policy(
        "float16" if FLAGS.dtype == "float16" else "float32")

    if FLAGS.generated_data:
        batch_generator = GeneratedGraphData(
            FLAGS.micro_batch_size,
            nodes_per_graph=FLAGS.generated_data_n_nodes,
            edges_per_graph=FLAGS.generated_data_n_edges,
            batches_per_epoch=FLAGS.generated_batches_per_epoch,
            n_graphs_per_pack=FLAGS.n_graphs_per_pack,
            latent_size=FLAGS.n_latent)
    else:
        batch_generator = PackedBatchGenerator(
            dataset_name=FLAGS.dataset_name,
            n_packs_per_batch=FLAGS.micro_batch_size,
            fold='train',
            max_graphs_per_pack=FLAGS.n_graphs_per_pack,
            max_edges_per_pack=FLAGS.n_edges_per_pack,
            max_nodes_per_pack=FLAGS.n_nodes_per_pack,
            n_epochs=FLAGS.epochs)

    ds = batch_generator.get_tf_dataset()

    if FLAGS.global_batch_size is not None:
        gradient_accumulation_factor = FLAGS.global_batch_size // (FLAGS.micro_batch_size * FLAGS.replicas)
    else:
        gradient_accumulation_factor = 1

    steps_per_epoch = batch_generator.batches_per_epoch
    steps_per_execution_per_replica = steps_per_epoch // FLAGS.replicas
    steps_per_execution_per_replica = gradient_accumulation_factor * (steps_per_execution_per_replica // gradient_accumulation_factor)
    new_steps_per_epoch = steps_per_execution_per_replica * FLAGS.replicas
    if new_steps_per_epoch != steps_per_epoch:
        logging.warning(
            "Steps per epoch has been truncated from"
            f" {steps_per_epoch} to {new_steps_per_epoch}"
            " in order for it to be divisible by steps per execution.")
    steps_per_epoch = new_steps_per_epoch
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Steps per execution per replica: {steps_per_execution_per_replica}")

    optimizer_options = dict(
        name=FLAGS.opt.lower(),
        learning_rate=FLAGS.lr,
        dtype=str_dtype_to_tf_dtype(FLAGS.dtype),
        m_dtype=str_dtype_to_tf_dtype(FLAGS.adam_m_dtype),
        v_dtype=str_dtype_to_tf_dtype(FLAGS.adam_v_dtype),
        gradient_accumulation_factor=gradient_accumulation_factor,
        replicas=FLAGS.replicas,
    )

    # Create a temporary directory for the checkpoints
    with tempfile.TemporaryDirectory() as model_dir:

        if FLAGS.do_training:
            strategy_training = xpu.configure_and_get_strategy(FLAGS.replicas)
            with strategy_training.scope():
                model = create_model()
                utils.print_trainable_variables(model)

                if FLAGS.loss_scaling > 1.0:
                    losses = ScaledBinaryCrossentropy(
                        scaling_factor=FLAGS.loss_scaling,
                        dtype=str_dtype_to_tf_dtype(FLAGS.dtype)
                    )
                else:
                    losses = tf.keras.losses.BinaryCrossentropy()

                # Weighted metrics are used because of the batch packing
                weighted_metrics = [
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.AUC()
                ]

                callbacks = [
                    ThroughputCallback(
                        # the throughput depends on the COMPUTE batch size, not the TOTAL batch size
                        samples_per_epoch=batch_generator.n_graphs_per_epoch,
                        log_wandb=FLAGS.wandb)
                ]
                if FLAGS.cosine_lr:
                    callbacks.append(tf.keras.callbacks.LearningRateScheduler(cosine_lr))
                if FLAGS.execution_profile:
                    callbacks.append(tf.keras.callbacks.TensorBoard(profile_batch=[2], log_dir='logs'))
                if FLAGS.wandb:
                    callbacks.append(wandb.keras.WandbCallback())

                logging.info("Running training...")
                logging.info(f"Saving weights to {model_dir}")
                model_path = os.path.join(model_dir, 'model-{epoch:05d}')
                callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(
                        model_path,
                        verbose=1,
                        save_best_only=False,
                        save_weights_only=True,
                        period=FLAGS.checkpoint_every_n_epochs))

                model.compile(
                    optimizer=get_optimizer(**optimizer_options),
                    loss=losses,
                    weighted_metrics=weighted_metrics,
                    steps_per_execution=steps_per_execution_per_replica)

                # if the total batch size exceeds the compute batch size
                if xpu.IS_IPU:
                    model.set_gradient_accumulation_options(
                        gradient_accumulation_steps_per_replica=gradient_accumulation_factor,
                        offload_weight_update_variables=False)

                model.fit(
                    ds,
                    steps_per_epoch=steps_per_epoch,
                    epochs=FLAGS.epochs,
                    callbacks=callbacks
                )

        if FLAGS.do_validation:
            logging.info(f"Running validation...")
            # Use default 1 replica for validation
            strategy_val = xpu.configure_and_get_strategy(num_replicas=1)
            with strategy_val.scope():
                evaluator = Evaluator(name=FLAGS.dataset_name)

                if FLAGS.generated_data:
                    valid_gen = batch_generator
                else:
                    valid_gen = PackedBatchGenerator(
                        dataset_name=FLAGS.dataset_name,
                        n_packs_per_batch=FLAGS.micro_batch_size,
                        fold='valid',
                        max_graphs_per_pack=FLAGS.n_graphs_per_pack,
                        max_edges_per_pack=FLAGS.n_edges_per_pack,
                        max_nodes_per_pack=FLAGS.n_nodes_per_pack,
                        randomize=False
                    )

                val_ds = valid_gen.get_tf_dataset().take(valid_gen.batches_per_epoch).cache().repeat()
                val_ground_truth, val_include_mask = valid_gen.get_ground_truth_and_masks()
                val_ground_truth = val_ground_truth[val_include_mask]

                best_val_auc = 0

                model = create_model()
                model.compile(
                    optimizer=get_optimizer(**optimizer_options),
                    weighted_metrics=weighted_metrics,
                    steps_per_execution=valid_gen.batches_per_epoch)

                if FLAGS.checkpoint_path:
                    checkpoint_paths = {-1: FLAGS.checkpoint_path}
                    logging.info(f"Validating the model with the checkpoint at path {checkpoint_paths}")
                else:
                    checkpoint_paths = {
                        epoch: os.path.join(model_dir, f"model-{epoch:05d}")
                        for epoch in range(1, FLAGS.epochs + 1)
                    }
                    logging.info(f"Validating over a sweep of checkpoints: {checkpoint_paths}")

                for epoch, checkpoint_path in checkpoint_paths.items():
                    model.load_weights(checkpoint_path).expect_partial()
                    prediction = model.predict(val_ds, steps=valid_gen.batches_per_epoch).squeeze()

                    if len(val_include_mask) > len(prediction):
                        val_include_mask = val_include_mask[:len(prediction)]
                        val_ground_truth = val_ground_truth[:len(prediction)]

                    # val_include_mask may be shorter than the predictions —
                    #   that is fine (it will just be padding after that point)
                    prediction = prediction[:len(val_include_mask)][val_include_mask.squeeze() == 1]

                    # we will use the official AUC evaluator from the OGB repo, not the keras one
                    result = evaluator.eval({'y_true': val_ground_truth[:, None], 'y_pred': prediction[:, None]})
                    this_auc = result['rocauc']

                    if FLAGS.wandb:
                        wandb.log({'epoch': epoch, 'val_auc': this_auc})

                    if this_auc > best_val_auc:
                        best_val_auc = this_auc
                        best_model_path = checkpoint_path
                        logging.info(f"{best_val_auc:.3f} is the validation set AUC for the model at {best_model_path}")

        if FLAGS.do_test:
            logging.info(f"Running test...")

            # Use default 1 replica for test
            strategy_test = xpu.configure_and_get_strategy(num_replicas=1)
            with strategy_test.scope():

                if FLAGS.generated_data:
                    test_gen = batch_generator
                else:
                    test_gen = PackedBatchGenerator(
                        dataset_name=FLAGS.dataset_name,
                        n_packs_per_batch=FLAGS.micro_batch_size,
                        fold='test',
                        max_graphs_per_pack=FLAGS.n_graphs_per_pack,
                        max_edges_per_pack=FLAGS.n_edges_per_pack,
                        max_nodes_per_pack=FLAGS.n_nodes_per_pack,
                        randomize=False
                    )

                model = create_model()
                model.compile(
                    optimizer=get_optimizer(**optimizer_options),
                    weighted_metrics=weighted_metrics,
                    steps_per_execution=test_gen.batches_per_epoch)

                test_ds = test_gen.get_tf_dataset().take(test_gen.batches_per_epoch).cache().repeat()
                test_ground_truth, test_include_mask = test_gen.get_ground_truth_and_masks()
                test_ground_truth = test_ground_truth[test_include_mask]

                if FLAGS.checkpoint_path:
                    checkpoint_path = FLAGS.checkpoint_path
                    logging.info(f"Testing the model with the checkpoint at path {checkpoint_path}")
                else:
                    logging.info(f"Testing the model with the best validation loss (at {best_model_path})")
                    checkpoint_path = best_model_path
                model.load_weights(checkpoint_path).expect_partial()

                prediction = model.predict(test_ds, steps=test_gen.batches_per_epoch).squeeze()

                if len(test_include_mask) > len(prediction):
                    test_include_mask = test_include_mask[:len(prediction)]
                    test_ground_truth = test_ground_truth[:len(prediction)]

                prediction = prediction[:len(test_include_mask)][test_include_mask.squeeze() == 1]

                result = evaluator.eval({'y_true': test_ground_truth[:, None], 'y_pred': prediction[:, None]})
                this_auc = result['rocauc']
                logging.info(f"\nTest AUC of {this_auc:.3f}")
                if FLAGS.wandb:
                    wandb.log({'test_auc': this_auc, 'best_val_auc': best_val_auc})


if __name__ == '__main__':
    app.run(main)
