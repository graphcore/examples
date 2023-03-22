# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
from datetime import datetime
import logging

import popdist.tensorflow
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import distributed
from tensorflow.python.ipu.gradient_accumulation import GradientAccumulationReductionMethod
from transformers import BertConfig
import wandb

from data_utils.batch_config import BatchConfig, get_pretraining_task, Task
from data_utils.wikipedia.load_wikipedia_data import get_pretraining_dataset
from keras_extensions.callbacks.callback_factory import CallbackFactory
from keras_extensions.learning_rate.scheduler_builder import get_lr_scheduler
from keras_extensions.optimization import get_optimizer
from model.accuracy import pretraining_accuracy_fn
from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.ipu_pretraining_model import IpuTFBertForPreTraining
from model.losses import MLMLossFunction, NSPLossFunction
from model.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from utilities.argparser import add_shared_arguments, combine_config_file_with_args
from utilities.assign_pipeline_stages import PipelineStagesAssigner
from utilities.checkpoint_utility import load_checkpoint_into_model
from utilities.ipu_utils import create_ipu_strategy, get_poplar_options_per_pipeline_stage, set_random_seeds
from utilities.metric_enqueuer import wrap_loss_in_enqueuer, wrap_stateless_metric_in_enqueuer
from utilities.options import PretrainingOptions


def pretrain(config):
    """Main function to run pretraining for BERT.
    :param config: A pydantic model object that contains the configuration to
        options for this application. See utilities/options.py for accepted
        options.
    """
    distributed_training = popdist.isPopdistEnvSet()  # Check if `poprun` has initiated distributed training.

    # Set a name for this run, and be sure it is shared by all hosts when using distributed training.
    if distributed_training:
        popdist.init()
        time_now = float(
            distributed.broadcast(tf.convert_to_tensor(value=datetime.now().timestamp(), dtype=tf.float32), 0)
        )
    else:
        time_now = datetime.now().timestamp()
    universal_run_name = config.config.stem if config.name is None else config.name
    universal_run_name += "-" + datetime.fromtimestamp(time_now).strftime("%Y%m%d_%H%M%S")
    logging.info(f"Universal name for run: {universal_run_name}")

    if config.enable_wandb and popdist.getInstanceIndex() == 0:
        wandb.init(
            entity=config.wandb_entity_name,
            project=config.wandb_project_name,
            name=universal_run_name,
            config=config,
            tags=config.wandb_tags,
        )

    # Get config parameters expected by HF model.
    bert_config = BertConfig(**config.bert_config.dict(), hidden_act=ipu.nn_ops.gelu)

    # Calculate the number of pipeline stages and the number of required IPUs per replica.
    num_pipeline_stages = len(config.ipu_config.pipeline_device_mapping)
    num_ipus_per_replica = max(config.ipu_config.pipeline_device_mapping) + 1

    # Load training data
    # ==================
    dataset, num_samples = get_pretraining_dataset(
        micro_batch_size=config.global_batch.micro_batch_size,
        dataset_dir=config.dataset_dir,
        max_seq_length=config.max_seq_length,
        max_predictions_per_seq=config.max_predictions_per_seq,
        vocab_size=bert_config.vocab_size,
        seed=config.seed,
        data_type=tf.float16,
        distributed_worker_count=popdist.getNumInstances(),
        distributed_worker_index=popdist.getInstanceIndex(),
        generated_dataset=config.generated_dataset,
    )

    task = get_pretraining_task(config.max_seq_length)
    batch_config = BatchConfig(
        micro_batch_size=config.global_batch.micro_batch_size,
        num_replicas=config.global_batch.replicas,
        gradient_accumulation_count=config.global_batch.grad_acc_steps_per_replica,
        num_pipeline_stages=num_pipeline_stages,
        dataset_size=num_samples,
        global_batches_per_log=config.global_batches_per_log,
        total_num_train_samples=config.total_num_train_samples,
        task=task,
    )

    # Create training strategy
    # ============
    strategy = create_ipu_strategy(
        num_ipus_per_replica=num_ipus_per_replica,
        num_replicas=config.global_batch.replicas,
        distributed_training=distributed_training,
        fp_exceptions=config.fp_exceptions,
        enable_recomputation=config.enable_recomputation,
        min_remote_tensor_size=config.min_remote_tensor_size,
        compile_only=config.compile_only,
    )

    set_random_seeds(config.seed)

    # Create model
    # ============
    policy = tf.keras.mixed_precision.Policy("float16")
    tf.keras.mixed_precision.set_global_policy(policy)

    with strategy.scope():
        model = IpuTFBertForPreTraining(config=bert_config)

        # Convert subclass model to functional, expand main layers to enable pipelining, and replace some layers to
        # optimise performance.
        model = convert_tf_bert_model(
            model,
            dataset,
            post_process_bert_input_layer,
            replace_layers=config.replace_layers,
            use_outlining=config.use_outlining,
            enable_recomputation=config.enable_recomputation,
            embedding_serialization_factor=config.embedding_serialization_factor,
            use_prediction_bias=config.use_prediction_bias,
            use_projection_bias=config.use_projection_bias,
            use_cls_layer=config.use_cls_layer,
            use_qkv_bias=config.use_qkv_bias,
            use_qkv_split=config.use_qkv_split,
        )

        # Load from pretrained checkpoint if requested.
        ckpt_found = False
        if config.pretrained_ckpt_path:
            logging.info("Attempting to load pretrained checkpoint from path " f"{config.pretrained_ckpt_path}")
            ckpt_found = load_checkpoint_into_model(model, config.pretrained_ckpt_path)

        if task == Task.PRETRAIN_PHASE_TWO and not ckpt_found:
            logging.warning(
                "WARNING: Phase 2 pre-training should be done from a completed Phase 1 checkpoint. "
                "Please specify the path to the Phase 1 checkpoint with 'pretrained_ckpt_path' in the "
                "config or as a command line argument."
            )

        # Configure pipeline stages
        # =========================
        if num_pipeline_stages > 1:
            pipeline_assigner = PipelineStagesAssigner(PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES)
            assignments = model.get_pipeline_stage_assignment()
            assignments = pipeline_assigner.assign_pipeline_stages(assignments, config.ipu_config.pipeline_stages)
            model.set_pipeline_stage_assignment(assignments)
            model.print_pipeline_stage_assignment_summary(print_fn=logging.info)
            poplar_options_per_pipeline_stage = get_poplar_options_per_pipeline_stage(
                num_ipus_per_replica,
                config.ipu_config.pipeline_device_mapping,
                config.ipu_config.matmul_available_memory_proportion_per_pipeline_stage,
                config.matmul_partials_type,
            )
            model.set_pipelining_options(
                gradient_accumulation_steps_per_replica=batch_config.gradient_accumulation_count,
                gradient_accumulation_reduction_method=GradientAccumulationReductionMethod.RUNNING_MEAN,
                pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Grouped,
                device_mapping=config.ipu_config.pipeline_device_mapping,
                offload_weight_update_variables=config.optimizer_state_offchip,
                replicated_optimizer_state_sharding=config.ipu_config.replicated_tensor_sharding,
                forward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
                backward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
                recomputation_mode=ipu.pipelining_ops.RecomputationMode.RecomputeAndBackpropagateInterleaved,
            )

        # Compile the model for training
        # ==============================
        # Wrap losses in an out-feed queue.
        nsp_loss = wrap_loss_in_enqueuer(NSPLossFunction, ["nsp_loss"])()
        mlm_loss = wrap_loss_in_enqueuer(MLMLossFunction, ["mlm_loss"])()

        # Prepare learning rate and wrap it in an out-feed queue.
        lr_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)
        lr_scheduler = get_lr_scheduler(
            scheduler_name=config.optimizer_opts.learning_rate.schedule_name,
            max_learning_rate=config.optimizer_opts.learning_rate.max_learning_rate,
            warmup_frac=config.optimizer_opts.learning_rate.warmup_frac,
            num_train_steps=batch_config.num_train_steps,
            queue=lr_outfeed_queue,
        )

        # Construct outfeed queues
        outfeed_queues = [lr_outfeed_queue, nsp_loss.outfeed_queue, mlm_loss.outfeed_queue]

        # Prepare accuracy and wrap it in an out-feed queue.
        if config.show_accuracy:
            pretraining_accuracy = wrap_stateless_metric_in_enqueuer(
                "accuracy", pretraining_accuracy_fn, ["mlm_acc_average", "nsp_acc_average"]
            )
            outfeed_queues.append(pretraining_accuracy.outfeed_queue)

        # Prepare optimizer.
        outline_optimizer_apply_gradients = (
            False if config.ipu_config.replicated_tensor_sharding else config.use_outlining
        )
        optimizer = get_optimizer(
            optimizer_name=config.optimizer_opts.name,
            num_replicas=popdist.getNumTotalReplicas() if distributed_training else config.global_batch.replicas,
            learning_rate_schedule=lr_scheduler,
            use_outlining=outline_optimizer_apply_gradients,
            loss_scaling=config.optimizer_opts.loss_scaling,
            weight_decay_rate=config.optimizer_opts.weight_decay_rate,
        )
        # Compile the model.
        model.compile(
            optimizer=optimizer,
            loss={"nsp___cls": nsp_loss, "mlm___cls": mlm_loss},
            steps_per_execution=batch_config.steps_per_execution,
            metrics=[pretraining_accuracy if config.show_accuracy else None],
        )
        # Train the model
        # ===============
        # Set up callbacks
        callbacks = CallbackFactory.get_callbacks(
            universal_run_name=universal_run_name,
            batch_config=batch_config,
            model=model,
            checkpoint_path=config.save_ckpt_path,
            ckpt_every_n_steps_per_execution=config.ckpt_every_n_steps_per_execution,
            outfeed_queues=outfeed_queues,
            distributed_training=distributed_training,
            enable_wandb=config.enable_wandb,
        )
        # Print configs to be logged in wandb's terminal.
        logging.info(f"Application config:\n{config}")
        logging.info(f"Training batch config:\n{batch_config}")
        # Train the model
        # In order to achieve a specific number of steps, we set the number of
        # epochs to 1 and the steps per epoch to the number of steps we require.
        logging.info(
            "Forcing `model.fit` to run a particular number of steps by"
            " running a single 'epoch' with the number of steps we"
            " require. This allows running a fraction of actual"
            " epochs."
        )
        # Set verbose to 0 so the default progress bar, which is unreliable
        # with `steps_per_execution > 1`, is hidden in favour of using a
        # logging callback included in callbacks dir.
        history = model.fit(
            dataset,
            steps_per_epoch=batch_config.total_num_micro_batches_per_instance,
            epochs=1,
            callbacks=callbacks,
            verbose=0,
        )
        return history


if __name__ == "__main__":
    # Combine arguments and config file
    parser = argparse.ArgumentParser(
        description="TF2 BERT Pretraining", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_shared_arguments(parser)
    args = parser.parse_args()
    cfg = combine_config_file_with_args(args, PretrainingOptions)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=cfg.logging, datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Prevent doubling of TF logs.
    tf.get_logger().propagate = False

    # Run pretraining
    pretrain(cfg)
