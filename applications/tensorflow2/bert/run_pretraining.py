# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.python import ipu
from transformers import BertConfig

from data_utils.batch_config import BatchConfig, Task
from data_utils.wikipedia.load_wikipedia_data import get_pretraining_dataset, get_dataset_files_count
from keras_extensions.callbacks.callback_factory import CallbackFactory
from keras_extensions.learning_rate.scheduler_builder import get_lr_scheduler
from keras_extensions.optimization import get_optimizer
from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.ipu_pretraining_model import IpuTFBertForPreTraining
from model.losses import MLMLossFunction, NSPLossFunction
from model.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from utilities.argparser import parse_arguments
from utilities.assign_pipeline_stages import PipelineStagesAssigner
from utilities.checkpoint_utility import load_checkpoint_into_model
from utilities.ipu_utils import create_ipu_strategy, get_poplar_options_per_pipeline_stage, set_random_seeds
from utilities.loss_enqueuer import wrap_loss_in_enqueuer


def pretrain(**config):
    # Get required options
    micro_batch_size = config["micro_batch_size"]
    replicas = config["replicas"]
    grad_acc_steps_per_replica = config["grad_acc_steps_per_replica"]
    optimizer_opts = config["optimizer_opts"]
    use_outlining = config["use_outlining"]
    replicated_tensor_sharding = config["replicated_tensor_sharding"]
    fp_exceptions = config["fp_exceptions"]
    bert_config = config["bert_config"]
    wandb_opts = config["wandb_opts"]
    pipeline_stages = config["pipeline_stages"]
    device_mapping = config["device_mapping"]

    # Get optional options
    total_num_train_samples = config.get("total_num_train_samples", None)
    save_ckpt_path = config.get("save_ckpt_path",
                                Path(__file__).parent.joinpath("checkpoints").absolute())
    pretrained_ckpt_path = config.get("pretrained_ckpt_path", None)
    ckpt_every_n_steps_per_execution = config.get("ckpt_every_n_steps_per_execution",
                                                  2000)

    universal_run_name = config.get(
        "name", f"{Path(config['config']).stem}-{wandb_opts['init']['name']}")
    universal_run_name += f"-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Universal name for run: {universal_run_name}")

    set_random_seeds(config["seed"])
    num_pipeline_stages = max(device_mapping) + 1
    num_ipus = replicas * num_pipeline_stages

    bert_config = BertConfig(**bert_config, hidden_act=ipu.nn_ops.gelu)

    dataset, filenames = get_pretraining_dataset(micro_batch_size=micro_batch_size,
                                                 dataset_dir=config["dataset_dir"],
                                                 max_seq_length=bert_config.max_seq_length,
                                                 max_predictions_per_seq=bert_config.max_predictions_per_seq,
                                                 distributed_worker_count=1,
                                                 seed=config["seed"],
                                                 data_type=tf.float16)
    num_samples = get_dataset_files_count(filenames)
    if bert_config.max_seq_length == 128:
        task = Task.PRETRAIN_PHASE_ONE
    elif bert_config.max_seq_length == 384:
        task = Task.PRETRAIN_PHASE_TWO
    else:
        raise ValueError("Sequence length must be 128 or 384")
    batch_config = BatchConfig(micro_batch_size=micro_batch_size,
                               num_replicas=replicas,
                               gradient_accumulation_count=grad_acc_steps_per_replica,
                               dataset_size=num_samples,
                               global_batches_per_log=config["global_batches_per_log"],
                               total_num_train_samples=total_num_train_samples,
                               task=task)

    policy = tf.keras.mixed_precision.Policy("float16")
    tf.keras.mixed_precision.set_global_policy(policy)

    strategy = create_ipu_strategy(num_ipus,
                                   fp_exceptions=fp_exceptions,
                                   enable_recomputation=config["enable_recomputation"],
                                   min_remote_tensor_size=config["min_remote_tensor_size"])
    with strategy.scope():
        model = IpuTFBertForPreTraining(config=bert_config)

        # Convert subclass model to functional, expand main layers to enable pipelining, and replace some layers to
        # optimise performance.
        model = convert_tf_bert_model(
            model,
            dataset,
            post_process_bert_input_layer,
            replace_layers=config["replace_layers"],
            use_outlining=use_outlining,
            embedding_serialization_factor=config["embedding_serialization_factor"]
        )

        # Load from pretrained checkpoint if requested.
        if pretrained_ckpt_path:
            print("Attempting to load pretrained checkpoint from"
                  f" path {pretrained_ckpt_path}")
            load_checkpoint_into_model(model, pretrained_ckpt_path)
        else:
            if task == Task.PRETRAIN_PHASE_TWO:
                print("WARNING: Phase 2 pre-training should be done from a completed Phase 1 checkpoint. "
                      "Please specify the path to the Phase 1 checkpoint with 'pretrained_ckpt_path' in the config or "
                      "as a command line argument.")

        if num_pipeline_stages > 1:
            # Configure pipeline stages.
            pipeline_assigner = PipelineStagesAssigner(PIPELINE_ALLOCATE_PREVIOUS,
                                                       PIPELINE_NAMES)
            assignments = model.get_pipeline_stage_assignment()
            assignments = pipeline_assigner.assign_pipeline_stages(assignments,
                                                                   pipeline_stages)
            model.set_pipeline_stage_assignment(assignments)
            model.print_pipeline_stage_assignment_summary()
            poplar_options_per_pipeline_stage = get_poplar_options_per_pipeline_stage(
                num_pipeline_stages,
                device_mapping,
                config["matmul_available_memory_proportion_per_pipeline_stage"],
                config["matmul_partials_type"])
            model.set_pipelining_options(
                gradient_accumulation_steps_per_replica=batch_config.gradient_accumulation_count,
                pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Grouped,
                device_mapping=device_mapping,
                offload_weight_update_variables=config["optimizer_state_offchip"],
                replicated_optimizer_state_sharding=replicated_tensor_sharding,
                forward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
                backward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
                recomputation_mode=ipu.pipelining_ops.RecomputationMode.RecomputeAndBackpropagateInterleaved,
            )

        # Prepare losses and wrap them in an out-feed queue.
        nsp_loss_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
        nsp_loss = wrap_loss_in_enqueuer(NSPLossFunction,
                                         nsp_loss_outfeed_queue,
                                         ["nsp_loss_average"])()
        mlm_loss_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
        mlm_loss = wrap_loss_in_enqueuer(MLMLossFunction,
                                         mlm_loss_outfeed_queue,
                                         ["mlm_loss_average"])()
        # Prepare learning rate and wrap it in an out-feed queue.
        config['learning_rate']['lr_schedule_params']['total_steps'] = batch_config.num_train_steps
        lr_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)
        lr_scheduler = get_lr_scheduler(scheduler_name=config["learning_rate"]["lr_schedule"],
                                        schedule_params=config["learning_rate"]["lr_schedule_params"],
                                        queue=lr_outfeed_queue)
        # Prepare optimizer.
        outline_optimizer_apply_gradients = False if replicated_tensor_sharding else use_outlining
        optimizer = get_optimizer(
            optimizer_opts["name"],
            grad_acc_steps_per_replica,
            replicas,
            lr_scheduler,
            outline_optimizer_apply_gradients,
            loss_scaling=config["loss_scaling"],
            weight_decay_rate=optimizer_opts["params"]["weight_decay_rate"],
        )
        # Compile the model.
        model.compile(
            optimizer=optimizer,
            loss={"nsp___cls": nsp_loss,
                  "mlm___cls": mlm_loss},
            steps_per_execution=batch_config.steps_per_execution,
        )
        # Set up callbacks
        callbacks = CallbackFactory.get_callbacks(
            universal_run_name=universal_run_name,
            batch_config=batch_config,
            model=model,
            checkpoint_path=save_ckpt_path,
            ckpt_every_n_steps_per_execution=ckpt_every_n_steps_per_execution,
            outfeed_queues=[lr_outfeed_queue,
                            nsp_loss_outfeed_queue,
                            mlm_loss_outfeed_queue],
            config=config,
        )
        # Print configs to be logged in wandb's terminal.
        print(config)
        print(f"Training batch config:\n{batch_config}")
        # Train the model
        # In order to achieve a specific number of steps, we set the number of
        # epochs to 1 and the steps per epoch to the number of steps we require.
        print("Forcing `model.fit` to run a particular number of steps by"
              " running a single 'epoch' with the number of steps we"
              " require. This allows running a fraction of actual"
              " epochs.")
        history = model.fit(dataset,
                            steps_per_epoch=batch_config.total_num_micro_batches,
                            epochs=1,
                            callbacks=callbacks)
        return history


if __name__ == "__main__":
    pretrain(**parse_arguments("TF2 BERT Pretraining"))
