# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2020 The HuggingFace Team All rights reserved.
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
#
# This file has been modified by Graphcore Ltd.

from datetime import datetime
from pathlib import Path

from datasets import load_metric
import tensorflow as tf
from tensorflow.python import ipu
from transformers import BertConfig, TFBertForQuestionAnswering

from data_utils.batch_config import BatchConfig, Task
from data_utils.squad_v1.load_squad_data import format_raw_data_for_metric, get_prediction_dataset, get_squad_data
from data_utils.squad_v1.postprocess_squad_predictions import postprocess_qa_predictions
from keras_extensions.callbacks.callback_factory import CallbackFactory
from keras_extensions.learning_rate.scheduler_builder import get_lr_scheduler
from keras_extensions.optimization import get_optimizer
from model.losses import QuestionAnsweringLossFunction
from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from utilities.argparser import parse_arguments
from utilities.assign_pipeline_stages import PipelineStagesAssigner
from utilities.checkpoint_utility import load_checkpoint_into_model
from utilities.ipu_utils import create_ipu_strategy, get_poplar_options_per_pipeline_stage, set_random_seeds
from utilities.loss_enqueuer import wrap_loss_in_enqueuer


def fine_tune_squad(**config):
    # Get checkpoint from Hugging Face's models repo or from our own checkpoint.
    bert_model_name = config.get("bert_model_name", None)
    pretrained_ckpt_path = config.get("pretrained_ckpt_path", None)  # This can be passed from both cli and config file.
    assert bert_model_name is not None or pretrained_ckpt_path is not None, \
        "SQuAD requires a pretrained model, either `bert_model_name` (via config file) or `pretrained_ckpt_path` " \
        "(via config file or `--pretrained-ckpt-path` command line argument) but none provided."
    assert (bert_model_name is not None and pretrained_ckpt_path is None) \
        or (bert_model_name is None and pretrained_ckpt_path is not None), \
        f"Only one checkpoint is accepted, but two provided: `bert_model_name`={bert_model_name}, " \
        f"and `pretrained_ckpt_path`={pretrained_ckpt_path}."
    if pretrained_ckpt_path is not None:
        bert_config_params = config["bert_config"]

    # Get required options
    micro_batch_size = config["micro_batch_size"]
    num_epochs = config["num_epochs"]
    optimizer_opts = config["optimizer_opts"]
    learning_rate = config['learning_rate']
    replicas = config["replicas"]
    grad_acc_steps_per_replica = config["grad_acc_steps_per_replica"]
    wandb_opts = config["wandb_opts"]
    use_outlining = config["use_outlining"]
    replace_layers = config["replace_layers"]
    enable_recomputation = config["enable_recomputation"]
    embedding_serialization_factor = config["embedding_serialization_factor"]
    optimizer_state_offchip = config["optimizer_state_offchip"]
    matmul_available_memory_proportion_per_pipeline_stage = config[
        "matmul_available_memory_proportion_per_pipeline_stage"]
    matmul_partials_type = config["matmul_partials_type"]
    pipeline_stages = config["pipeline_stages"]
    device_mapping = config["device_mapping"]
    global_batches_per_log = config["global_batches_per_log"]
    seed = config["seed"]
    cache_dir = config["cache_dir"]
    output_dir = config["output_dir"]

    # Get optional options
    save_ckpt_path = config.get("save_ckpt_path", Path(__file__).parent.joinpath("checkpoints").absolute())
    ckpt_every_n_steps_per_execution = config.get("ckpt_every_n_steps_per_execution", 2000)

    universal_run_name = config.get("name", f"{Path(config['config']).stem}-{wandb_opts['init']['name']}")
    universal_run_name += f"-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Universal name for run: {universal_run_name}")
    set_random_seeds(seed)
    num_pipeline_stages = len(device_mapping)
    num_ipus_per_replicas = max(device_mapping) + 1
    num_ipus = replicas * num_ipus_per_replicas

    # Load training and validation data
    # =================================
    train_dataset, eval_dataset, num_train_samples, num_eval_samples, raw_datasets = get_squad_data(
        micro_batch_size,
        cache_dir
    )
    train_batch_config = BatchConfig(micro_batch_size=micro_batch_size,
                                     total_num_train_samples=num_epochs * num_train_samples.numpy(),
                                     num_replicas=replicas,
                                     gradient_accumulation_count=grad_acc_steps_per_replica,
                                     dataset_size=num_train_samples.numpy(),
                                     global_batches_per_log=global_batches_per_log,
                                     task=Task.OTHER)

    # Create model
    # ============
    policy = tf.keras.mixed_precision.Policy("float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    strategy = create_ipu_strategy(num_ipus, enable_recomputation=enable_recomputation)
    with strategy.scope():
        # Instantiate the pretrained model given in the config.
        if bert_model_name is not None:
            model = TFBertForQuestionAnswering.from_pretrained(bert_model_name)
        else:
            bert_config = BertConfig(**bert_config_params, hidden_act=ipu.nn_ops.gelu)
            model = TFBertForQuestionAnswering(config=bert_config)

        # Convert subclass model to functional, expand main layers to enable pipelining, and replace some layers to
        # optimise performance.
        model = convert_tf_bert_model(
            model,
            train_dataset,
            post_process_bert_input_layer,
            replace_layers=replace_layers,
            use_outlining=use_outlining,
            embedding_serialization_factor=embedding_serialization_factor,
            rename_outputs={'tf.compat.v1.squeeze': 'start_positions', 'tf.compat.v1.squeeze_1': 'end_positions'}
        )
        # Load from pretrained checkpoint if requested.
        if pretrained_ckpt_path is not None:
            print(f"Attempting to load pretrained checkpoint from path {pretrained_ckpt_path}. "
                  f"This will overwrite the current weights")
            load_checkpoint_into_model(model, pretrained_ckpt_path)

        # Configure pipeline stages
        # =========================
        if num_pipeline_stages > 1:
            pipeline_assigner = PipelineStagesAssigner(PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES)
            assignments = model.get_pipeline_stage_assignment()
            assignments = pipeline_assigner.assign_pipeline_stages(assignments, pipeline_stages)
            model.set_pipeline_stage_assignment(assignments)
            model.print_pipeline_stage_assignment_summary()
            poplar_options_per_pipeline_stage = get_poplar_options_per_pipeline_stage(
                num_ipus_per_replicas,
                device_mapping,
                matmul_available_memory_proportion_per_pipeline_stage,
                matmul_partials_type
            )
            model.set_pipelining_options(
                gradient_accumulation_steps_per_replica=grad_acc_steps_per_replica,
                pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Grouped,
                device_mapping=device_mapping,
                offload_weight_update_variables=optimizer_state_offchip,
                forward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
                backward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
                recomputation_mode=ipu.pipelining_ops.RecomputationMode.RecomputeAndBackpropagateInterleaved,
            )

        # Compile the model for training
        # ==============================
        # Wrap loss in an out-feed queue.
        loss_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
        qa_loss = wrap_loss_in_enqueuer(QuestionAnsweringLossFunction,
                                        loss_outfeed_queue,
                                        ["end_positions_loss", "start_positions_loss"])()
        # Define optimiser with polynomial decay learning rate.
        learning_rate['lr_schedule_params']['total_steps'] = train_batch_config.num_train_steps
        lr_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)
        lr_scheduler = get_lr_scheduler(scheduler_name=learning_rate["lr_schedule"],
                                        schedule_params=learning_rate["lr_schedule_params"],
                                        queue=lr_outfeed_queue)
        # Prepare optimizer.
        outline_optimizer_apply_gradients = use_outlining
        optimizer = get_optimizer(
            optimizer_opts["name"],
            grad_acc_steps_per_replica,
            replicas,
            lr_scheduler,
            outline_optimizer_apply_gradients,
            weight_decay_rate=optimizer_opts["params"]["weight_decay_rate"],
        )
        # Compile the model.
        model.compile(
            optimizer=optimizer,
            loss={"end_positions": qa_loss, "start_positions": qa_loss},
            metrics='accuracy',
            steps_per_execution=train_batch_config.steps_per_execution
        )

        # Train the model
        # ===============
        # Set up callbacks
        callbacks = CallbackFactory.get_callbacks(
            universal_run_name=universal_run_name,
            batch_config=train_batch_config,
            model=model,
            checkpoint_path=save_ckpt_path,
            ckpt_every_n_steps_per_execution=ckpt_every_n_steps_per_execution,
            outfeed_queues=[lr_outfeed_queue, loss_outfeed_queue],
            config=config,
        )
        # Print configs to be logged in wandb's terminal.
        print(config)
        print(f"Training batch config:\n{train_batch_config}")
        # Train the model
        history = model.fit(
            train_dataset,
            steps_per_epoch=train_batch_config.num_micro_batches_per_epoch,
            epochs=num_epochs,
            callbacks=callbacks
        )

    # Evaluate the model on the validation set
    # ========================================
    # Prepare the dataset to be evaluated in the IPU.
    eval_batch_config = BatchConfig(micro_batch_size=micro_batch_size,
                                    total_num_train_samples=num_eval_samples.numpy(),
                                    num_replicas=replicas,
                                    gradient_accumulation_count=grad_acc_steps_per_replica,
                                    dataset_size=num_eval_samples.numpy(),
                                    task=Task.OTHER)
    max_eval_samples = eval_batch_config.micro_batch_size * eval_batch_config.num_micro_batches_per_epoch
    eval_pred_dataset = get_prediction_dataset(eval_dataset, max_eval_samples)
    with strategy.scope():
        # Re-compile the model for prediction if needed.
        if train_batch_config.steps_per_execution != eval_batch_config.steps_per_execution:
            model.compile(steps_per_execution=eval_batch_config.steps_per_execution)
        # Get predictions for the validation data.
        print(f"Running inference:\nGenerating predictions on the validation data...")
        predictions = model.predict(
            eval_pred_dataset,
            batch_size=eval_batch_config.micro_batch_size
        )
    # The predictions for the end position goes first in the model outputs tuple (note the output of model.summary()).
    end_predictions, start_predictions = predictions
    # Match the predictions to answers in the original context.
    # This will also write out the predictions to a json file in the directory given by `output_dir`.
    final_predictions = postprocess_qa_predictions(
        list(raw_datasets["validation"].as_numpy_iterator()),
        list(eval_dataset.unbatch().as_numpy_iterator()),
        (start_predictions, end_predictions),
        output_dir=output_dir
    )
    # Format the predictions and the actual labels as expected by the metric.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    formatted_labels = format_raw_data_for_metric(raw_datasets["validation"])
    metric = load_metric("squad")
    metrics = metric.compute(predictions=formatted_predictions, references=formatted_labels)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")

    return history


if __name__ == "__main__":
    if __name__ == "__main__":
        fine_tune_squad(**parse_arguments("TF2 BERT SQuAD Fine Tuning"))
