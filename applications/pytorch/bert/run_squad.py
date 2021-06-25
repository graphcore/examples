# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import poptorch
import transformers
from transformers import default_data_collator
from modeling import PipelinedBertForQuestionAnswering
from optimization import get_lr_scheduler, get_optimizer
from ipu_options import get_options
from datasets import load_dataset, load_metric
from squad_data import PadCollate, prepare_train_features, prepare_validation_features, postprocess_qa_predictions
from utils import logger, parse_bert_args


def main():
    config = transformers.BertConfig(**(vars(parse_bert_args())))
    if not config.checkpoint_file:
        logger("[warning] --checkpoint-file was not specified; training with uninitialized BERT...")
    # Warnings for configs where embeddings may not fit
    if config.embedding_serialization_factor == 1:
        if config.replication_factor == 1:
            logger("[warning] With replication_factor == 1 you may need to set "
                   "embedding_serialization_factor > 1 for the model to fit")
        elif not config.replicated_tensor_sharding:
            logger("[warning] With replicated_tensor_sharding=False you may need to set "
                   "embedding_serialization_factor > 1 for the model to fit")
    samples_per_step = config.batches_per_step * config.batch_size * \
        config.gradient_accumulation * config.replication_factor
    do_training = config.squad_do_training
    do_validation = config.squad_do_validation
    opts = get_options(config)

    logger("Loading Dataset...")
    datasets = load_dataset("squad")
    train_dataset = datasets["train"]

    # Create train features from dataset
    logger("Tokenizing Train Dataset...")
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=True,
    )

    # Create validation features from dataset
    logger("Tokenizing Validation Dataset...")
    validation_features = datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        num_proc=1,
        remove_columns=datasets["validation"].column_names,
        load_from_cache_file=True,
    )

    # Create the model
    model_ipu = PipelinedBertForQuestionAnswering(config).half()

    # Load pretrained checkpoint
    if config.checkpoint_file:
        checkpoint = torch.load(config.checkpoint_file)
        if config.embedding_serialization_factor:
            word_embedding_weight = checkpoint["model_state_dict"]["bert.embeddings.word_embeddings.weight"]
            split_size = word_embedding_weight.shape[0] // config.embedding_serialization_factor
            for i in range(config.embedding_serialization_factor):
                checkpoint["model_state_dict"][f"bert.embeddings.word_embeddings.split_embeddings.{i}.weight"] = word_embedding_weight[i*split_size:(i+1)*split_size, :]
        model_ipu.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if do_training:
        train_dl = poptorch.DataLoader(opts,
                                       train_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       drop_last=False,
                                       collate_fn=PadCollate(samples_per_step,
                                                             {"input_ids": 0,
                                                              "attention_mask": 0,
                                                              "token_type_ids": 0,
                                                              "start_positions": config.sequence_length,
                                                              "end_positions": config.sequence_length}))
        optimizer = get_optimizer(config, model_ipu)
        model_ipu.train()
        training_model = poptorch.trainingModel(model_ipu, opts, optimizer)

        sample_batch = next(iter(train_dl))
        logger("Compiling Model...")
        start_compile = time.perf_counter()
        training_model.compile(sample_batch["input_ids"],
                               sample_batch["attention_mask"],
                               sample_batch["token_type_ids"],
                               sample_batch["start_positions"],
                               sample_batch["end_positions"])

        duration_compilation = time.perf_counter() - start_compile
        logger(f"Compiled/Loaded model in {duration_compilation} secs")

        # Train
        factor = config.gradient_accumulation * config.batches_per_step
        scheduler = get_lr_scheduler(optimizer, "linear", config.lr_warmup, config.num_epochs * len(train_dl))
        logger("Training...")
        for epoch in range(config.num_epochs):
            for step, batch in enumerate(train_dl):
                start_step = time.perf_counter()
                outputs = training_model(batch["input_ids"],
                                         batch["attention_mask"],
                                         batch["token_type_ids"],
                                         batch["start_positions"],
                                         batch["end_positions"])

                scheduler.step()
                training_model.setOptimizer(optimizer)
                step_length = time.perf_counter() - start_step
                step_throughput = samples_per_step / step_length
                loss = outputs[0].div(factor).mean().item()
                logger(f"Epoch: {epoch}, Step:{step}, LR={scheduler.get_last_lr()[0]:.2e}, loss={loss:3.3f}, throughput={step_throughput:3.3f} samples/s")
        training_model.detachFromDevice()

    if do_validation:
        config.batch_size = 2
        config.batches_per_step = 16
        config.gradient_accumulation = 1
        config.replication_factor = 1
        samples_per_step = config.batches_per_step * config.batch_size * \
            config.gradient_accumulation * config.replication_factor
        opts = get_options(config)
        opts.anchorMode(poptorch.AnchorMode.All)
        val_dl = poptorch.DataLoader(opts,
                                     validation_features.remove_columns(
                                         ['example_id', 'offset_mapping']),
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     collate_fn=default_data_collator)
        raw_predictions = [[], []]
        model_ipu.eval()
        inference_model = poptorch.inferenceModel(model_ipu, opts)
        sample_batch = next(iter(val_dl))
        logger("Compiling Inference Model...")
        inference_model.compile(sample_batch["input_ids"],
                                sample_batch["attention_mask"],
                                sample_batch["token_type_ids"])
        logger("Validating...")
        for step, batch in enumerate(val_dl):
            start_step = time.perf_counter()
            outputs = inference_model(batch["input_ids"],
                                      batch["attention_mask"],
                                      batch["token_type_ids"])
            step_length = time.perf_counter() - start_step
            step_throughput = samples_per_step / step_length
            raw_predictions[0].append(outputs[0])
            raw_predictions[1].append(outputs[1])
            logger(f"Step:{step}, throughput={step_throughput} samples/s")

        raw_predictions[0] = torch.vstack(raw_predictions[0]).float().numpy()
        raw_predictions[1] = torch.vstack(raw_predictions[1]).float().numpy()
        final_predictions = postprocess_qa_predictions(datasets["validation"],
                                                       validation_features,
                                                       raw_predictions)
        metric = load_metric("squad")
        formatted_predictions = [{"id": k, "prediction_text": v}
                                 for k, v in final_predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]}
                      for ex in datasets["validation"]]
        print(metric.compute(predictions=formatted_predictions, references=references))


if __name__ == "__main__":
    main()
