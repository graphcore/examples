# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import import_helper

from args import parse_bert_args
from datasets import load_dataset
from examples_tests.execute_once_per_fs import ExecuteOncePerFS
from ipu_options import get_options
from model_export import export_model
from modeling import PipelinedBertForQuestionAnswering
import numpy as np
import poptorch
import pytest
from squad_data import prepare_validation_features
import transformers
from transformers import default_data_collator
from triton_server.server_setup import *
from utils import logger

test_configs = {
    'bert': 'squad_large_384'
}


class Empty:
    pass


@pytest.fixture(scope="session")
def configure_bert_model():
    model = Empty()

    pargs = ('--config', "squad_large_384", "--dataset", "generated")
    model.config = transformers.BertConfig(
        **(vars(parse_bert_args(args=pargs, config_file="configs_squad.yml"))))
    if not model.config.pretrained_checkpoint:
        logger(
            "[warning] --pretrained-checkpoint was not specified; training with uninitialized BERT...")
    # Warnings for configs where embeddings may not fit
    if model.config.embedding_serialization_factor == 1:
        if model.config.replication_factor == 1:
            logger("[warning] With replication_factor == 1 you may need to set "
                   "embedding_serialization_factor > 1 for the model to fit")
        elif not model.config.replicated_tensor_sharding:
            logger("[warning] With replicated_tensor_sharding=False you may need to set "
                   "embedding_serialization_factor > 1 for the model to fit")

    model.opts = get_options(model.config)
    model.opts.outputMode(poptorch.OutputMode.All)

    squad_v2 = model.config.squad_v2
    dataset_name = "squad_v2" if squad_v2 else "squad"
    model.config.dataset = dataset_name

    logger("Loading Dataset {} ...".format(dataset_name))
    datasets = load_dataset(dataset_name)

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
    if model.config.pretrained_checkpoint:
        model.model_ipu = PipelinedBertForQuestionAnswering.from_pretrained(
            model.config.pretrained_checkpoint, config=model.config).parallelize().half()
    else:
        model.model_ipu = PipelinedBertForQuestionAnswering(
            model.config).parallelize().half()

    model.config.micro_batch_size = 2
    model.config.device_iterations = 16
    model.config.gradient_accumulation = 1
    model.config.replication_factor = 1

    model.config.batch_size = model.config.device_iterations * model.config.micro_batch_size * \
        model.config.gradient_accumulation * model.config.replication_factor

    model.opts = get_options(model.config)
    model.opts.outputMode(poptorch.OutputMode.All)
    model.val_dl = poptorch.DataLoader(model.opts,
                                       validation_features.remove_columns(
                                           ['example_id', 'offset_mapping']),
                                       batch_size=model.config.micro_batch_size,
                                       shuffle=False,
                                       drop_last=False,
                                       collate_fn=default_data_collator)
    model.datasets = datasets
    model.validation_features = validation_features
    yield model


@pytest.fixture(scope="session", autouse=True)
@ExecuteOncePerFS(lockfile=str(Path(__file__).parent.absolute()) + "/test_environment_ready.lock",
                  file_list=[], timeout=120, retries=20)
def initialize_test_environment(request, configure_bert_model):
    export_model(request, configure_bert_model, "bert")


@pytest.fixture(scope="module")
def poptorch_ref_model(request, triton_server, configure_bert_model):
    benchmark_only = request.config.getoption(benchmark_opt)
    # model name should be parametrized when new model will be added
    model_name = 'bert'
    if not benchmark_only:
        logger("Compiling Inference Model...")
        popef_file = triton_server.model_repo_path + \
            '/' + model_name + '/1/executable.popef'
        if not os.path.exists(popef_file):
            pytest.fail("Popef file: " + popef_file + " doesn't exist!")

        configure_bert_model.model_ipu.eval()
        ref_model = poptorch.inferenceModel(
            configure_bert_model.model_ipu, configure_bert_model.opts)
        ref_model.loadExecutable(popef_file)
    else:
        ref_model = None
    yield ref_model
