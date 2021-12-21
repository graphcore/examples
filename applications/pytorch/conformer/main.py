# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import fire
import pprint
import yaml
import torch
import wandb
import logging
import popart
import poptorch
import numpy as np
from src.conformer_encoder import ConformerEncoder
from src.transformer_decoder import TransformerDecoder
from src.global_cmvn import GlobalMVN
from src.label_smoothing_loss import LabelSmoothingLoss, CrossEntropyLoss
from src.utils.pipeline_wrapper import PipelineWrapper
from src.conformer import Conformer
from src.utils.initializer import initialize
from src.iterator.vocab import Vocab
from src.iterator.dataset import AishellDataset
from src.iterator.dataset import CollateFn
from src.trainer import Trainer
from src.utils.lr_scheduler import WarmupLR
from src.utils.checkpoint import CheckPoint


def load_yaml(yaml_file):
    args = yaml.safe_load(open(yaml_file, 'r'))
    return args


class Workflow:
    def __init__(self, config_file='configs/train.yaml', **kwargs):
        self.config_file = config_file
        self.args = load_yaml(self.config_file)
        self.parse_args(kwargs)
        self.ipu_options = None
        self.vocab = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_iterator = None
        self.val_iterator = None
        self.model = None
        self.trainer = None
        self.checkpoint = None

    def parse_args(self, kwargs):
        for key, value in kwargs.items():
            try:
                module_name, config_name = key.split('.')
                self.args[module_name][config_name] = value
            except:
                raise KeyError(f'{key} is not found in {self.config_file}.')

    def build_optimizer(self):
        accum_type = torch.float16 if self.dtype == 'FLOAT16' else torch.float32
        beta2 = 0.999 if self.dtype == 'FLOAT16' else 0.98
        self.optimizer = poptorch.optim.Adam(
            self.model.parameters(), betas=(0.9, beta2), accum_type=accum_type, **self.args['optimizer']
        )
        adim = self.args['encoder']['output_size']
        self.scheduler = WarmupLR(self.optimizer, adim, **self.args['scheduler'])

    def build_wandb(self):
        if not self.args['wandb']['name']:
            self.wandb = None
        else:
            wandb.init(project=self.args['wandb']['name'], settings=wandb.Settings(console='off'))
            wandb.config.update(self.args)
            self.wandb = wandb

    def build_logger(self):
        """ build logger both for print in terminal and log.txt """
        logger_args = self.args['logger']
        log_level = {'info': logging.INFO, 'debug': logging.DEBUG}.get(
            logger_args['level'], logging.INFO
        )
        self.logger = logging.getLogger(logger_args['name'])
        self.logger.setLevel(level=log_level)
        formatter = logging.Formatter(
            '%(asctime)s %(filename)s line:%(lineno)d %(levelname)s %(message)s'
        )
        handler = logging.FileHandler(logger_args['log_file'])
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def build_ipu_options(self, is_train=True):
        ipu_args = self.args['ipu_options']
        if is_train:
            self.ipu_options = poptorch.Options()
            self.ipu_options.replicationFactor(ipu_args['num_replicas'])
            self.ipu_options.autoRoundNumIPUs(True)
            self.ipu_options.deviceIterations(ipu_args['batches_per_step'])
            self.ipu_options.Training.gradientAccumulation(ipu_args['gradient_accumulation'])
            self.ipu_options.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
            self.ipu_options.outputMode(poptorch.OutputMode.All)
            self.ipu_options.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))
            self.ipu_options.setAvailableMemoryProportion({f'IPU{i}': ipu_args['available_memory_proportion'] for i in range(len(self.args['pipeline']) + 1)})
            self.ipu_options.TensorLocations.setOptimizerLocation(
                poptorch.TensorLocationSettings()
                .useOnChipStorage(not ipu_args['optimizer_state_offchip'])
                .useReplicatedTensorSharding(ipu_args['replicated_tensor_sharding'])
            )

            if ipu_args['executable_cache_dir']: self.ipu_options.enableExecutableCaching(ipu_args['executable_cache_dir'])
            if ipu_args['enable_stochastic_rounding']: self.ipu_options.Precision.enableStochasticRounding(True)
            if ipu_args['enable_half_partials']: self.ipu_options.Precision.setPartialsType(torch.float16)
            if ipu_args['enable_synthetic_data']: self.ipu_options.enableSyntheticData(True)
            # Options for profiling with Popvision
            if ipu_args['enable_profiling']:
                engine_options = {
                    "opt.useAutoloader": "true",
                    "target.syncReplicasIndependently": "true",
                    "debug.allowOutOfMemory": "true",
                    "profiler.format": "v3",
                    "autoReport.all": "true",
                    "autoReport.executionProfileProgramRunCount": "2",
                }
                self.ipu_options._Popart.set("engineOptions", engine_options)

            # PopART performance options #
            # Enable recomputation of operations in the graph in the backwards pass to reduce model size at the cost of computation cycles
            self.ipu_options._Popart.set('autoRecomputation', 3)
            # Only stream needed tensors back to host
            self.ipu_options._Popart.set('disableGradAccumulationTensorStreams', True)
            # The incremental value that a sub-graph requires, relative to its nested sub-graphs (if any), to be eligible for outlining
            self.ipu_options._Popart.set('outlineThreshold', 10.0)
            # Set the execution strategy to use to partition the graph.
            self.ipu_options.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
            # Parallelize optimizer step update across IPUs
            self.ipu_options._Popart.set('accumulateOuterFragmentSettings.schedule', int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
            self.ipu_options._Popart.set('accumulateOuterFragmentSettings.excludedVirtualGraphs', ['0'])
            # Enable patterns for better throughput and memory reduction
            self.ipu_options._Popart.set('subgraphCopyingStrategy', int(popart.SubgraphCopyingStrategy.JustInTime))
            # When #shouldDelayVarUpdates is true, the other ops in the proximity of the delayed var updates may inherit the -inf schedule priority used to delay the var updates.
            self.ipu_options._Popart.set('scheduleNonWeightUpdateGradientConsumersEarly', True)
            self.ipu_options._Popart.setPatterns({'TiedGather': True, 'TiedGatherAccumulate': True, 'UpdateInplacePrioritiesForIpu': True})
        else:
            self.ipu_options = poptorch.Options()
            self.ipu_options.replicationFactor(ipu_args['num_replicas'])
            self.ipu_options.autoRoundNumIPUs(True)
            self.ipu_options.deviceIterations(ipu_args['batches_per_step'])
            self.ipu_options.Training.gradientAccumulation(1)
            self.ipu_options.outputMode(poptorch.OutputMode.All)
            self.ipu_options.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))
            self.ipu_options.setAvailableMemoryProportion({f'IPU{i}': ipu_args['available_memory_proportion'] for i in range(len(self.args['pipeline']) + 1)})

    def set_random_seed(self):
        np.random.seed(self.args['trainer']['random_seed'])

    def bulid_checkpoints(self):
        self.checkpoint = CheckPoint(0, self.logger)

    def build_vocab(self):
        self.vocab = Vocab(self.use_generate, **self.args['vocab'])

    def build_dataset(self):
        self.build_vocab()
        self.train_dataset = AishellDataset(vocab=self.vocab, args = self.args, dataset=self.args['train_dataset'])
        if not self.use_generate:
            self.val_dataset = AishellDataset(vocab=self.vocab, args = self.args, dataset=self.args['val_dataset'])


    def build_iterator(self, is_train=True):
        if not self.ipu_options:
            self.build_ipu_options()
        self.build_dataset()
        if is_train:
            self.is_spec_aug = self.args['train_dataset']['is_spec_aug']
            collate_fn = CollateFn(self.vocab.sos_id, self.vocab.eos_id, self.is_spec_aug, self.args['train_dataset']['dtype'])
        else:
            self.is_spec_aug = False
            collate_fn = CollateFn(self.vocab.sos_id, self.vocab.eos_id, self.is_spec_aug, self.args['val_dataset']['dtype'])
        self.train_iterator = poptorch.DataLoader(
            self.ipu_options,
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            mode=poptorch.DataLoaderMode.Async,
            shuffle=True,
            **self.args['train_iterator'],
        )
        if not self.use_generate:
            self.val_iterator = poptorch.DataLoader(
                self.ipu_options,
                dataset=self.val_dataset,
                collate_fn=collate_fn,
                mode=poptorch.DataLoaderMode.Async,
                shuffle=True,
                **self.args['val_iterator'],
            )

    def build_model(self):
        encoder = ConformerEncoder(**self.args['encoder'])
        decoder = TransformerDecoder(**self.args['decoder'])
        self.feature_len = self.args['encoder']['input_size']
        self.use_generate = self.args['train_dataset']['use_generated_data']

        normalizer = GlobalMVN(self.use_generate, self.feature_len, **self.args['normalizer'])
        loss_fn = LabelSmoothingLoss(**self.args['loss_fn'])
        self.dtype = self.args['trainer']['dtype']

        self.model = Conformer(
            normalizer=normalizer, encoder=encoder, decoder=decoder, loss_fn=loss_fn
        )
        self.init_type = self.args['trainer']['init_type']
        initialize(self.model, self.init_type)
        if self.dtype == 'FLOAT16':
            self.model.half()
        self.model.autocast(enabled=True)

    def wrap_pipeline_model(self):
        assert self.model
        pipeline_args = self.args['pipeline']
        self.model = PipelineWrapper(self.model)
        self.model.set_start_point_list(pipeline_args)

    def build_trainer(self):
        self.trainer = Trainer(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            model=self.model,
            train_iterator=self.train_iterator,
            val_iterator=self.val_iterator,
            ipu_options=self.ipu_options,
            wandb=self.wandb,
            logger=self.logger,
            args=self.args,
            checkpoint=self.checkpoint,
        )

    def print_args(self):
        pprint.pprint(self.args)

    def train(self):
        self.print_args()
        self.set_random_seed()
        self.build_wandb()
        self.build_logger()
        self.build_ipu_options()
        self.bulid_checkpoints()
        self.build_model()
        self.wrap_pipeline_model()
        self.build_optimizer()
        self.build_iterator()
        self.build_trainer()
        self.trainer.run()

    def validate(self):
        self.print_args()
        self.build_wandb()
        self.build_logger()
        self.build_ipu_options(False)
        self.bulid_checkpoints()
        self.build_model()
        self.wrap_pipeline_model()
        self.build_optimizer()
        self.build_iterator(False)
        self.build_trainer()
        self.trainer.validate()


if __name__ == '__main__':
    fire.Fire(Workflow)
