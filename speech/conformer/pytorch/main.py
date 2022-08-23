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
from src.conformer import Conformer
from src.utils.initializer import initialize
from src.trainer import Trainer
from src.utils.lr_scheduler import WarmupLR
from src.utils.checkpoint import CheckPoint
from src.iterator.dataset import Dataset
from src.utils.file_utils import read_symbol_table, read_non_lang_symbols
from src.iterator.dataset import IPUCollateFn
from src.iterator.generate_data import GenerateDataset
from src.utils.cmvn import load_cmvn

import copy
import popdist
import popdist.poptorch
import horovod.torch as hvd
import os


def load_yaml(yaml_file):
    args = yaml.safe_load(open(yaml_file, 'r'))
    return args


class Workflow:
    def __init__(self, config_file='configs/train.yaml', **kwargs):
        self.config_file = config_file
        self.args = load_yaml(self.config_file)
        self.parse_args(kwargs)
        self.ipu_options = None
        self.vocab_size = self.args['decoder']['vocab_size']
        self.train_dataset = None
        self.val_dataset = None
        self.train_iterator = None
        self.val_iterator = None
        self.model = None
        self.trainer = None
        self.checkpoint = None
        self.optimizer = None
        self.scheduler = None
        self.wandb = None

    def parse_args(self, kwargs):
        for key, value in kwargs.items():
            try:
                module_name, config_name = key.split('.')
                self.args[module_name][config_name] = value
            except:
                raise KeyError(f'{key} is not found in {self.config_file}.')

    def build_optimizer(self):
        accum_type = self.dtype
        self.optimizer = poptorch.optim.Adam(
            self.model.parameters(), accum_type=accum_type, **self.args['optimizer'])
        self.scheduler = WarmupLR(self.optimizer, **self.args['scheduler'])

    def build_wandb(self):
        if not self.args['trainer']['wandb_name']:
            self.wandb = None
        else:
            if self.args['popdist_rank'] == 0:
                wandb.init(project=self.args['trainer']['wandb_name'], settings=wandb.Settings(console='off'))
                wandb.config.update(self.args)
                self.wandb = wandb

    def build_logger(self):
        """ build logger both for print in terminal and log.txt """
        logger_args = self.args['trainer']['logger']
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

    def launch_popdist(self):
        if popdist.isPopdistEnvSet():
            hvd.init()
            self.args['popdist_replicas'] = int(
                popdist.getNumLocalReplicas())
            self.args['popdist_rank'] = popdist.getInstanceIndex()
            self.args['NumInstances'] = popdist.getNumInstances()
        else:
            self.args['NumInstances'] = 1
            self.args['popdist_rank'] = 0

    def build_ipu_options(self, is_train=True):
        ipu_args = self.args['ipu_options']
        ipus_per_replica = len(ipu_args['pipeline']) + 1
        replicas = ipu_args['num_replicas']

        if is_train:
            lbs = self.args['train_iterator']['batch_size']
            ga = ipu_args["gradient_accumulation"]
            if popdist.isPopdistEnvSet():
                self.ipu_options = popdist.poptorch.Options(
                    ipus_per_replica=ipus_per_replica)
            else:
                self.ipu_options = poptorch.Options()
                self.ipu_options.replicationFactor(replicas)
            self.ipu_options.randomSeed(self.args['trainer']['random_seed'])
            self.ipu_options.autoRoundNumIPUs(True)
            self.ipu_options.Training.gradientAccumulation(ga)
            self.ipu_options.deviceIterations(ipu_args['device_iterations'])
            self.ipu_options.outputMode(poptorch.OutputMode.Final)
            self.ipu_options.enableExecutableCaching(
                ipu_args['executable_cache_dir'])
            self.ipu_options.Training.accumulationAndReplicationReductionType(
                poptorch.ReductionType.Mean)
            self.ipu_options.setExecutionStrategy(
                poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
            self.ipu_options.TensorLocations.setOptimizerLocation(
                poptorch.TensorLocationSettings()
                .useOnChipStorage(not ipu_args['optimizer_state_offchip'])
                .useReplicatedTensorSharding(ipu_args['replicated_tensor_sharding'] if replicas > 1 else False)
            )
            if ipu_args['enable_half_partials']:
                self.ipu_options.Precision.setPartialsType(torch.half)
            self.ipu_options.setAvailableMemoryProportion(
                {f'IPU{i}': ipu_args['available_memory_proportion'][i] for i in range(len(ipu_args['pipeline']) + 1)})

            self.ipu_options.Precision.enableStochasticRounding(
                ipu_args["enable_stochastic_rounding"])
            # Popart settings
            # enable recomputation in pipeline mode
            self.ipu_options._Popart.set(
                'disableGradAccumulationTensorStreams', True)
            self.ipu_options._Popart.set('outlineThreshold', 10.0)
            self.ipu_options._Popart.set("timeLimitScheduler", float(120))
            self.ipu_options._Popart.set(
                'accumulateOuterFragmentSettings.excludedVirtualGraphs', ['0'])
            self.ipu_options._Popart.set(
                'scheduleNonWeightUpdateGradientConsumersEarly', True)
            self.ipu_options._Popart.setPatterns({
                'TiedGather': True,
                'TiedGatherAccumulate': True,
                'UpdateInplacePrioritiesForIpu': True
            })

        else:
            lbs = self.args['val_iterator']['batch_size']
            ga = 1
            self.ipu_options = poptorch.Options()
            self.ipu_options.replicationFactor(replicas)
            self.ipu_options.autoRoundNumIPUs(True)
            self.ipu_options.deviceIterations(ipu_args['device_iterations'])
            self.ipu_options.Training.gradientAccumulation(ga)
            self.ipu_options.outputMode(poptorch.OutputMode.Final)
            self.ipu_options.setExecutionStrategy(
                poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))
            if ipu_args['enable_half_partials']:
                self.ipu_options.Precision.setPartialsType(torch.half)
            self.ipu_options.setAvailableMemoryProportion(
                {f'IPU{i}': ipu_args['available_memory_proportion'][i] for i in range(len(ipu_args['pipeline']) + 1)})

        if ipu_args['enable_profiling']:
            ampstr = ','.join(
                [str(amp_) for amp_ in ipu_args['available_memory_proportion']])
            sdk_version = "-".join(os.environ.get(
                "POPLAR_SDK_ENABLED").split("/")[-2].split("-")[-3:])
            pipeline_str = ""
            for stage in ipu_args['pipeline']:
                # encoder and decoder
                prefix = stage[0].split("__")
                if len(prefix) == 1:
                    pipeline_str += prefix[0][:3]   # eg:ctc
                else:
                    pipeline_str += prefix[0][:3] + prefix[1]   # eg:enc3

            profile_path = os.path.join(
                os.path.abspath(ipu_args["profile_path"]),
                f"bs{lbs}-ga{ga}-amp{ampstr}-rep{replicas}-pl{ipus_per_replica}-{pipeline_str}-{sdk_version}")

            os.makedirs(profile_path, exist_ok=True)

            engine_options = {
                "autoReport.directory": profile_path,
                "target.syncReplicasIndependently": "true",
                "debug.allowOutOfMemory": "true",
                "profiler.includeFlopEstimates": "true",
                "profiler.includeCycleEstimates": "true",
                "autoReport.all": "true",
                "autoReport.executionProfileProgramRunCount": "2",
            }
            self.ipu_options._Popart.set("engineOptions", engine_options)

        if ipu_args['compile_only']:
            self.ipu_options.useOfflineIpuTarget()

    def set_random_seed(self):
        np.random.seed(self.args['trainer']['random_seed'])
        torch.manual_seed(self.args['trainer']['random_seed'])

    def build_checkpoints(self):
        self.checkpoint = CheckPoint(0, self.logger)

    def build_dataset(self):
        self.train_conf = self.args['train_conf']
        self.cv_conf = copy.deepcopy(self.train_conf)
        self.cv_conf['speed_perturb'] = False
        self.cv_conf['spec_aug'] = False
        self.cv_conf['spec_sub'] = False
        self.cv_conf['shuffle'] = False

        if not self.args['train_dataset']['use_generated_data']:
            lang = self.args['vocab']['vocab_path']
            symbol_table = read_symbol_table(lang)
            self.train_dataset = Dataset(self.args['train_dataset']['data_mode'], self.args['train_dataset']
                                         ['data_list'], symbol_table, self.train_conf, None, None, True)
            self.val_dataset = Dataset(self.args['train_dataset']['data_mode'], self.args['val_dataset']
                                       ['data_list'], symbol_table, self.cv_conf, None, None, partition=False)
        else:
            self.train_dataset = GenerateDataset(self.args)

    def build_iterator(self, is_train=True):
        if not self.ipu_options:
            self.build_ipu_options(is_train)
        self.build_dataset()
        if self.args['train_iterator']['async_mode']:
            mode = poptorch.DataLoaderMode.Async
        else:
            mode = poptorch.DataLoaderMode.Sync
        if not self.args['train_dataset']['use_generated_data']:
            self.train_iterator = poptorch.DataLoader(
                self.ipu_options,
                self.train_dataset,
                batch_size=self.args['train_iterator']['batch_size'],
                num_workers=self.args['train_iterator']['num_workers'],
                persistent_workers=self.args['train_iterator']['persistent_workers'],
                async_options=self.args['train_iterator']['async_options'],
                mode=mode,
                collate_fn=IPUCollateFn(
                    self.train_conf["filter_conf"]["max_length"],
                    self.train_conf["filter_conf"]["token_max_length"],
                    dtype=self.dtype,
                    sos_id=self.vocab_size-1,
                    eos_id=self.vocab_size-1)
            )
            self.val_iterator = poptorch.DataLoader(
                self.ipu_options,
                self.val_dataset,
                batch_size=self.args['val_iterator']['batch_size'],
                num_workers=self.args['val_iterator']['num_workers'],
                persistent_workers=self.args['train_iterator']['persistent_workers'],
                async_options=self.args['train_iterator']['async_options'],
                mode=mode,
                collate_fn=IPUCollateFn(
                    self.train_conf["filter_conf"]["max_length"],
                    self.train_conf["filter_conf"]["token_max_length"],
                    dtype=self.dtype,
                    sos_id=self.vocab_size-1,
                    eos_id=self.vocab_size-1)
            )
        else:
            self.train_iterator = poptorch.DataLoader(
                self.ipu_options,
                dataset=self.train_dataset,
                persistent_workers=self.args['train_iterator']['persistent_workers'],
                async_options=self.args['train_iterator']['async_options'],
                mode=mode,
                shuffle=False,
                batch_size=self.args['train_iterator']['batch_size'],
                num_workers=self.args['train_iterator']['num_workers'],
            )

    def build_model(self):
        encoder = ConformerEncoder(**self.args['encoder'])
        decoder = TransformerDecoder(**self.args['decoder'])
        self.feature_len = self.args['encoder']['input_size']
        self.use_generate = self.args['train_dataset']['use_generated_data']
        if not self.use_generate:
            cmvn_file = self.args['normalizer']['cmvn']
            mean, istd = load_cmvn(cmvn_file, True)
            mean = torch.from_numpy(mean).float()
            istd = torch.from_numpy(istd).float()
        else:
            mean, istd = None, None
        normalizer = GlobalMVN(
            self.use_generate, self.feature_len, mean=mean, inv_std=istd)
        loss_fn = LabelSmoothingLoss(**self.args['loss_fn'])
        self.dtype = torch.float16 if self.args['train_dataset']['dtype'] == "FLOAT16" else torch.float32
        self.model = Conformer(
            normalizer=normalizer, encoder=encoder, decoder=decoder, loss_fn=loss_fn, args=self.args, dtype=self.dtype
        )
        self.init_type = self.args['trainer']['init_type']
        initialize(self.model, self.init_type)
        if self.args['trainer']['dtype'] == 'FLOAT16':
            self.model.half()
        self.model.autocast(enabled=True)

    def wrap_pipeline_model(self):
        assert self.model
        self.model.set_start_point_list(self.args['ipu_options']['pipeline'])

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
        self.launch_popdist()
        self.build_wandb()
        self.build_logger()
        self.build_checkpoints()
        self.build_model()
        self.wrap_pipeline_model()
        self.build_optimizer()
        self.build_iterator()
        self.build_trainer()
        self.trainer.run()

    def validate(self):
        self.print_args()
        self.launch_popdist()
        self.build_wandb()
        self.build_logger()
        self.build_checkpoints()
        self.build_model()
        self.wrap_pipeline_model()
        self.build_optimizer()
        self.build_iterator(False)
        self.build_trainer()
        self.trainer.validate()

    def recognize(self):
        self.print_args()
        self.build_logger()
        self.build_model()
        self.build_iterator(False)
        self.build_trainer()
        self.trainer.recognize()


if __name__ == '__main__':
    fire.Fire(Workflow)
