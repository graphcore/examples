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


import os
import time
import torch
import numpy as np
import poptorch
from poptorch import trainingModel
from poptorch import inferenceModel
from src.iterator.cycle import cycle
from src.utils.score import get_cer
import horovod.torch as hvd
import popdist
import popdist.poptorch


class Trainer:
    def __init__(
        self,
        optimizer,
        scheduler,
        model,
        train_iterator,
        val_iterator,
        ipu_options,
        wandb,
        logger,
        args,
        checkpoint,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.torch_model = model
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.ipu_options = ipu_options
        self.wandb = wandb
        self.logger = logger
        self.args = args
        self.checkpoint = checkpoint

    def run(self):
        self._prepare_training()
        self._train_epochs()

    def validate(self):
        self._prepare_val()
        self._valid_epochs()

    def _prepare_training(self):
        self.logger.info(f'preparing training.')
        self.logger.info(f'preparing experiment folder.')
        self.steps_per_epoch = len(self.train_iterator)
        self.logger.info(f'step num per epoch is: {self.steps_per_epoch}.')
        self.samples_per_step = self.train_iterator._combined_batch_size
        self.logger.info(f'sample num per step is: {self.samples_per_step}.')
        self.train_iterator = cycle(self.train_iterator)
        self.logger.info(f'preparing training done.')
        self.num_epochs = self.args['trainer']['num_epochs']
        self.log_every_n_step = self.args['trainer']['log_every_n_step']
        self.pretrained_checkpoint = self.args['checkpoints']['pretrained_checkpoint']
        self.save_checkpoint_path = self.args['checkpoints']['save_checkpoint_path']
        self.save_ck_epoch = self.args['checkpoints']['save_ck_epoch']
        self.resume = False
        self.num_instance = 1

    def _prepare_val(self):
        self.logger.info(f'preparing val.')
        self.steps_per_epoch = len(self.val_iterator)
        self.logger.info(f'step num per epoch is: {self.steps_per_epoch}.')
        self.samples_per_step = self.val_iterator._combined_batch_size
        self.logger.info(f'preparing val done.')
        self.num_epochs = self.args['trainer']['num_epochs']
        self.log_every_n_step = self.args['trainer']['log_every_n_step']
        self.save_checkpoint_path = self.args['checkpoints']['save_checkpoint_path']

    def _wrap_model(self, type):
        self.logger.info(f'wrapping model.')
        if type == 'train':
            self.torch_model.train()
            self.training_model = trainingModel(
                model=self.torch_model,
                options=self.ipu_options,
                optimizer=self.optimizer,
            )
            self.logger.info(f'wrapped training model.')
        elif type == 'val':
            self.torch_model.eval()
            self.val_model = inferenceModel(model=self.torch_model, options=self.ipu_options)
            self.logger.info(f'wrapped inference model.')

    def sync_duration(self, outputs, factor=1, average=True):
        if popdist.isPopdistEnvSet():
            if isinstance(outputs, float):
                return float(hvd.allreduce(torch.Tensor([outputs]), average=average).item())
            else:
                return [hvd.allreduce(output.div(factor), average=average).mean().item() for output in outputs]
        else:
            if isinstance(outputs, float):
                return outputs
            else:
                return [output.div(factor).mean().item() for output in outputs]


    def _train_one_epoch(self, current_epoch):
        for step in range(self.steps_per_epoch):
            start = time.time()
            feature, feature_length, target_in, target_out, target_length = next(
                self.train_iterator
            )
            target_in = target_in.int()
            target_out = target_out.int()
            data_time = time.time()
            start_step = time.perf_counter()
            if self.args['ipu_options']['compile_only']:  # Compile model
                start_compile = time.perf_counter()
                self.training_model.compile(feature, feature_length, target_in, target_out, target_length)
                duration_compilation = time.perf_counter() - start_compile
                self.logger.info(f"Compiled/Loaded model in {duration_compilation} secs")
                self.logger.info("-----------------------------------------------------------")
                self.logger.info("Model successfully compiled. Exiting now as '--compile-only' argument was passed.")
                exit(0)
            loss, outputs = self.training_model(
                feature, feature_length, target_in, target_out, target_length
            )
            if self.resume:
                self.scheduler.resume = True
                self.scheduler.steps_per_epoch = self.steps_per_epoch
                self.scheduler.current_epoch = current_epoch
                self.scheduler.step()
                self.resume = False
            else:
                self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            self.scheduler.resume = False
            self.training_model.setOptimizer(self.optimizer)
            end = time.time()
            if popdist.isPopdistEnvSet():
                pure_tput = self.sync_duration(self.samples_per_step / (time.perf_counter() - start_step), average=False)
                tput = self.sync_duration(self.samples_per_step / (end - start), average=False)
            else:
                tput = self.samples_per_step / (end - start)
                pure_tput = self.samples_per_step / (end - data_time)
            data_consumption_ratio = (data_time - start) / (end - start)
            if self.scheduler.global_step % self.log_every_n_step == 0:
                self.logger.info(f'Epochs: {current_epoch}/{self.num_epochs}, step: {self.scheduler.global_step}/{self.steps_per_epoch*self.num_epochs}, tput: {tput:3.0f}, data consumption ratio: {data_consumption_ratio:0.3f}, pure_tput: {pure_tput:0.3f}, loss: {loss.mean().item():3.3f}, lr: {lr:.2e}')
            if self.wandb:
                self.wandb.log(
                    {
                        'step': self.scheduler.global_step,
                        'tput': tput,
                        'data consumption ratio': data_consumption_ratio,
                        'pure_tput': pure_tput,
                        'loss': loss.mean().item(),
                        'lr': lr,
                    }
                )
        if self.wandb:
                self.wandb.log({'Epochs': current_epoch})


    def _train_epochs(self):
        self.logger.info(f'start training.')
        self.start_epoch = 0

        if self.pretrained_checkpoint:
            self.start_epoch = self.load_pretrain()
            self.resume = True

        self._wrap_model('train')

        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info(f'training epoch: {epoch}')
            start = time.time()
            self._train_one_epoch(epoch)
            end = time.time()
            if popdist.isPopdistEnvSet():
                epoch_tput = self.sync_duration(self.steps_per_epoch * self.samples_per_step / (end - start), average=False)
            else:
                epoch_tput = self.steps_per_epoch * self.samples_per_step / (end - start)
            self.logger.info(
                f'training epoch: {epoch} done, epoch time: {end-start:.3f}, sample num: {self.steps_per_epoch*self.samples_per_step}, epoch average tput: {epoch_tput:3.0f}'
            )

            if self.save_ck_epoch and (epoch % self.save_ck_epoch) == 0:
                self.save(epoch)

        self.save(self.num_epochs - 1)


    def _valid_one_epoch(self, current_epoch):
        self.pretrained_checkpoint = self.save_checkpoint_path + '/epoch_' + str(current_epoch) + '/training_state.pt'
        self.load_pretrain(False)
        loss_ = []
        cer_ = []
        copyWeightsToDevice_flag = 0
        for step, batch in enumerate(self.val_iterator):
            feature, feature_length, target_in, target_out, target_length = batch
            loss, output = self.val_model(
                feature, feature_length, target_in, target_out, target_length
            )
            if copyWeightsToDevice_flag == 0:
                self.val_model.model.load_state_dict(self.torch_model.state_dict())
                self.val_model.copyWeightsToDevice()
                copyWeightsToDevice_flag = 1
            loss_.append(torch.mean(loss))
            cer_.append(get_cer(output, target_out, self.args['vocab']['vocab_path']))
        average_loss = np.mean(loss_)
        cer = np.mean(cer_)
        self.logger.info(f'Epochs: {current_epoch}/{self.num_epochs}, loss_valid: {average_loss},  cer: {cer}')

        if self.wandb:
            self.wandb.log({'Epochs': current_epoch,  'loss_valid': average_loss, 'cer': cer})


    def _valid_epochs(self):
        self.logger.info(f'start validing.')
        self._wrap_model('val')
        for epoch in range(self.num_epochs):
            self.logger.info(f'validing epoch: {epoch}')
            self._valid_one_epoch(epoch)


    def load_pretrain(self, is_train=True):
        training_state, self.torch_model, self.optimizer = self.checkpoint.load(self.pretrained_checkpoint, self.optimizer, self.torch_model, is_train)
        epoch_finished = 0
        if training_state is not None:
            epoch_finished = training_state['epoch']
        return epoch_finished

    def save(self, epoch):
        self.checkpoint.save(self.save_checkpoint_path, self.torch_model, self.optimizer, epoch, self.args, self.scheduler.global_step)
