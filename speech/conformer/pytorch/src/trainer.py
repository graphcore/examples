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
import horovod.torch as hvd
import popdist.poptorch

from src.utils.score import get_recog_predict, get_char_dict
from src.utils.compute_cer import compute_cer
from src.utils.average_model import average_epoch


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

    def recognize(self, ckpt_path=""):
        self._prepare_val()
        if self.args['compute_cer']['average_num'] is not None:
            self.val_loss = []
            with open("val_loss.txt", 'r') as val_loss_file:
                for value in val_loss_file.readlines():
                    self.val_loss += [[int(float(value.split(",")[0])), float(value.split(",")[1].strip())]]
            checkpoint = self.args['checkpoints']['save_checkpoint_path']
            dest_path = checkpoint + '/average_' + str(self.args['compute_cer']['average_num']) + '/'
            os.makedirs(dest_path, exist_ok=True)
            self.checkpoint_path = dest_path + 'average.pt'
            path_list = [
                self.args['checkpoints']['save_checkpoint_path'] + '/epoch_{}/training_state.pt'.format(int(epoch[0]))
                for epoch in self.val_loss[:self.args['compute_cer']['average_num']]
            ]
            average_epoch(self.args['compute_cer']['average_num'], path_list, self.args['checkpoints']['save_checkpoint_path'])
        else:
            self.checkpoint_path = ckpt_path

        # Extract the key corresponding to wav and label in order to compute_cer
        self.torch_model.eval()
        training_state = torch.load(self.checkpoint_path)
        self.torch_model.load_state_dict(training_state['model_weight'])

        predict = []
        for step, batch in enumerate(self.val_iterator):
            char_dict = get_char_dict(self.args['vocab']['vocab_path'])
            predict_ = self.get_seqence(self.args['compute_cer']['decode_mode'], batch, char_dict)
            predict += predict_

        with open("pre.txt", 'w') as pre_file:
            for index in range(len(predict)):
                pre_file.write(predict[index])
        cer_txt = self.args['checkpoints']['save_checkpoint_path'] + "/final_cer.txt"
        label_txt = self.args['compute_cer']['label_text']
        compute_cer(label_txt, './pre.txt', cer_txt)

    def _prepare_training(self):
        self.logger.info(f'preparing training.')
        self.logger.info(f'preparing experiment folder.')
        count = 0
        for b, s in enumerate(self.train_iterator):
            count += 1
        self.steps_per_epoch = count
        self.logger.info(f'step num per epoch is: {self.steps_per_epoch}.')
        self.samples_per_step = self.train_iterator._combined_batch_size
        self.logger.info(f'sample num per step is: {self.samples_per_step}.')
        self.logger.info(f'preparing training done.')
        self.num_epochs = self.args['trainer']['num_epochs']
        self.log_every_n_step = self.args['trainer']['log_every_n_step']
        self.pretrained_checkpoint = self.args['checkpoints']['pretrained_checkpoint']
        self.save_checkpoint_path = self.args['checkpoints']['save_checkpoint_path']
        self.save_per_epoch = self.args['checkpoints']['save_per_epoch']
        self.resume = False
        self.num_instance = 1

    def _prepare_val(self):
        self.logger.info(f'preparing val.')
        count = 0
        for b, s in enumerate(self.val_iterator):
            count += 1
        self.steps_per_epoch = count
        self.logger.info(f'step num per epoch is: {self.steps_per_epoch}.')
        self.samples_per_step = self.val_iterator._combined_batch_size
        self.logger.info(f'sample num per step is: {self.samples_per_step}.')
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


    def sync_metrics(self, outputs, factor=1, average=True):
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
        copyWeightsToDevice_flag = 0
        for step, batch in enumerate(self.train_iterator):
            start = time.time()
            keys, feature, feature_length, target_in, target_out, target_length = batch
            target_in = target_in.int()
            target_out = target_out.int()
            data_time = time.time()
            start_step = time.perf_counter()
            if copyWeightsToDevice_flag == 0:
                self.training_model.compile(feature, feature_length, target_in, target_out, target_length)
                copyWeightsToDevice_flag = 1
            if self.args['ipu_options']['compile_only']:  # Compile model
                start_compile = time.perf_counter()
                self.training_model.compile(feature, feature_length, target_in, target_out, target_length)
                duration_compilation = time.perf_counter() - start_compile
                self.logger.info(f"Compiled/Loaded model in {duration_compilation} secs")
                self.logger.info("-----------------------------------------------------------")
                self.logger.info("Model successfully compiled. Exiting now as '--compile-only' argument was passed.")
                exit(0)

            loss, loss_att, loss_ctc = self.training_model(
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
            if self.args['popdist_rank'] == 0:
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
                            'loss_att': loss_att.mean().item(),
                            'loss_ctc': loss_ctc.mean().item(),
                            'lr': lr,
                            'epoch': current_epoch,
                        }
                    )


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
            if self.args['popdist_rank'] == 0:
                self.logger.info(
                    f'training epoch: {epoch} done, epoch time: {end-start:.3f}, sample num: {self.steps_per_epoch*self.samples_per_step}, epoch average tput: {epoch_tput:3.0f}'
                )

            if self.save_per_epoch and (epoch % self.save_per_epoch) == 0:
                self.save(epoch)

        self.save(self.num_epochs - 1)


    def _valid_one_epoch(self, current_epoch):
        self.pretrained_checkpoint = self.save_checkpoint_path + '/epoch_' + str(current_epoch) + '/training_state.pt'
        self.load_pretrain(False)
        loss_ = []
        cer_ = []
        copyWeightsToDevice_flag = 0
        for step, batch in enumerate(self.val_iterator):
            keys, feature, feature_length, target_in, target_out, target_length = batch
            loss, loss_att, loss_ctc = self.val_model(
                feature, feature_length, target_in, target_out, target_length
            )
            if copyWeightsToDevice_flag == 0:
                self.val_model.model.load_state_dict(self.torch_model.state_dict())
                self.val_model.copyWeightsToDevice()
                copyWeightsToDevice_flag = 1
            if self.wandb and (self.args['popdist_rank'] == 0):
                self.wandb.log(
                    {
                        'cv_loss': loss.mean().item(),
                        'cv_loss_att': loss_att.mean().item(),
                        'cv_loss_ctc': loss_ctc.mean().item(),
                        'epoch': current_epoch,
                    }
                )
            loss_.append(loss.mean().item())
        average_loss = np.mean(loss_)
        self.logger.info(f'Epochs: {current_epoch}/{self.num_epochs}, loss_valid: {average_loss}')

        if self.wandb and (self.args['popdist_rank'] == 0):
            self.wandb.log({'Epochs': current_epoch,  'loss_valid': average_loss})
        return average_loss


    def _valid_epochs(self):
        self.logger.info(f'start validing.')
        self._wrap_model('val')
        self.val_loss = []

        for epoch in range(self.num_epochs):
            self.logger.info(f'validing epoch: {epoch}')
            average_loss = self._valid_one_epoch(epoch)
            self.val_loss += [[epoch, average_loss]]
        self.val_loss = np.array(self.val_loss)
        sort_idx = np.argsort(self.val_loss[:, -1])
        self.val_loss = self.val_loss[sort_idx][::1]
        with open("val_loss.txt", 'w') as val_loss_file:
            for value in self.val_loss:
                val_loss_file.write(str(value[0])+","+str(value[1])+"\n")


    def get_seqence(self, mode, batch_data, char_dict):
        keys, feature, feature_length, target_in, target_out, target_length = batch_data
        if mode == 'attention_decode':
            hyps, scores = self.torch_model.recognize(feature, feature_length)
            predict_ = get_recog_predict(hyps, char_dict, keys)
        elif mode == 'attention_rescoring':
            predict_ = []
            for index in range(len(feature)):
                key_ = [keys[index]]
                feature_ = feature[index].unsqueeze(0)
                feature_length_ = feature_length[index].unsqueeze(0)
                best_hyps, _ = self.torch_model.attention_rescoring(feature_, feature_length_, beam_size=self.args['compute_cer']['beam_size'])
                tor_arr = torch.Tensor((best_hyps)).unsqueeze(0)
                pre = get_recog_predict(tor_arr, char_dict, key_)
                predict_ += pre
        elif mode == 'ctc_greedy_search':
            predict_ = []
            best_hyps, best_scores = self.torch_model.ctc_greedy_search(feature, feature_length)
            for index in range(len(best_hyps)):
                key_ = [keys[index]]
                best_hyps_ = torch.tensor(best_hyps[index]).unsqueeze(0)
                pre = get_recog_predict(best_hyps_, char_dict, key_)
                predict_ += pre
        return predict_


    def load_pretrain(self, is_train=True):
        training_state, self.torch_model, self.optimizer = self.checkpoint.load(self.pretrained_checkpoint, self.optimizer, self.torch_model, is_train)
        epoch_finished = 0
        if training_state is not None:
            epoch_finished = training_state['epoch']
        return epoch_finished

    def save(self, epoch):
        self.checkpoint.save(self.save_checkpoint_path, self.torch_model, self.optimizer, epoch, self.args, self.scheduler.global_step)
