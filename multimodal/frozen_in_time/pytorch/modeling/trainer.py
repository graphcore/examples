# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Max Bain
# This file has been modified by Graphcore

import os
import sys
import time
from asyncio.log import logger

import numpy as np
from numpy import inf
import time
import wandb

import datasets.data_loader as module_data
import modeling.loss as module_loss
import modeling.metric as module_metric
import modeling.model as module_arch
import poptorch
import torch
import transformers
from configs import options
from configs.parse_config import ConfigParser
from tqdm.autonotebook import tqdm


class TrainerIPU:
    def __init__(self, config: ConfigParser):
        self.config = config
        self.logger = config.get_logger(
            'trainer', config['trainer']['verbosity'])

        self.epochs = config['trainer']['epochs']
        self.save_period = config['trainer']['save_period']
        self.init_val = config['trainer'].get('init_val', True)

        # Configuration to monitor model performance and save best
        self.monitor = config['trainer'].get('monitor', 'max t2v_metrics')
        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        assert self.mnt_metric.split(
            '@')[0] in ['t2v_metrics', 'v2t_metrics', 'train_loss']
        assert self.mnt_metric.split(
            '@')[1] in ['R1', 'R5', 'R10', 'R50', 'MedR', 'MeanR', 'geometric_mean_R1-R5-R10']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = config['trainer'].get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # Setup data_loader instances

        if not self.config["validation_only"]:
            config['data_loader']['args'] = config['data_loader']['training']
            self.data_loader = config.initialize("data_loader", module_data)
            self.logger.info(
                f'Train dataset: {self.data_loader.n_samples} samples')

        config['data_loader']['args'] = config['data_loader']['inference']
        self.valid_data_loader = config.initialize("data_loader", module_data)
        self.logger.info(
            f'Val dataset: {self.valid_data_loader.n_samples} samples')

        os.environ['TOKENIZERS_PARALLELISM'] = "false"
        # Build tokenizer
        text_model_name = config['arch']['args']['text_params']['model']
        if "openai/clip" in text_model_name:
            tokenizer_builder = transformers.CLIPTokenizer
        else:
            tokenizer_builder = transformers.AutoTokenizer
        self.tokenizer = tokenizer_builder.from_pretrained(
            text_model_name,
            model_max_length=config['arch']['args']['text_params'].get(
                'max_length', 77),
            TOKENIZERS_PARALLELISM=False)

        self.max_samples_per_epoch = config['trainer']['max_samples_per_epoch']

        # Build model architecture, then print to console

        self.model = module_arch.PipelinedWithLoss(config.initialize('arch', module_arch),
                                                   config.initialize(
                                                       name="loss", module=module_loss), logger=self.logger).parallelize(
                                                   config['IPU_options']['pipelined_layers'])

        if self.config["arch"].get("precision", "16.16").split(".")[-1] == "16":
            self.logger.info("Setting model weights to 16-bit precision")                                                   
            self.model = self.model.half()

        self.logger.info(self.model.loss)
        # Get function handles of metrics
        self.metrics = [getattr(module_metric, met)
                        for met in config['metrics']]

        # Build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(
            lambda p: p.requires_grad, self.model.parameters())
        if 'accum_type' not in config['optimizer']['args']:
            pass
        elif config['optimizer']['args']['accum_type'] == "FP32":
            config['optimizer']['args']['accum_type'] = torch.float32
        else:
            config['optimizer']['args']['accum_type'] = torch.float16
        config['optimizer']['args']['betas'] = tuple(
            config['optimizer']['args']['betas'])

        self.optimizer = config.initialize(
            'optimizer', poptorch.optim, trainable_params)

        if 'lr_scheduler' in config._config:
            if hasattr(transformers, config._config['lr_scheduler']['type']):
                self.lr_scheduler = config.initialize(
                    'lr_scheduler', transformers, self.optimizer)
        else:
            self.lr_scheduler = None
            self.logger.debug('lr scheduler not found')

        if not self.config["validation_only"]:
            datum = next(iter(self.data_loader))
            self.training_model = self._compile_model(
                datum, modelType='training', optimizer=self.optimizer)

        datum = next(iter(self.valid_data_loader))
        self.inference_model = self._compile_model(
            datum, modelType='inference')
        
        if self.config["compile_only"]:
            sys.exit(0)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        if self.init_val:
            _ = self._valid_epoch(-1)

        for epoch in range(self.start_epoch, self.epochs + 1):
            log = self._train_epoch(epoch)

            self.logger.info(f'Training epoch {epoch} summary:')
            # print logged info to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False

            # check whether model performance improved or not, according to specified metric(mnt_metric)
            improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                (self.mnt_mode ==
                 'max' and log[self.mnt_metric] >= self.mnt_best)

            if improved:
                self.logger.info(
                    f"Performance {self.mnt_metric} improve from {self.mnt_best} to {log[self.mnt_metric]}")
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info("performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best)

        self.logger.info(f'Training complete')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self._switch_model('training')

        total_loss = 0
        total_steps = min(self.max_samples_per_epoch //
                          (self.data_loader.batch_size*self.config["IPU_options"]["training"].get("gradientAccumulation", 15)), len(self.data_loader))
        steps = 0
        self.logger.info(f'\ntrain_epoch{epoch}_begin')
        start_time = time.time()
        with tqdm(self.data_loader, desc=f"Training epoch {epoch}", total=total_steps) as progress:
            for data in progress:
                data_time = time.time() - start_time
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding="max_length",
                                                  truncation=True)
                loss = self.training_model(
                    data['text']['input_ids'], data['text']['attention_mask'], data['video'])

                detached_loss = loss.detach().item()
                total_loss += detached_loss
                
                step_time = time.time() - start_time
                tput = data['video'].shape[0] / step_time
                start_time = time.time()
                
                if self.config._config['trainer'].get("wandb",False):
                    wandb.log({"Throughput": tput, "Step time": step_time, "Data time": data_time, "Loss": detached_loss})
                
                progress.set_postfix({"loss": detached_loss,"tput":tput})

                steps += 1
                if steps >= total_steps:
                    break

        log = {'train_loss': total_loss / total_steps}

        log.update(self._valid_epoch(epoch))

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.train_model.setOptimizer(self.optimizer)
        return log

    def validate(self):
        val_log = self._valid_epoch(1)
        self.logger.info('Validation complete. Summary:')
        # print logged info to the screen
        for key, value in val_log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self._switch_model('inference')
        text_embed_arr = []
        vid_embed_arr = []
        self.logger.info(f'\nvalid_epoch_{epoch}_begin')
        with torch.no_grad():
            # For validation we switch the nested loop order, because alternate batches not needed
            # and dataloaders can be of different length
            start_time = time.time()
            with tqdm(self.valid_data_loader, desc=f"Validating") as progress:
                for data in progress:
                    data_time = time.time() - start_time
                    if self.tokenizer is not None:
                        text = self.tokenizer(
                            data['text'], return_tensors='pt', padding='max_length', truncation=True)

                    results = self.inference_model(
                        text['input_ids'], text['attention_mask'], data['video'])

                    text_embed, vid_embed = results
                    text_embed_arr.append(text_embed.cpu())
                    vid_embed_arr.append(vid_embed.cpu())
                    step_time = time.time() - start_time
                    start_time = time.time()

                    tput = data['video'].shape[0] / step_time
                    progress.set_postfix({"tput":tput})

                    if self.config["validation_only"] and self.config._config['trainer'].get("wandb",False):
                        wandb.log({"Samples per sec": tput, "Data time": data_time, "Step time": step_time})



        val_log = {}
        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)
        sims = module_arch.sim_matrix(
            text_embeds, vid_embeds).numpy()
        
        for metric in self.metrics:
            metric_name = metric.__name__
            res = metric(sims)
            self.verbose(epoch=epoch, metrics=res, dataset=self.valid_data_loader.dataset_name,
                         mode=metric_name)
            for key, value in res.items():
                val_log[f'{metric_name}@{key}'] = value

        if self.config._config['trainer'].get("wandb",False) and self.config["validation_only"]:
            wandb.log(val_log)

        return val_log

    def _switch_model(self, switch_to='training'):
        # Switch the graph on device and sync the state dict
        assert switch_to in ['training', 'inference']
        if switch_to == 'training':
            self.model.train()
            self.training_model.train()
            if self.inference_model.isAttachedToDevice():
                self.inference_model.detachFromDevice()
            self.training_model.attachToDevice()
        else:
            self.model.eval()
            self.inference_model.eval()
            if not self.config["validation_only"]:
                if self.training_model.isAttachedToDevice():
                    self.training_model.detachFromDevice()
                model_state_dict = self.training_model.state_dict()
                self.inference_model.load_state_dict(model_state_dict)
            self.inference_model.attachToDevice()

    def _compile_model(self, datum, modelType='training', optimizer=None):
        assert modelType in ['training', 'inference']
        opts = options.get_opts(self.config['IPU_options'], modelType)
        if modelType == 'training':
            model = poptorch.trainingModel(
                self.model.train(), options=opts, optimizer=optimizer)
        else:
            model = poptorch.inferenceModel(self.model.eval(), options=opts)
        # Compile model
        self.logger.info("---------- Compilation Started ---------")
        start_compile = time.perf_counter()
        text = self.tokenizer(
            datum['text'], return_tensors='pt', padding='max_length', truncation=True)
        datum = {'input_ids': text['input_ids'],
                 'attention_mask': text['attention_mask'], 'video': datum['video']}
        self.logger.info(
            f"text shape (iters*ga*localbs, max_length): {datum['input_ids'].shape}")
        self.logger.info(
            f"video shape(iters*ga*localbs, n_frames, ch, w, h):{datum['video'].shape}")
        model.compile(**datum)
        duration_compilation = time.perf_counter() - start_compile
        self.logger.info(
            (f"Compiled {modelType} model in {duration_compilation} secs"))
        self.logger.info("---------------------------------------")
        model.detachFromDevice()
        return model

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        cfg_name = str(self.config.cfg_fname).split(".")[0].replace("configs/", "")
        filename = str(self.checkpoint_dir /
                       '{}-checkpoint-epoch{}.pth'.format(cfg_name, epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / f'{cfg_name}_model_best.pth')
            torch.save(state, best_path)
            self.logger.info(f"Saving current best: {cfg_name}_model_best.pth ...")

    def verbose(self, epoch, metrics, mode: str, dataset: str):
        r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        msg = f"[{mode}]{dataset:s} epoch {epoch}, R@1: {r1:.1f}"
        msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
        msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
        self.logger.info(msg)
