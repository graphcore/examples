# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import wandb
import tensorboardX as tb
import logging


GLOBAL_LOGGER = None


class Logger:
    def __init__(self,
                 log_dir,
                 log_name,
                 post_fix='',
                 tb_on=True,
                 wandb_on=True,
                 project_name='faster-rcnn',
                 resume=False):
        self.log_dir = log_dir
        self.log_name = log_name
        self.tb_on = tb_on
        self.wandb_on = wandb_on

        log_filemode = 'a' if resume else 'w'
        logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(
                                self.log_dir, 'log{}.txt'.format(post_fix)),
                            filemode=log_filemode)
        self.logger = logging.getLogger("Train")

        if self.tb_on:
            tb_dir = os.path.join(log_dir, 'tensorboard')
            if not os.path.exists(tb_dir):
                os.mkdir(tb_dir)
            self.tbWriter = tb.writer.FileWriter(tb_dir)
        if self.wandb_on:
            wandb.init(project=project_name,
                       settings=wandb.Settings(console='off'),
                       id=self.log_name,
                       name=self.log_name)

    def log_str(self, str_input):
        self.logger.info(str_input)
        print(str_input)

    def log_data(self, name, var, step):
        if self.tb_on:
            tb_data = tb.summary.scalar(name, float(var))
            self.tbWriter.add_summary(tb_data, float(step))
        if self.wandb_on:
            wandb.log({name: float(var)}, step=step)


def init_log(log_dir,
             log_name,
             post_fix='',
             tb_on=True,
             wandb_on=True,
             project_name='faster-rcnn',
             resume=False):
    global GLOBAL_LOGGER
    assert GLOBAL_LOGGER is None
    GLOBAL_LOGGER = Logger(log_dir, log_name, post_fix, tb_on, wandb_on,
                           project_name, resume)


def log_str(*elements):
    global GLOBAL_LOGGER
    str_input = ''
    for ele in elements:
        str_input = str_input + str(ele)
    GLOBAL_LOGGER.log_str(str_input)


def log_data(name, var, step):
    global GLOBAL_LOGGER
    GLOBAL_LOGGER.log_data(name, var, step)
