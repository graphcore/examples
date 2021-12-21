# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


# set pipeline and ipu partials for Faster-RCNN
def get_stage_configs(ipu_set=None, train=True, fp16=False):
    # input param ipu_set and train will be used in the future
    if train:
        if fp16:
            stage_configs = [
                '0_0', '0_0', '1_1', '1_1', '1_1', '1_1',
                ['2_2', '2_2', '2_2', '3_3', '3_3', '3_3'], '3_3'
            ]
        else:
            stage_configs = [
                '0_0', '7_0', '1_0', '1_0', '1_0', '3_0',
                ['6_0', '3_0', '4_0', '2_0', '7_0', '5_0'], '5_0'
            ]
    else:
        if fp16:
            stage_configs = ['0_0', '0_0', '1_0', ['2_0', '2_0', '2_0', '3_0', '3_0', '3_0'], '3_0']
        else:
            stage_configs = ['0_0', '0_0', '1_0', ['2_0', '3_0', '4_0', '5_0', '5_0', '6_0'], '6_0']
    return stage_configs
