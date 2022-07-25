# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

# coding=utf-8
from models.vae.architectures.vcd_ruiz_2019 import encoder as encoder_vcd
from models.vae.architectures.vcd_ruiz_2019 import decoder as decoder_vcd

encoders = {'vcd': encoder_vcd}
decoders = {'vcd': decoder_vcd}
