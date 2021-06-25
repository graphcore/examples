#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
docker run -it -v ${PWD}/../data/:/data espnet:data bash -c 'cd /home/espnet/egs/aishell/asr1 && ./run.sh --data /data/ --dumpdir /data/ --stage -1 --stop_stage 2 && cp /home/espnet/egs/aishell/asr1/data/lang_1char/train_sp_units.txt /data/'
