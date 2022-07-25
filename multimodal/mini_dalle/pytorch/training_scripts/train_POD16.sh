#!/bin/bash

python train.py --config L16_POD16 \
        --generated-data \
        --byteio True \
        --checkpoint-output-dir "" \
        --epochs 2
