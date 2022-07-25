#!/bin/bash
# run gpt2-small on single IPU
python text_generate_gpt2.py \
      --model-name-or-path gpt2 \
      --fp16 true \
      --single-ipu true \
      --poptorch-loop true \
      --output-len 256

# run gpt2-medium on 4 IPUs
# python text_generate_gpt2.py \
#       --model-name-or-path gpt2-medium \
#       --fp16 true \
#       --single-ipu false \
#       --poptorch-loop false \
#       --layers-per-ipu 1 7 8 8 \
#       --matmul-proportion 0.2 0.2 0.2 0.2 \
#       --output-len 256

# run gpt2-large on 8 IPUs
# python text_generate_gpt2.py \
#       --model-name-or-path gpt2-large \
#       --fp16 true \
#       --single-ipu false \
#       --poptorch-loop false \
#       --layers-per-ipu 1 5 5 5 5 5 5 5 \
#       --matmul-proportion 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 \
#       --output-len 256

# run gpt2-xlarge on 8 IPUs
# python text_generate_gpt2.py \
#       --model-name-or-path gpt2-xl \
#       --fp16 true \
#       --single-ipu false \
#       --poptorch-loop false \
#       --layers-per-ipu 1 6 6 7 7 7 7 7 \
#       --matmul-proportion 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 \
#       --output-len 256
