#!/bin/bash

# Run this script from inside the applications/popart/bert directory

# Inputs:
# Preprocessed Wikipedia data in files named data/wikipedia/preprocessed/wiki_XX_cleaned
# The Bert-Base, uncased checkpoint has been downloaded from Google and unzipped into to data/ckpts

# Outputs:
# Tokenised Wikipedia data will be written to data/wikipedia/AA/sequence_128/wiki_XX_tokenised
# and data/wikipedia/AA/sequence_384/wiki_XX_tokenised


# Tokenise the data for pretraining phase 1
# Note that the following options
#    --duplication-factor
#    --sequence-length
#    --mask-tokens
# must be provided with the same values written in
# the config file. For example: bert/configs/mk2/pretrain_base_128.json
# Run these processes in parallel 7 proccesses at a time
mkdir -p data/wikipedia/AA/sequence_128
for i in $(seq -f "%02g" 0 6); do (nohup python bert_data/create_pretraining_data.py --input-file data/wikipedia/preprocessed/wiki_${i}_cleaned --output-file data/wikipedia/AA/sequence_128/wiki_${i}_tokenised --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt --sequence-length 128 --mask-tokens 20 --duplication-factor 6 --do-lower-case >> /dev/null &); done
# Wait until the first 7 tokenisation processes finish
ps aux | grep sequence_128 | grep -v grep | awk '{print $2}' | while read pid; do tail --pid=${pid} -f /dev/null; done
for i in $(seq -f "%02g" 7 13); do (nohup python bert_data/create_pretraining_data.py --input-file data/wikipedia/preprocessed/wiki_${i}_cleaned --output-file data/wikipedia/AA/sequence_128/wiki_${i}_tokenised --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt --sequence-length 128 --mask-tokens 20 --duplication-factor 6 --do-lower-case >> /dev/null &); done
# Wait until the other 7 tokenisation processes finish
ps aux | grep sequence_128 | grep -v grep | awk '{print $2}' | while read pid; do tail --pid=${pid} -f /dev/null; done


# Tokenise the data for pretraining phase 2
# Note that the following options
#    --duplication-factor
#    --sequence-length
#    --mask-tokens
# must be provided with the same values written in
# the config file. For example: bert/configs/mk2/pretrain_base_384.json
# Run these processes in parallel 7 proccesses at a time (if you run all 14 processes simultaneously, it often makes for 1 or 2 of them finished abruptly)
mkdir -p data/wikipedia/AA/sequence_384
for i in $(seq -f "%02g" 0 6); do (nohup python bert_data/create_pretraining_data.py --input-file data/wikipedia/preprocessed/wiki_${i}_cleaned --output-file data/wikipedia/AA/sequence_384/wiki_${i}_tokenised --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt --sequence-length 384 --mask-tokens 56 --duplication-factor 6 --do-lower-case >> /dev/null &); done
# Wait until the first 7 tokenisation processes finish
ps aux | grep sequence_384 | grep -v grep | awk '{print $2}' | while read pid; do tail --pid=${pid} -f /dev/null; done
for i in $(seq -f "%02g" 7 13); do (nohup python bert_data/create_pretraining_data.py --input-file data/wikipedia/preprocessed/wiki_${i}_cleaned --output-file data/wikipedia/AA/sequence_384/wiki_${i}_tokenised --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt --sequence-length 384 --mask-tokens 56 --duplication-factor 6 --do-lower-case >> /dev/null &); done
# Wait until the other 7 tokenisation processes finish
ps aux | grep sequence_384 | grep -v grep | awk '{print $2}' | while read pid; do tail --pid=${pid} -f /dev/null; done
