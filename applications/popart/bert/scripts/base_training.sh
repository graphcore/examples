#!/bin/bash

# Download Wikipedia pre-training data (takes about an hour)
cd bert_data
mkdir -p ../data/wikipedia/extracted
mkdir -p ../data/wikipedia/preprocessed
./wiki_downloader.sh ../data/wikipedia

# Install wikiextractor package
pip install wikiextractor
# Run wikiextractor (takes about 2 hours)
./extract_wiki.sh ../data/wikipedia/wikidump.xml ../data/wikipedia/extracted
# Run wikipedia preprocessing (takes about an hour)
python wikipedia_preprocessing.py --input-file ../data/wikipedia/extracted/AA/ --output-file ../data/wikipedia/preprocessed

# Now go back to /path/to/examples/applications/popart/bert
cd ..

# Prepare training for bert/configs/pretrain_base_128.json
# Note that
# * L means "num_layers" in the config
# * H means "hidden_size" in the config
# * A means "attention_heads" in the config
curl --create-dirs -L https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -o data/ckpts/uncased_L-12_H-768_A-12.zip
unzip data/ckpts/uncased_L-12_H-768_A-12.zip -d data/ckpts

# Tokenise the data for bert/configs/pretrain_base_128.json
# Note that the following options
#    --duplication-factor
#    --sequence-length
#    --mask-tokens
# must be provided with the same values written in
# the config file bert/configs/pretrain_base_128.json
# Run these processes in parallel 7 proccesses at a time
mkdir -p data/wikipedia/AA/sequence_128
for i in $(seq -f "%02g" 0 6); do (nohup python bert_data/create_pretraining_data.py --input-file data/wikipedia/preprocessed/wiki_${i}_cleaned --output-file data/wikipedia/AA/sequence_128/wiki_${i}_tokenised --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt --sequence-length 128 --mask-tokens 20 --duplication-factor 6 --do-lower-case >> /dev/null &); done
# Wait until the first 7 tokenisation processes finish
ps aux | grep sequence_128 | grep -v grep | awk '{print $2}' | while read pid; do tail --pid=${pid} -f /dev/null; done
for i in $(seq -f "%02g" 7 13); do (nohup python bert_data/create_pretraining_data.py --input-file data/wikipedia/preprocessed/wiki_${i}_cleaned --output-file data/wikipedia/AA/sequence_128/wiki_${i}_tokenised --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt --sequence-length 128 --mask-tokens 20 --duplication-factor 6 --do-lower-case >> /dev/null &); done
# Wait until the other 7 tokenisation processes finish
ps aux | grep sequence_128 | grep -v grep | awk '{print $2}' | while read pid; do tail --pid=${pid} -f /dev/null; done


# Tokenise the data for bert/configs/pretrain_base_384.json
# Note that the following options
#    --duplication-factor
#    --sequence-length
#    --mask-tokens
# must be provided with the same values written in
# the config file bert/configs/pretrain_base_384.json
# Run these processes in parallel 7 proccesses at a time (if you run all 14 processes simultaneously, it often makes for 1 or 2 of them finished abruptly)
mkdir -p data/wikipedia/AA/sequence_384
for i in $(seq -f "%02g" 0 6); do (nohup python bert_data/create_pretraining_data.py --input-file data/wikipedia/preprocessed/wiki_${i}_cleaned --output-file data/wikipedia/AA/sequence_384/wiki_${i}_tokenised --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt --sequence-length 384 --mask-tokens 56 --duplication-factor 6 --do-lower-case >> /dev/null &); done
# Wait until the first 7 tokenisation processes finish
ps aux | grep sequence_384 | grep -v grep | awk '{print $2}' | while read pid; do tail --pid=${pid} -f /dev/null; done
for i in $(seq -f "%02g" 7 13); do (nohup python bert_data/create_pretraining_data.py --input-file data/wikipedia/preprocessed/wiki_${i}_cleaned --output-file data/wikipedia/AA/sequence_384/wiki_${i}_tokenised --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt --sequence-length 384 --mask-tokens 56 --duplication-factor 6 --do-lower-case >> /dev/null &); done
# Wait until the other 7 tokenisation processes finish
ps aux | grep sequence_384 | grep -v grep | awk '{print $2}' | while read pid; do tail --pid=${pid} -f /dev/null; done


# Pre-train base
./scripts/pretrain_base_full.sh

# Final pre-training result will be in ckpts/pretrain_base_384
PHASE2_DIR=$(ls ckpts/pretrain_base_384 -1 | tail -n 1)

# Download SQUAD 1.1 dataset
curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/squad/dev-v1.1.json
curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -o data/squad/train-v1.1.json
curl -L https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py -o data/squad/evaluate-v1.1.py

# Run the training on SQUAD 1.1 from the pretrained model
python bert.py --config configs/squad_base_384.json \
    --onnx-checkpoint ckpts/pretrain_base_384/$PHASE2_DIR/model.onnx \
    2>&1 | tee squad_train_384_log.txt

