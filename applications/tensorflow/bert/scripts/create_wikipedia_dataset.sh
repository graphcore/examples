#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
set -eu

# This script will create the Wikiepdia dataset used to pretrain BERT. 
# It will download the latest Wikipedia dump archive and preprocess, tokenize, and convert to TensorFlow TFRecord files.
# The entire process may take up to several hours.

bert_dir="$(dirname $0)/.."
bert_dir="$(realpath ${bert_dir})"
download_path="${bert_dir}/data/tf_wikipedia/downloads"
extract_path="${bert_dir}/data/tf_wikipedia/extracted"
preprocess_path="${bert_dir}/data/tf_wikipedia/preprocessed"
tokenised_128_path="${bert_dir}/data/tf_wikipedia/tokenised_128_dup5_mask20"
tokenised_384_path="${bert_dir}/data/tf_wikipedia/tokenised_384_dup5_mask58"
wikiextractor_path="${bert_dir}/scripts/wikiextractor"

# Donwload wikidump
xml_dump_file="enwiki-latest-pages-articles.xml"
if [ ! -f "${download_path}/${xml_dump_file}" ];then
    mkdir -p "${download_path}"
    cd "${download_path}" || exit 1
    echo "Downloading Wikipedia dump archive"
    wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    echo "Unzipping the compressed archive"
    bunzip2 -v "${xml_dump_file}.bz2"
    cd "${bert_dir}" || exit 1
else
    echo "Skipping download. Wikipedia dump file ${download_path}/${xml_dump_file} already exists."
fi

if [ ! -d "${wikiextractor_path}" ];then
     echo "WikiExtractor can not be found. Please clone in the scripts directory and add it to your PYTHONPATH:"
     echo "    git clone https://github.com/attardi/wikiextractor.git ${bert_dir}/scripts"
     echo "    export PYTHONPATH=\$PYTHONPATH:${wikiextractor_path}"
     exit 1
fi

# Extract wiki
if [ ! -d "${extract_path}/AA" ];then
    mkdir -p "${extract_path}"
    LC_CTYPE=C.UTF-8 python3  -m wikiextractor.WikiExtractor -b 1000M --processes 64 -o "${extract_path}" "${download_path}/${xml_dump_file}"
else
    echo "Skipping extraction. Wikipedia pages already extracted in ${extract_path}."
fi

# Preprocess
if [ ! -d "${preprocess_path}" ];then
    mkdir -p $preprocess_path
    python -u $bert_dir/bert_data/wiki_preprocess.py --input-file "${extract_path}/AA" --output-file "${preprocess_path}"
else
    echo "Skipping preprocessing. Preprocessed files in ${preprocess_path} already exist."
fi


# Tokenise
if [ ! -f "$bert_dir/vocab.txt" ]; then
        echo "Downloading vocab file"
        curl -L https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt -o $bert_dir/vocab.txt
else
        echo "Skipping donwloading vocab file. vocab.txt already exists."
fi

NUM_FILES=`ls $preprocess_path | wc -l`
FIRST_HALF=$((NUM_FILES / 2))
SECOND_HALF=$((NUM_FILES - FIRST_HALF))

echo "NUM_FILES: ${NUM_FILES}"

# For phase 1 pretraining
if [ ! -d "$tokenised_128_path" ]; then
        mkdir -p ${tokenised_128_path}/logs
        for file in `ls $preprocess_path | head -$FIRST_HALF`; do
                echo "Launching process for ${file}"
                nohup python -u $bert_dir/bert_data/create_pretraining_data.py \
                --input-file $preprocess_path/$file \
                --output-file $tokenised_128_path/$file.tfrecord \
                --vocab-file $bert_dir/vocab.txt \
                --sequence-length 128 \
                --mask-tokens 20 \
                --duplication-factor 5 \
                --remask \
                --do-lower-case &> $tokenised_128_path/logs/$file.log &
        done && wait
        for file in `ls $preprocess_path | tail -$SECOND_HALF`; do
                echo "Launching process for ${file}"
                nohup python -u $bert_dir/bert_data/create_pretraining_data.py \
                --input-file $preprocess_path/$file \
                --output-file $tokenised_128_path/$file.tfrecord \
                --vocab-file $bert_dir/vocab.txt \
                --sequence-length 128 \
                --mask-tokens 20 \
                --duplication-factor 5 \
                --remask \
                --do-lower-case &> $tokenised_128_path/logs/$file.log &
        done && wait
else
        echo "Skipping tokenization 128, Directory ${tokenised_128_path} already exists."
fi

# For phase 2 fine-tuning
if [ ! -d "$tokenised_384_path" ]; then
        mkdir -p ${tokenised_384_path}/logs
        for file in `ls $preprocess_path | head -$FIRST_HALF`; do
                echo "Launching process for ${file}"
                nohup python -u $bert_dir/bert_data/create_pretraining_data.py \
                --input-file $preprocess_path/$file \
                --output-file $tokenised_384_path/$file.tfrecord \
                --vocab-file $bert_dir/vocab.txt \
                --sequence-length 384 \
                --mask-tokens 58 \
                --duplication-factor 5 \
                --remask \
                --do-lower-case &> $tokenised_384_path/logs/$file.log &
        done && wait
        for file in `ls $preprocess_path | tail -$SECOND_HALF`; do
                echo "Launching process for ${file}"
                nohup python -u $bert_dir/bert_data/create_pretraining_data.py \
                --input-file $preprocess_path/$file \
                --output-file $tokenised_384_path/$file.tfrecord \
                --vocab-file $bert_dir/vocab.txt \
                --sequence-length 384 \
                --mask-tokens 58 \
                --duplication-factor 5 \
                --remask \
                --do-lower-case &> $tokenised_384_path/logs/$file.log &
        done && wait
else
        echo "Skipping tokenization 384, Directory ${tokenised_384_path} already exists."
fi
