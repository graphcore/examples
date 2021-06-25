#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
set -eu

# This script will create the Wikiepdia dataset used to pretrain BERT. 
# It will download the latest Wikipedia dump archive and preprocess, tokenize, and convert to Tensorflow TFRecord files.
# The entire process may take up to several hours.

REL_SCRIPTS_DIR="$(dirname $0)/../bert_data"
SCRIPTS_DIR="$(realpath ${REL_SCRIPTS_DIR})"

download_path="tmp/downloads"
extract_path="tmp/extracted"
preprocess_path="tmp/preprocessed"
tokenised_128_path="tokenised_128_dup5_mask20"
tokenised_384_path="tokenised_384_dup5_mask56"
wikiextractor_py_path="${SCRIPTS_DIR}/wikiextractor/WikiExtractor.py"



xml_dump_file="enwiki-latest-pages-articles.xml"
if [ ! -f "${download_path}/${xml_dump_file}" ];then
    mkdir -p "${download_path}"
    cd "${download_path}" || exit 1
    echo "Downloading Wikipedia dump archive"
    wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

    echo "Unzipping the compressed archive"
    bunzip2 -v "${xml_dump_file}.bz2"
    cd .. || exit 1
else
    echo "Skipping download. Wikipedia dump file ${download_path}/${xml_dump_file} already exists."
fi
 
if [ ! -f "${wikiextractor_py_path}" ];then
     echo "WikiExtractor can not be found. Please clone in the scripts directory:"
     echo "git clone https://github.com/attardi/wikiextractor.git ${SCRIPTS_DIR}"
fi

if [ ! -d "${extract_path}/AA" ];then
    mkdir -p "${extract_path}"
    python3 ${wikiextractor_py_path} -b 1000M --processes 64 -o "${extract_path}" "${download_path}/${xml_dump_file}"
else
    echo "Skipping extraction. Wikipedia pages already extracted in ${extract_path}."
fi

if [ ! -d "${preprocess_path}" ];then
    mkdir -p $preprocess_path
    python ${SCRIPTS_DIR}/wiki_preprocess.py --input-file "${extract_path}/AA" --output-file "${preprocess_path}"
else
    echo "Skipping preprocessing. Preprocessed files in ${preprocess_path} already exist."
fi

if [ ! -f "vocab.txt" ]; then
    echo 'File "vocab.txt" does not exist, please download from an existing source, for example  "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt".'
fi

if [ ! -d "$tokenised_128_path" ]; then
    mkdir -p ${tokenised_128_path}
    for i in {00..14}; do
            name='wiki_'$i'_cleaned'
            output=$tokenised_128_path/$name.tfrecord
            python ${SCRIPTS_DIR}/create_pretraining_data.py \
                --input-file $preprocess_path/$name \
                --output-file $output \
                --vocab-file ./vocab.txt \
                --sequence-length 128 \
                --mask-tokens 20 \
                --duplication-factor 5 \
                --remask \
                --do-lower-case 2>&1 > "tokenization_128_process${i}.log"  &
            sleep 10
    done
    wait
else
    echo "Skipping tokenization 128, Directory ${tokenised_128_path} already exists."
fi

if [ ! -d "$tokenised_384_path" ]; then
    mkdir -p ${tokenised_384_path}
    for i in {00..14}; do
            name='wiki_'$i'_cleaned'
            output=$tokenised_384_path/$name.tfrecord
            python ${SCRIPTS_DIR}/create_pretraining_data.py \
                --input-file $preprocess_path/$name \
                --output-file $output \
                --vocab-file ./vocab.txt \
                --sequence-length 384 \
                --mask-tokens 56 \
                --duplication-factor 5 \
                --remask \
                --do-lower-case 2>&1 > "tokenization_384_process${i}.log" &
            sleep 10
    done
    wait
else
    echo "Skipping tokenization 384, Directory ${tokenised_384_path} already exists."
fi
