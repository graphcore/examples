#!/bin/bash
input_folder_path="${1}"
output_folder_path="${2}"

python3 -m data.packing.pack_pretraining_data \
        --input-files=${input_folder_path}/wiki_'*'.tfrecord \
        --output-dir=data/packing/packed_unshuffled_512 \
        --mask-tokens=76 \
        --sequence-length=512;

cd data/packing/packed_unshuffled_512
for f in *.tfrecord; do python3 -m tfrecord.tools.tfrecord2idx $f `basename $f .tfrecord`.index; done
cd ../../..

python3 -m data.packing.shuffle_packed_data \
        --input-files=data/packing/packed_unshuffled_512/wiki_'*'${file_number}.tfrecord \
        --output-dir=${output_folder_path};

rm -rf data/packing/packed_unshuffled_512;

cd ${output_folder_path}
for f in *.tfrecord; do python3 -m tfrecord.tools.tfrecord2idx $f `basename $f .tfrecord`.index; done