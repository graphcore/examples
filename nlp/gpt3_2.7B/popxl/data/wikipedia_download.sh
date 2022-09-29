#!/bin/bash

#Download the latest wikipedia dump and save it in the path given by the first argument

folder_path="${1}"

file_path="${1}/wikidump.xml.bz2"

echo "Downloading the latest wikipedia dump in the folder"
echo $folder_path

echo "under the name of"
echo $file_path

wget 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2' -O "${file_path}"

# Unzip
cd ${folder_path}

echo "Unzipping the compressed dump"
bunzip2 -v wikidump.xml.bz2