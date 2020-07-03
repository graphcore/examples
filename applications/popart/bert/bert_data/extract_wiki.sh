#!/bin/bash

#Download the latest wikipedia dump and save it in the path given by the first argument

if [ $# -ne 3 ]; then
    echo "Usage: ${0} path-to-wikidump.xml path-to-destination path-to-extractor-tools"
    exit
fi

extractor_path="${3}"

dump_path="${1}"

output_path="${2}"

python3 "${extractor_path}"/WikiExtractor.py -b 1000M --processes 16 --filter_disambig_pages -o "${output_path}" "${dump_path}"
