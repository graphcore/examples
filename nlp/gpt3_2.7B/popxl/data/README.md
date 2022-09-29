# Data
To obtain the data used for pre-training follow the below instructions.

## 1. Raw data

Download the latest raw wikipedia dump using:
```bash
bash wikipedia_download.sh wikipedia_raw
```

Extract the data into another format:
```bash
pip3 install wikiextractor
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
bash wikipedia_extract.sh wikipedia_raw/wikidump.xml wikipedia_extracted
```

## 2. Preprocessing

Preprocess the data:
```bash
mkdir wikipedia_preprocessed
python3 wikipedia_preprocess.py --input-file-path wikipedia_extracted --output-file-path wikipedia_preprocessed
```

## 3. Generate TFRecords

To generate TFRecords from the preprocessed data
```bash
pip install tensorflow==1.15.0
mkdir wikipedia_tf
python3 write_into_tfrecord.py --input-file-path wikipedia_preprocessed/wikicorpus_en_one_article_per_line.pkl --output-file-path wikipedia_tf --seq-length 129 --stride 129
```

Then you need to generate the indices for the TFRecords
```bash
cd wikipedia_tf
for f in *.tfrecord; do python3 -m tfrecord.tools.tfrecord2idx $f `basename $f .tfrecord`.index; done
``` 