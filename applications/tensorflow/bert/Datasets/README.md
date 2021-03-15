# Graphcore: BERT datasets

This directory contains the information required to create pre-training and training datasets for BERT.

`sample.txt` is a simple text file for initial pre-training with a small dataset.

The Wikipedia dataset used for pre-training contains approximately 2.5 billion wordpiece tokens. This is only an approximate size since the Wikipedia dump file is updated all the time.

**NOTE**: these are large datasets - at least 300GB of disk space will be required and the data should be stored on NVMe SSDs for maximum performance. The examples below use a folder structure that matches the config files. If you use a different folder structure, make sure that it is correctly represented in the config file you use.

## File structure

The following files are found in the `Datasets/` folder:

* `create_pretraining_data.py`: Creates pre-training dataset from a txt file.
* `data_loader.py`: Loads datasets from TFRecord.
* `sample.txt`:  Sample text used as minimal example to pre-train BERT.
* `wiki_preprocess.py`: Outputs files that can be used as input for  `create_pretraining_data.py`.
* `tokenization.py` Tokenizer file used to create pre-training data.


## Pre-training data from `sample.txt`

The sample text is in a form that is already suitable to create the pre-training data. Run the script as:

`python3 create_pretraining_data.py --input-file path/to/sample.txt --output-file data/sample.tfrecord --vocab-file path_to_the_vocab/vocab.txt --sequence-length 128 -- mask-tokens 20 --duplication-factor 6`

## Wikipedia pre-training data

All the instructions given below should be executed from the `Datasets/` folder. If necessary move to the  `Datasets/` folder first:

```shell
cd /path/to/examples/applications/tensorflow/bert/Datasets
```

**1) Download the latest Wikipedia dump**

BERT is trained on text taken from Wikipedia articles.
The user needs to download and extract in XML format the latest Wikipedia dump.

Dumps are available from https://dumps.wikimedia.org/ and are licensed under CC BY-SA 3.0 and GNU Free Documentation Licenses.

**2) Extract the data**

In order to create the pre-training data we need to extract the Wikipedia dump and put it in this form:

```
<doc id = article1>
Title of article 1

Body of article 1

</doc>

<doc id = article2>
Title of article 2

Body of article 2
</doc>
```

and so on.

One of the tools that can be used to do so is wikiextractor, https://github.com/attardi/wikiextractor.
A command line example to perform the extraction is the following:

```shell
python -m wikiextractor.WikiExtractor -b 1000M --processes 16 --filter_disambig_pages -o /output/path/ /path/to/the/wikidump.xml
```

**3) Pre-process the files**

The files from step 3 require further pre-processing with the `wiki_preprocess.py` script:

`python3 wiki_preprocess.py --input-file /target_folder/AA/ --output-file /preprocessed_target_folder`

where `target_folder/AA` contains the files from step 3 and `preprocessed_target_folder` will contain the new files (wiki_00_cleaned, wiki_01_cleaned, ...). The structure of the text in these files is now the same as the structure of the text in the `sample.txt` file.

**4) Tokenise the data**

The data can now be tokenised to create the pre-training dataset for BERT. For this step a vocabulary file is required. A vocabulary can be downloaded from the pre-trained model checkpoints at https://github.com/google-research/bert. We recommend to use the pre-trained BERT-Base Uncased model checkpoints.

The script `create_pretraining_data.py` will accept a glob of input and output files to tokenise. However, attempting to process them all at once may result in the process being killed by the operating system for consuming too much memory. It is therefore preferable to convert the files one by one.

In order to create pre-training data for phase1 the example script is:

```shell
python create_pretraining_data.py \
        --input-file /path/to/preprocessed/file \
        --output-file /path/to/output_seq_128 \
        --vocab-file ./vocab.txt \
        --sequence-length 128 \
        --mask-tokens 20 \
        --duplication-factor 5 \
        --do-lower-case
```

For Phase2 the script is similar:

```shell
python create_pretraining_data.py \
        --input-file /path/to/preprocessed/file \
        --output-file /path/to/output/seq_384 \
        --vocab-file ./vocab.txt \
        --sequence-length 384 \
        --mask-tokens 56 \
        --duplication-factor 5 \
        --do-lower-case
```


**NOTE:** When using an uncased vocab, use `--do-lower-case`.

**NOTE:** Make sure to use the same values for `mask-tokens` and `duplication-factor` when generating the data and using the pre-training script.

**NOTE:** The option `--remask` can be used to move the masked elements at the beginning of the sequence. This will improve the inference and training performance.

The Wikipedia dataset is now ready to be used in the Graphcore BERT model.
