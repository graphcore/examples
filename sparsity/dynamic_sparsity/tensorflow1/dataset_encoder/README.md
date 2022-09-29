# README

## Encoding the Wikitext dataset

The script will download and encode articles from the wikitext-2 and wikitext-103 datasets using the GPT-2 encoder. The WikiText-2 and WikiText-103 datasets were created by [Stephen Merity & the SalesForce Einstein AI lab](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) and made available by them under the following license: [CC-BY-SA](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License).

## Setup & Prerequisites

Install the requirements for this repo:

    cd <examples>/sparsity/dynamic_sparsity/tensorflow1/dataset-encoder
    pip install -r requirements.txt

Clone the GPT-2 Repo:

    git clone https://github.com/openai/gpt-2

Next, download the model, which includes the pre-trained encoder parameters (BPE pairs and merges):

    cd <gpt2_repo>
    pip install -r requirements.txt
    python download_model.py 124M

_Note: The choice of model doesn't matter here - the packaged encoder is the same for all. `124M` is chosen
as a relatively small example model._

## Encoding the dataset

The script will download the raw dataset automatically if it doesn't already exist in the `--dataset-dir`
path (defaults to `./datasets`).

The datasets will then be saved in the corresponding subdirectories: `<dataset-dir>/<dataset-name>-gpt2/<sequence-length>`

Then, for training, validation and test in turn, it will load all articles from the file and cut them
down to individual sequences.

For the purposes of this dataset, a single sequence is all text between individual headings. The headings
themselves are excluded (any trimmed lines beginning with "="), as are blank lines.

The sequences are then tokenised using the GPT-2 encoder (which is called from the GPT-2 repo dowloaded in
setup). Sequences shorter than `--sequence-length` tokens are discarded. Those which are longer than
`--sequence-length` tokens are cropped to that length.

We write them out as Numpy files, with the Numpy version suffixed in the name to help identify pickling
issues with different Numpy versions. The files are written alongside the raw text.

## Example command line

    python encode_dataset.py --gpt2-repo-path <gpt2_repo> --dataset-name [wikitext-2|wikitext-103]

Parameters:

    --gpt2-repo-path    The path to the GPT-2 repo downloaded during setup.
    --sequence-length   The sequence length used to crop/discard tokens
    --dataset-dir       The path to which results are downloaded and saved.
    --dataset-name      Which dataset to encode [wikitext-2|wikitext-103]

## Unit tests

To run the unit tests, one can simply call `pytest` from this directory on the test directory:

    pytest test

To run tests against the GPT-2 encoder, please also provide the GPT-2 repo path:

    pytest test --gpt2-repo-path <gpt2_repo>
