# Copyright 2019 Graphcore Ltd.
import re
import argparse
import os
import nltk

# This checks if the tokenizers is installed. If it is not installed it downloads it.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def main(args):

    # Fisrt we check if the path is a file or a directory
    input_files = []
    output_files = []
    if os.path.isfile(args.input_file):
        input_files.append(args.input_file)
        output_files.append(args.output_file)
    else:
        # In this case the file passed is a path to a folder
        # What we want to do is then to apply the following transformation to all of the files inside the folder
        for _, __, f in os.walk(args.input_file):
            for file in f:
                input_files.append(os.path.join(args.input_file, file))
                output_files.append(os.path.join(
                    args.output_file, file+'_cleaned'))

    if args.small_files:
        print('The code is going to use just the first {} lines of the documents'.format(
            args.max_len))

    for idx, el in enumerate(input_files):
        wiki_in = open(el, 'r')
        with open(output_files[idx], 'w')as o:
            wiki_iterator = enumerate(wiki_in)
            for c, line in wiki_iterator:
                # We skip empty lines in the file
                if len(line.strip()):
                    if re.search('<doc', line):
                        # In this case we are dealing with the title so we skip a sentence
                        next(wiki_iterator)
                    elif re.search('</doc', line):
                        # In this case we have reached the end of the wikipedia page, so we leave an empty line
                        # in order to make the create_pretraining_data script know that it does not have to commect the sentences
                        o.write('\n')
                    else:
                        # We use the NLTK tokenizer in order to separate the sentences into each paragraph.
                        paragraph = nltk.sent_tokenize(line)
                        for sentence in paragraph:
                            o.write(sentence)
                            o.write('\n')
                if args.small_files:
                    if c > args.max_len:
                        # To avoid generating too long files for testing
                        break
        wiki_in.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, type=str)
    parser.add_argument('--output-file', required=True, type=str)
    parser.add_argument('--small-files', action='store_true',
                        help='If this argument is passed, just the first 100 sequences of each single file.')
    parser.add_argument('--max-len', default=100, type=int,
                        help='If the small-files flag is activated, this number is going to set the maximum number of lines per document to be analyzed.')
    args = parser.parse_args()
    main(args)
