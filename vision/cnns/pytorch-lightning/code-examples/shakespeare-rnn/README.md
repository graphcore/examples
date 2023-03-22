# Shakespeare RNN-LSTM example

This example showcases a RNN-LSTM model running on the Graphcore IPU, with PyTorch Lightning.
A custom mapping-style dataset is used which takes in a text file and splits it into chunks.

## Setup
To set up the script, first go to the examples/vision/cnns/pytorch directory and follow the instructions for creating a virtual environment and installing dependencies.
Return to the pytorch-lightning directory

Now enable the Poplar SDK:

```console
source <path-to-poplar-sdk>/popart-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
source <path-to-poplar-sdk>/poplar-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
```

## Instructions
Download the example Shakespeare text file from [the GitHub source](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and name it as shakespeare.txt:
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespeare.txt


Then train on the Shakespeare data run:
```console
python3 train.py
```

To then generate new Shakespeare text run:
```console
python3 train.py --generate
```

To change the length of a generated sample use the option --length-to-generate.


For the other options:

`--temperature` - Changes the generation accuracy. A lower value makes the model pick more likely options from the distribution. Acceptable values are 0 to 1 excluding 0. Default is 0.5.

`--epochs` - Number of training epochs. Default is 500.

`--text-file` - The text to train on, default is shakespeare.txt.

`--length-to-generate` - The size of the text in characters to be generated. Default is 1000.

## License

This example is licensed under the MIT license - see the LICENSE file at the top-level of this repository.

This directory includes derived work from the following:

The Practical PyTorch Series - PyTorch char-rnn for character-level text generation example: https://github.com/spro/char-rnn.pytorch

MIT License

Copyright (c) 2017 Sean Robertson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
