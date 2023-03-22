# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2017 Sean Robertson
# Original code here https://github.com/spro/char-rnn.pytorch

import poptorch
import pytorch_lightning as pl
from pytorch_lightning.strategies import IPUStrategy
import unidecode
import torch
import string
import torch.nn as nn
import argparse


def convertToCharTensor(the_file):
    """Turn a file of text into something we can consume in our network. First
    we convert it to a tensor then pull out one chunk at a time."""
    all_characters = string.printable
    tensor = torch.zeros(len(the_file)).long()
    for c in range(len(the_file)):
        tensor[c] = all_characters.index(the_file[c])
    return tensor


class TextToChunkDataset(torch.utils.data.Dataset):
    def __init__(self, data_chunksize, file_name):
        """Breaks a given text file into discrete chunks of text.
        Args:
            data_chunksize: number of chars to be in each chunk
            file_name: the file to train with.
        """
        super().__init__()
        self.chunksize = data_chunksize

        file = open(file_name)
        self.buffer = convertToCharTensor(unidecode.unidecode(file.read()))
        file.close()

        # Number of chunks in the dataset
        self._length = len(self.buffer) // self.chunksize

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index >= self._length:
            raise StopIteration
        i = index * self.chunksize
        chunk = self.buffer[i : i + self.chunksize]
        input = chunk[:-1]
        target = chunk[1:]
        return input, target


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        data_chunksize=20,
        training_generation_len=200,
        generation_char="\n",
        temperature=0.5,
        epoch_print_schedule=50,
    ):
        """LSTM model that trains to output text following the same
           structure and dialect as the input text file. In this case,
           a Shakespearean text file is used. Instructions on how to
           retrieve this file explained in readme.md
        Args:
            data_chunksize: number of chars in each chunk.
            training_generation_len: the length of the generation during training.
            generation_char: the initial char beginning generation string.
            temperature: changes how the model selects from generated char probabilities. Lower = more likely selected, higher less likely.
            epoch_print_schedule: generates a segment of text after this number of training epochs. (Lower slows training)
        """
        super(LSTMModel, self).__init__()

        all_characters = string.printable
        n_characters = len(all_characters)
        input_size, output_size = n_characters, n_characters

        self.data_chunk_size = data_chunksize
        self.training_generation_len = training_generation_len
        self.generation_char = generation_char
        self.input_size = input_size
        self.hidden_size = 100
        self.output_size = output_size
        self.n_layers = 1
        self.encoder = nn.Embedding(input_size, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers)
        self.decoder = nn.Linear(self.hidden_size, output_size)
        self.temperature = temperature
        self.epoch_print_schedule = epoch_print_schedule

    def training_step(self, batch, _):
        """Run a training step on IPU."""
        x, y = batch
        batchsize = x.size()[0]
        hidden = self.init_hidden(batchsize)

        loss = 0
        # The incoming chunk is on element short due to a one character offset between x and y.
        for count in range(self.data_chunk_size - 1):
            slic = x[:, count]
            output, hidden = self.forward(slic, hidden)
            output = output.flatten(1)
            loss += torch.nn.functional.cross_entropy(output, y[:, count])

        return poptorch.identity_loss(loss, reduction="mean")

    def training_epoch_end(self, _):
        """At the end of epoch_print_schedule epochs, generate and print some text
        for manual user validation."""
        if self.current_epoch % self.epoch_print_schedule == 0:
            self.generate("\n", self.training_generation_len, self.temperature)

    def generate(self, seed_string, length_to_generate, temperature):
        """Generate a text segment of length |length_to_generate| starting with
        the string |seed_string|. We run this on CPU but for larger batching validation
        should be used."""
        # Inference Stage
        generation_char = self.generation_char
        batch_size = 1

        seed_string = convertToCharTensor(unidecode.unidecode(generation_char))
        hidden = self.init_hidden(batch_size)

        # Init hidden state
        for char in seed_string:
            _, hidden = self(char.reshape(1), hidden)

        inp = seed_string[0]
        inp = torch.reshape(inp, [1])

        all_characters = string.printable

        # Actual generation
        out_str = generation_char
        for i in range(0, length_to_generate):
            out, hidden = self.forward(inp, hidden)
            out = torch.nn.functional.softmax(out) / temperature

            selected_char_index = torch.multinomial(out, 1)[0]

            # Adding highest probable character to string out_str
            out_str += all_characters[selected_char_index]

            # Setting the next inp as the current highest probable character
            inp = selected_char_index

        print(f"\n\nOutput:\n{out_str}\n\n")

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size), torch.zeros(
            self.n_layers, batch_size, self.hidden_size
        )


def train(num_ipus, epochs, target_file, temperature):
    batchsize = 4
    chunk_size = 20

    options = poptorch.Options()
    options.deviceIterations(500)
    options.replicationFactor(num_ipus)

    text_to_learn_from = TextToChunkDataset(chunk_size, target_file)

    data = poptorch.DataLoader(
        dataset=text_to_learn_from, batch_size=batchsize, options=options, mode=poptorch.DataLoaderMode.Async
    )

    # Initializing the model
    model = LSTMModel(data_chunksize=chunk_size, temperature=temperature)

    # Create a trainer class which will run on IPU.

    trainer = pl.Trainer(
        accelerator="ipu",
        devices=1,
        max_epochs=epochs,
        log_every_n_steps=1,
        strategy=IPUStrategy(training_opts=options),
    )

    # Fit the model.
    trainer.fit(model, data)

    trainer.save_checkpoint("trained.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipus", help="Number of IPUs to use", type=int, default=8)
    parser.add_argument("--epochs", help="Train for this number of epochs.", type=int, default=100)
    parser.add_argument("--text-file", help="Text file to train the network on.", type=str, default="shakespeare.txt")
    parser.add_argument(
        "--generate", help="Generate a new sample from the previously trained examples.", action="store_true"
    )
    parser.add_argument(
        "--length-to-generate", help="How much text to generate (in characters)", type=int, default=1000
    )
    parser.add_argument(
        "--temperature",
        help="Floating point value between 0 and 1, when lower the model will pick safer predictions.",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    if args.generate:
        model = LSTMModel.load_from_checkpoint("trained.ckpt")
        model.generate("\n", args.length_to_generate, args.temperature)
    else:
        train(args.ipus, args.epochs, args.text_file, args.temperature)
