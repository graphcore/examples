# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
import os
import time
import argparse
import numpy as np
import random

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import ipu_compiler

from seq2seq_edits import AttentionWrapperNoAssert, dynamic_decode, TrainingHelperNoCond, GreedyEmbeddingHelperNoCond

from data_gen.reader import Data, Vocabulary
from tensorflow.python.ipu import utils
import util

try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

tf.logging.set_verbosity(tf.logging.ERROR)

time_major = True
DTYPE = tf.float16
forget_bias = 1.0
max_gradient_norm = 1
learning_rate = 1
CHECKPOINT_FILE = './weights/'


def print_data(src, src_vocab, tgt, tgt_vocab):
    for i, s in enumerate(src.T):
        t = tgt.T[i]
        src_end_idx = list(s).index(src_vocab.end_id())
        try:
            tgt_end_idx = list(t).index(tgt_vocab.end_id())
        except ValueError:
            tgt_end_idx = len(t) - 1
        print("{} -> {}".format(
            ''.join(src_vocab.int_to_string(s[:src_end_idx])),
            ''.join(tgt_vocab.int_to_string(t[:tgt_end_idx])),
            ))


class Nmt(object):

    def __init__(self, opts):
        self.opts = opts
        self.src_length = opts.sequence_length
        self.tgt_length = 11  # YYYY-MM-DD<eot>

    def _build_generator(self, data, vocab):
        instance_id = range(len(data.inputs))
        priming = True
        if priming and self.opts.infer:
            priming = False
            batch_ids = random.sample(instance_id, self.opts.batch_size)
            src = np.array(data.inputs[batch_ids], dtype=np.int32)
            yield {self.placeholders['source']: src.T, }
        while True:
            batch_ids = random.sample(instance_id, self.opts.batch_size)
            src = np.array(data.inputs[batch_ids], dtype=np.int32)
            if self.opts.infer:
                if self.opts.interact:
                    src = np.array([vocab[0].string_to_int(input("Enter a human date: ").strip())])
                yield {
                    self.placeholders['source']: src.T,
                }
            else:
                tgt = np.roll(np.array(data.targets[batch_ids], dtype=np.int32), 1)
                tgt[:, 0] = self.start_id
                lbl = np.array(data.targets[batch_ids], dtype=np.int32)
                mask = np.zeros(lbl.shape)
                for i, label in enumerate(lbl):
                    end_idx = list(label).index(self.end_id)
                    mask[i][:end_idx+1] = 1

                yield {
                    self.placeholders['source']: src.T,
                    self.placeholders['target']: tgt.T,
                    self.placeholders['label']: lbl.T,
                    self.placeholders['mask']: mask.T
                }

    def _build_inputs(self):
        input_vocab = Vocabulary('./data/human_vocab.json', padding=self.src_length)
        output_vocab = Vocabulary('./data/machine_vocab.json', padding=self.tgt_length)
        self.src_vocab_size = input_vocab.size()
        self.tgt_vocab_size = output_vocab.size()

        self.start_id = output_vocab.start_id()
        self.end_id = output_vocab.end_id()

        data_file = './data/validation.csv' if self.opts.infer else './data/training.csv'
        data = Data(data_file, input_vocab, output_vocab)
        data.load()
        data.transform()

        self.placeholders = {
            'source': tf.placeholder(tf.int32, shape=[self.src_length, self.opts.batch_size], name="source"),
            'target': tf.placeholder(tf.int32, shape=[self.tgt_length, self.opts.batch_size], name="target"),
            'label': tf.placeholder(tf.int32, shape=[self.tgt_length, self.opts.batch_size], name="label"),
            'mask': tf.placeholder_with_default(
                tf.constant(1, shape=[self.tgt_length, self.opts.batch_size], dtype=tf.float16),
                [self.tgt_length, self.opts.batch_size],
                name="mask")
        }
        vocab = (input_vocab, output_vocab)
        generator = self._build_generator(data, vocab)
        return generator, vocab

    def infer(self):
        def build_infer():
            embedding = Nmt._build_embedding(self.src_vocab_size, self.opts.embedding_size,
                                             name="source_embedding")
            input_, encoder_outputs, encoder_state = self._build_encoder(embedding)
            embedding = Nmt._build_embedding(self.tgt_vocab_size, self.opts.embedding_size, name="tgt_embedding")
            samples, logits = self._build_decoder(encoder_outputs, encoder_state, embedding, train=False)
            return samples, logits

        with ipu_scope('/device:IPU:0'):
            data, vocab = self._build_inputs()
            batch = ipu_compiler.compile(build_infer, [])

        # Create a restoring object
        saver = tf.train.Saver()

        ipu_options = util.get_config(report_n=0)
        utils.configure_ipu_system(ipu_options)
        session = tf.Session()
        checkpoint = CHECKPOINT_FILE + 'ckpt'
        saver.restore(session, checkpoint)
        # Run a dummy value to force the graph compilation
        session.run(batch, feed_dict=next(data))
        while True:
            feed_dict = next(data)
            predictions, _ = session.run(batch, feed_dict=feed_dict)
            print_data(feed_dict[self.placeholders['source']], vocab[0], predictions, vocab[1])
            if not self.opts.interact:
                break

    def train(self):
        def build_train():
            embedding = Nmt._build_embedding(self.src_vocab_size, self.opts.embedding_size,
                                             name="source_embedding")
            input_, encoder_outputs, encoder_state = self._build_encoder(embedding)
            embedding = Nmt._build_embedding(self.tgt_vocab_size, self.opts.embedding_size, name="tgt_embedding")
            samples, logits = self._build_decoder(encoder_outputs, encoder_state, embedding, train=True)
            loss, update = self._build_optimiser(logits)
            return loss, samples, logits, update

        with ipu_scope('/device:IPU:0'):
            data, _ = self._build_inputs()
            batch = ipu_compiler.compile(build_train, [])

        # Create a restoring object
        saver = tf.train.Saver()

        if self.opts.save_graph:
            # Dump the graph to a logdir
            writer = tf.summary.FileWriter(os.path.join('./logs', 'NMT', time.strftime('%Y%m%d_%H%M%S_%Z')))
            writer.add_graph(tf.get_default_graph())

        ipu_options = util.get_config(report_n=0)
        utils.configure_ipu_system(ipu_options)
        session = tf.Session()
        checkpoint = CHECKPOINT_FILE + 'ckpt'
        if self.opts.ckpt:
            saver.restore(session, checkpoint)
        else:
            utils.move_variable_initialization_to_cpu()
            session.run(tf.global_variables_initializer())
        print("Init done.")

        session.run(batch, feed_dict=next(data))  # Warmup
        duration = 0
        avg_loss = 0
        best_loss = float('Inf')
        for e in range(1, 1 + self.opts.steps):
            start = time.time()
            l, _, _ = session.run(batch, feed_dict=next(data))
            duration += time.time() - start
            avg_loss += l
            if (e <= 1000 and not e % 100) or not e % 1000:
                duration /= 100 if e <= 1000 else 1000
                avg_loss /= 100 if e <= 1000 else 1000
                print("Step: {:>5}. Average Loss {:.3}. Items/sec {:.4}. Tokens/sec {}".format(
                    e,
                    avg_loss,
                    self.opts.batch_size / duration,
                    self.opts.batch_size * (self.src_length + self.tgt_length) / duration))
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    saver.save(session, checkpoint)
                duration = 0
                avg_loss = 0

    @staticmethod
    def _build_embedding(vocab_size, embedding_size, name="embedding"):
        with tf.variable_scope("embedding", dtype=DTYPE, use_resource=True) as scope:
            # Random embedding
            embedding = tf.get_variable(
                name, [vocab_size, embedding_size], scope.dtype,
                initializer=tf.initializers.random_uniform(maxval=1.0, dtype=scope.dtype), trainable=False)
        return embedding

    @staticmethod
    def _build_cell(num_units, num_layers):
        if num_layers is 1:
            return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias, state_is_tuple=False)
        cell_list = []
        for i in range(num_layers):
            cell_list.append(tf.contrib.rnn.BasicLSTMCell(
               num_units,
               forget_bias=forget_bias, state_is_tuple=False))
        return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _build_encoder(self, embedding):
        with tf.variable_scope("input", dtype=DTYPE, use_resource=True) as scope:
            source = self.placeholders['source']
            encoder_emb_inp = tf.nn.embedding_lookup(
                embedding, source)

        with tf.variable_scope("encoder", dtype=DTYPE, use_resource=True) as scope:  # use resource
            dtype = scope.dtype
            cell = Nmt._build_cell(self.opts.num_units, self.opts.num_layers)

            if self.opts.bi:
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell,
                    Nmt._build_cell(self.opts.num_units, self.opts.num_layers),
                    encoder_emb_inp,
                    dtype=dtype,
                    time_major=time_major,
                    swap_memory=False)
                encoder_outputs = tf.add_n(outputs)
                encoder_state = states[0] + states[1]
            else:
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inp,
                    dtype=dtype,
                    time_major=time_major,
                    swap_memory=False)

        return source, encoder_outputs, encoder_state

    def _build_attention(self, encoder_outputs, decoder_cell):
        with tf.variable_scope("attention", dtype=DTYPE, use_resource=True) as scope:
            # Attention is batch major
            inputs = tf.transpose(encoder_outputs, [1, 0, 2])

            if self.opts.attention == "luong":
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.opts.num_units,
                    inputs,
                    dtype=scope.dtype,
                    )
            else:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.opts.num_units,
                    inputs,
                    dtype=scope.dtype,
                    )

            return AttentionWrapperNoAssert(
                decoder_cell, attention_mechanism)

    def _build_decoder(self, encoder_outputs, encoder_state, embedding, train=False):
        with tf.variable_scope("decoder", dtype=DTYPE, use_resource=True) as decoder_scope:
            dtype = decoder_scope.dtype
            tgt_length = self.src_length * 2
            decoder_num_units = self.opts.num_units
            atten_num_units = self.opts.num_units

            # RNN Cell
            cell = Nmt._build_cell(decoder_num_units, self.opts.num_layers)
            initial_state = encoder_state

            # Attention wrapper
            if self.opts.attention:
                cell = self._build_attention(encoder_outputs, cell)
                initial_state = tf.contrib.seq2seq.AttentionWrapperState(
                    cell_state=encoder_state,
                    attention=tf.zeros([self.opts.batch_size, atten_num_units], dtype),
                    time=tf.constant(0, tf.int32),
                    alignments=tf.zeros([self.opts.batch_size, self.src_length], dtype),
                    alignment_history=(),
                    attention_state=tf.zeros([self.opts.batch_size, self.src_length], dtype)
                    )

            # Projection Layer
            projection_layer = tf.layers.Dense(units=self.tgt_vocab_size, use_bias=False, name="projection")

            if train:
                tgt_length = self.tgt_length
                target = self.placeholders['target']
                decoder_emb_inp = tf.nn.embedding_lookup(
                    embedding, target)

                helper = TrainingHelperNoCond(
                    decoder_emb_inp, np.full([self.opts.batch_size], tgt_length, dtype=np.int32), time_major=time_major)
            else:
                # Inference
                tgt_sos_id = self.start_id
                tgt_eos_id = self.end_id

                start_tokens = np.full([self.opts.batch_size], tgt_sos_id, dtype=np.int32)
                end_token = tgt_eos_id

                helper = GreedyEmbeddingHelperNoCond(
                    embedding, start_tokens, end_token)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell,
                helper,
                initial_state=initial_state,
                output_layer=projection_layer if not train else None  # applied per timestep
                )

            # Dynamic decoding
            outputs, final_context_state, _ = dynamic_decode(  # Contains the XLA check
                decoder,
                maximum_iterations=tgt_length,  # Required for static TensorArrays
                output_time_major=time_major,
                swap_memory=False,
                scope=decoder_scope)

            if train:
                # Specify dynamic shapes to avoid Assert
                logits = outputs.rnn_output
                logits.set_shape([tgt_length, self.opts.batch_size, atten_num_units])
                logits = projection_layer(logits)
                return outputs.sample_id, logits
            else:
                return outputs.sample_id, outputs.rnn_output

    def _build_optimiser(self, logits):
        with tf.variable_scope("loss", use_resource=True):
            labels = self.placeholders['label']
            mask = self.placeholders['mask']

            # Logits is dynamic so an Assert is added to check shapes
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            train_loss = (tf.reduce_sum(crossent*mask) / self.opts.batch_size)

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)

        clipped_gradients = [tf.clip_by_norm(grad, max_gradient_norm) for grad in gradients]

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(
            zip(clipped_gradients, params))

        return train_loss, update_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT model in TensorFlow to run on the IPU')
    parser.add_argument('--infer', action="store_true",
                        help="Inference Only")
    parser.add_argument('--bi', action="store_true",
                        help="Use bidirectional layer in encoder (with outputs summed)")
    parser.add_argument('--attention', choices=['luong', 'bahdanau'], default='luong',
                        help="Add an attention model")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Set batch-size")
    parser.add_argument('--num-units', type=int, default=512,
                        help="Number of units in each LSTM cell")
    parser.add_argument('--num-layers', type=int, default=1,
                        help="Size of LSTM stack in the encoder and decoder")
    parser.add_argument('--embedding-size', type=int, default=32,
                        help="Size of source and target embedding")
    parser.add_argument('--sequence-length', type=int, default=20,
                        help="Size of input length (by padding or truncating)")
    parser.add_argument('--ckpt', action="store_true",
                        help="load weights from latest checkpoint")
    parser.add_argument('--seed', type=int, default=1984,
                        help="Random seed")
    parser.add_argument('--interact', action="store_true",
                        help="Perform inference on values entered from the command line")
    parser.add_argument('--save-graph', action="store_true",
                        help="Save the graph to './logs' to be viewed by TensorBoard")
    parser.add_argument('--steps', type=int, default=50000,
                        help="Number of steps to complete in training")
    args = parser.parse_args()

    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    if args.interact:
        args.batch_size = 1
        args.infer = True

    print("NMT {}.\n Batch size: {}.  Hidden units: {}.  Layers: {}.".format(
        "Inference" if args.infer else "Training", args.batch_size, args.num_units, args.num_layers))

    n = Nmt(args)
    if args.infer:
        n.infer()
    else:
        n.train()
