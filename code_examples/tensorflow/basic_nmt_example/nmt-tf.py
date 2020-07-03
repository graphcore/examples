# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
import os
import time
import argparse
import numpy as np
import random

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import embedding_ops
from seq2seq_edits import (
    AttentionWrapperNoAssert,
    dynamic_decode,
    TrainingHelperNoCond,
    GreedyEmbeddingHelperNoCond,
)

from data.reader import Data, Vocabulary
from tensorflow.python.ipu import utils
import util

try:
    import __builtin__

    input = getattr(__builtin__, "raw_input")
except (ImportError, AttributeError):
    pass

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

time_major = True
DTYPE = tf.float16
forget_bias = 1.0
max_gradient_norm = 1
learning_rate = 0.5
CHECKPOINT_FILE = "./weights/"


def start_id(vocab):
    return vocab.vocabulary["<sot>"]


def end_id(vocab):
    return vocab.vocabulary["<eot>"]


def transform(data):
    data.inputs = np.array(list(map(data.input_vocabulary.string_to_int, data.inputs)))
    data.targets = np.array(
        list(map(data.output_vocabulary.string_to_int, data.targets))
    )
    assert len(data.inputs.shape) == 2, "Inputs could not properly be encoded"
    assert len(data.targets.shape) == 2, "Targets could not properly be encoded"


class DataGenerator:
    def __init__(self, data, vocab, opts, start_id, end_id):
        self.instance_id = range(len(data.inputs))
        self.data = data
        self.vocab = vocab
        self.opts = opts
        self.start_id = start_id
        self.end_id = end_id
        self.counter = 0
        if self.opts.infer:
            batch_ids = random.sample(self.instance_id, self.opts.batch_size)
            src = np.array(data.inputs[batch_ids], dtype=np.int32)
            self.query = src.T
            self.batch = {"source": src.T}
        else:
            self.batch = None

    def __next__(self):
        self.counter += 1
        if self.batch is not None:
            ret = self.batch
            self.batch = None
            return ret

        batch_ids = random.sample(self.instance_id, self.opts.batch_size)
        if self.opts.infer:
            if self.opts.interact:
                src = np.array(
                    [self.vocab[0].string_to_int(input("Enter a human date: ").strip())]
                )
            else:
                src = np.array(self.data.inputs[batch_ids], dtype=np.int32)
            self.query = src.T
            return {"source": self.query}
        else:
            src = np.array(self.data.inputs[batch_ids], dtype=np.int32)
            tgt = np.roll(np.array(self.data.targets[batch_ids], dtype=np.int32), 1)
            tgt[:, 0] = self.start_id
            lbl = np.array(self.data.targets[batch_ids], dtype=np.int32)
            mask = np.zeros(lbl.shape, dtype=np.float16)
            for i, label in enumerate(lbl):
                end_idx = list(label).index(self.end_id)
                mask[i][: end_idx + 1] = 1
            return {"source": src.T, "target": tgt.T, "label": lbl.T, "mask": mask.T}

    def __call__(self):
        return self

    def __iter__(self):
        return self


def print_data(src, src_vocab, tgt, tgt_vocab):
    tgt = np.squeeze(tgt, axis=0)
    for i, s in enumerate(src.T):
        t = tgt.T[i]
        src_end_idx = list(s).index(end_id(src_vocab))
        try:
            tgt_end_idx = list(t).index(end_id(tgt_vocab))
        except ValueError:
            tgt_end_idx = len(t) - 1
        print(
            "{} -> {}".format(
                "".join(src_vocab.int_to_string(s[:src_end_idx])),
                "".join(tgt_vocab.int_to_string(t[:tgt_end_idx])),
            )
        )


class Nmt(object):
    def __init__(self, opts):
        self.opts = opts
        self.src_length = opts.sequence_length
        self.tgt_length = 11  # YYYY-MM-DD<eot>
        self.host_embeddings = opts.host_embeddings
        self.input_vocab = Vocabulary(
            "./data/human_vocab.json", padding=self.src_length
        )
        self.output_vocab = Vocabulary(
            "./data/machine_vocab.json", padding=self.tgt_length
        )
        self.src_vocab_size = self.input_vocab.size()
        self.tgt_vocab_size = self.output_vocab.size()

    def _build_dataset(self):
        self.start_id = start_id(self.output_vocab)
        self.end_id = end_id(self.output_vocab)
        data_file = (
            "./data/validation.csv" if self.opts.infer else "./data/training.csv"
        )
        data = Data(data_file, self.input_vocab, self.output_vocab)
        data.load()
        transform(data)
        vocab = (self.input_vocab, self.output_vocab)
        self.generator = DataGenerator(
            data, vocab, self.opts, self.start_id, self.end_id
        )
        items = next(self.generator)
        output_types = {i: tf.dtypes.as_dtype(items[i].dtype) for i in items}
        output_shapes = {i: tf.TensorShape(items[i].shape) for i in items}
        total_bytes = 0
        for i in items:
            total_bytes += items[i].nbytes
        dataset = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, output_shapes=output_shapes
        )
        infeed_queue = ipu_infeed_queue.IPUInfeedQueue(
            dataset, "InfeedQueue", replication_factor=1
        )
        data_init = infeed_queue.initializer

        return dataset, infeed_queue, data_init, vocab

    def infer(self):
        with tf.device("cpu"):
            dataset, infeed_queue, data_init, vocab = self._build_dataset()
            outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")
        if self.host_embeddings:
            src_embedding = Nmt._build_embedding(
                self.src_vocab_size,
                self.opts.embedding_size,
                self.opts.host_embeddings,
                name="source_embedding",
            )
            tgt_embedding = Nmt._build_embedding(
                self.tgt_vocab_size,
                self.opts.embedding_size,
                self.opts.host_embeddings,
                name="tgt_embedding",
            )

        def build_common(src_embedding, tgt_embedding, source):
            input_, encoder_outputs, encoder_state = self._build_encoder(
                src_embedding, source
            )
            samples, logits = self._build_decoder(
                encoder_outputs, encoder_state, tgt_embedding, None, train=False
            )
            outfeed = outfeed_queue.enqueue({"samples": samples})
            return outfeed

        def build_infer(source):
            src_embedding = Nmt._build_embedding(
                self.src_vocab_size,
                self.opts.embedding_size,
                self.opts.host_embeddings,
                name="source_embedding",
            )
            tgt_embedding = Nmt._build_embedding(
                self.tgt_vocab_size,
                self.opts.embedding_size,
                self.opts.host_embeddings,
                name="tgt_embedding",
            )
            return build_common(src_embedding, tgt_embedding, source)

        def build_infer_host_embeddings(source):
            nonlocal src_embedding, tgt_embedding
            return build_common(src_embedding, tgt_embedding, source)

        with ipu_scope("/device:IPU:0"):
            build = build_infer_host_embeddings if self.host_embeddings else build_infer
            batch = ipu_compiler.compile(
                lambda: loops.repeat(1, build, infeed_queue=infeed_queue, inputs=[])
            )

        # Create a restoring object
        saver = tf.train.Saver()

        ipu_options = util.get_config(report_n=0)
        utils.configure_ipu_system(ipu_options)
        session = tf.Session()
        checkpoint = CHECKPOINT_FILE + ("host_ckpt" if self.opts.host_embeddings else "ckpt")
        saver.restore(session, checkpoint)
        session.run(data_init)
        if self.host_embeddings:
            batch = [batch, src_embedding(1, 1, False), tgt_embedding(1, 1, False)]
        result_queue = outfeed_queue.dequeue()
        # Run a dummy value to force the graph compilation
        session.run(batch)
        result = session.run(result_queue)
        predictions = result["samples"]
        print_data(self.generator.query, vocab[0], predictions, vocab[1])

        while True:
            session.run(batch)
            result = session.run(result_queue)
            predictions = result["samples"]
            print_data(self.generator.query, vocab[0], predictions, vocab[1])
            if not self.opts.interact:
                break

    def train(self):
        with tf.device("cpu"):
            dataset, infeed_queue, data_init, vocab = self._build_dataset()
            outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")
        if self.host_embeddings:
            src_embedding = Nmt._build_embedding(
                self.src_vocab_size,
                self.opts.embedding_size,
                self.opts.host_embeddings,
                name="source_embedding",
            )
            tgt_embedding = Nmt._build_embedding(
                self.tgt_vocab_size,
                self.opts.embedding_size,
                self.opts.host_embeddings,
                name="tgt_embedding",
            )

        def build_common(src_embedding, tgt_embedding, source, target, label, mask):
            nonlocal outfeed_queue
            input_, encoder_outputs, encoder_state = self._build_encoder(
                src_embedding, source
            )
            samples, logits = self._build_decoder(
                encoder_outputs, encoder_state, tgt_embedding, target, train=True
            )
            loss = self._build_optimiser(logits, label, mask)
            outfeed = outfeed_queue.enqueue({"loss": loss, "logits": logits})
            return outfeed

        def build_train(source, target, label, mask):
            src_embedding = Nmt._build_embedding(
                self.src_vocab_size,
                self.opts.embedding_size,
                self.opts.host_embeddings,
                name="source_embedding",
            )
            tgt_embedding = Nmt._build_embedding(
                self.tgt_vocab_size,
                self.opts.embedding_size,
                self.opts.host_embeddings,
                name="tgt_embedding",
            )
            return build_common(
                src_embedding, tgt_embedding, source, target, label, mask
            )

        def build_train_host_embeddings(source, target, label, mask):
            nonlocal src_embedding, tgt_embedding
            return build_common(
                src_embedding, tgt_embedding, source, target, label, mask
            )

        with ipu_scope("/device:IPU:0"):
            build = build_train_host_embeddings if self.host_embeddings else build_train
            batch = ipu_compiler.compile(
                lambda: loops.repeat(
                    self.opts.batches_per_step,
                    build,
                    infeed_queue=infeed_queue,
                    inputs=[],
                )
            )

        # Create a restoring object
        saver = tf.train.Saver()

        if self.opts.save_graph:
            # Dump the graph to a logdir
            writer = tf.summary.FileWriter(
                os.path.join("./logs", "NMT", time.strftime("%Y%m%d_%H%M%S_%Z"))
            )
            writer.add_graph(tf.get_default_graph())

        ipu_options = util.get_config(report_n=0)
        utils.configure_ipu_system(ipu_options)
        session = tf.Session()
        checkpoint = CHECKPOINT_FILE + ("host_ckpt" if self.opts.host_embeddings else "ckpt")
        if self.opts.ckpt:
            saver.restore(session, checkpoint)
        else:
            utils.move_variable_initialization_to_cpu()
            session.run(tf.global_variables_initializer())
        session.run(data_init)
        print("Init done.")
        if self.host_embeddings:
            batch = [
                batch,
                src_embedding(self.opts.batches_per_step, 1),
                tgt_embedding(self.opts.batches_per_step, 1),
            ]
        result_queue = outfeed_queue.dequeue()
        session.run(batch)  # Warmup
        best_loss = float("Inf")
        for e in range(self.opts.iterations):
            start = time.time()
            session.run(batch)
            result = session.run(result_queue)
            l = result["loss"]
            avg_loss = np.mean(l)
            duration = (time.time() - start) / self.opts.batches_per_step

            print(
                "Step: {:>5}. Average Loss {:.3}. Items/sec {:.4}. Tokens/sec {}".format(
                    (e + 1),
                    avg_loss,
                    self.opts.batch_size / duration,
                    self.opts.batch_size * (self.src_length + self.tgt_length) / duration,
                )
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                saver.save(session, checkpoint)

    @staticmethod
    def _build_embedding(vocab_size, embedding_size, host_embeddings, name="embedding"):
        if host_embeddings:
            embedding = embedding_ops.create_host_embedding(
                name,
                shape=[vocab_size, embedding_size],
                dtype=DTYPE,
                optimizer_spec=embedding_ops.HostEmbeddingOptimizerSpec(0.03),
                initializer=tf.initializers.random_uniform(maxval=1.0, dtype=DTYPE),
            )
        else:
            with tf.variable_scope(
                "embedding", dtype=DTYPE, use_resource=True
            ) as scope:
                # Random embedding
                embedding = tf.get_variable(
                    name,
                    [vocab_size, embedding_size],
                    scope.dtype,
                    initializer=tf.initializers.random_uniform(
                        maxval=1.0, dtype=scope.dtype
                    ),
                    trainable=True,
                )
        return embedding

    @staticmethod
    def _build_cell(num_units, num_layers):
        if num_layers is 1:
            return tf.contrib.rnn.BasicLSTMCell(
                num_units, forget_bias=forget_bias, state_is_tuple=False
            )
        cell_list = []
        for i in range(num_layers):
            cell_list.append(
                tf.contrib.rnn.BasicLSTMCell(
                    num_units, forget_bias=forget_bias, state_is_tuple=False
                )
            )
        return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _build_encoder(self, embedding, source):
        with tf.variable_scope("input", dtype=DTYPE, use_resource=True):
            if self.host_embeddings:
                encoder_emb_inp = embedding.lookup(source)
            else:
                encoder_emb_inp = tf.nn.embedding_lookup(embedding, source)

        with tf.variable_scope(
            "encoder", dtype=DTYPE, use_resource=True
        ) as scope:  # use resource
            dtype = scope.dtype
            cell = Nmt._build_cell(self.opts.num_units, self.opts.num_layers)

            if self.opts.bi:
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell,
                    Nmt._build_cell(self.opts.num_units, self.opts.num_layers),
                    encoder_emb_inp,
                    dtype=dtype,
                    time_major=time_major,
                    swap_memory=False,
                )
                encoder_outputs = tf.add_n(outputs)
                encoder_state = states[0] + states[1]
            else:
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inp,
                    dtype=dtype,
                    time_major=time_major,
                    swap_memory=False,
                )

        return source, encoder_outputs, encoder_state

    def _build_decoder(
        self, encoder_outputs, encoder_state, embedding, target=None, train=False
    ):
        with tf.variable_scope(
            "decoder", dtype=DTYPE, use_resource=True
        ) as decoder_scope:
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
                    attention_state=tf.zeros(
                        [self.opts.batch_size, self.src_length], dtype
                    ),
                )

            # Projection Layer
            projection_layer = tf.layers.Dense(
                units=self.tgt_vocab_size, use_bias=False, name="projection"
            )

            if train:
                tgt_length = self.tgt_length
                if self.host_embeddings:
                    decoder_emb_inp = embedding.lookup(target)
                else:
                    decoder_emb_inp = tf.nn.embedding_lookup(embedding, target)

                helper = TrainingHelperNoCond(
                    decoder_emb_inp,
                    np.full([self.opts.batch_size], tgt_length, dtype=np.int32),
                    time_major=time_major,
                )
            else:
                # Inference
                tgt_sos_id = self.start_id
                tgt_eos_id = self.end_id

                start_tokens = np.full(
                    [self.opts.batch_size], tgt_sos_id, dtype=np.int32
                )
                end_token = tgt_eos_id
                if self.host_embeddings:
                    helper = GreedyEmbeddingHelperNoCond(
                        lambda i: embedding.lookup(i), start_tokens, end_token
                    )
                else:
                    helper = GreedyEmbeddingHelperNoCond(
                        embedding, start_tokens, end_token
                    )

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell,
                helper,
                initial_state=initial_state,
                output_layer=projection_layer
                if not train
                else None,  # applied per timestep
            )

            # Dynamic decoding
            outputs, final_context_state, _ = dynamic_decode(  # Contains the XLA check
                decoder,
                maximum_iterations=tgt_length,  # Required for static TensorArrays
                output_time_major=time_major,
                swap_memory=False,
                scope=decoder_scope,
            )

            if train:
                # Specify dynamic shapes to avoid Assert
                logits = outputs.rnn_output
                logits.set_shape([tgt_length, self.opts.batch_size, atten_num_units])
                logits = projection_layer(logits)
                return outputs.sample_id, logits
            else:
                samples = outputs.sample_id
                samples.set_shape([tgt_length, self.opts.batch_size])
                return samples, outputs.rnn_output

    def _build_attention(self, encoder_outputs, decoder_cell):
        with tf.variable_scope("attention", dtype=DTYPE, use_resource=True) as scope:
            # Attention is batch major
            inputs = tf.transpose(encoder_outputs, [1, 0, 2])

            if self.opts.attention == "luong":
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.opts.num_units, inputs, dtype=scope.dtype,
                )
            else:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.opts.num_units, inputs, dtype=scope.dtype,
                )

            return AttentionWrapperNoAssert(decoder_cell, attention_mechanism)

    def _build_optimiser(self, logits, labels, mask):
        with tf.variable_scope("loss", use_resource=True):
            # Logits is dynamic so an Assert is added to check shapes
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
            train_loss = tf.reduce_sum(crossent * mask) / self.opts.batch_size

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients = [
            grad if grad is None else tf.clip_by_norm(grad, max_gradient_norm)
            for grad in gradients
        ]

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        with tf.control_dependencies([update_step]):
            mean_loss = tf.reduce_mean(train_loss, name="train_loss")
        return mean_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NMT model in TensorFlow to run on the IPU"
    )
    parser.add_argument("--infer", action="store_true", help="Inference Only")
    parser.add_argument(
        "--bi",
        action="store_true",
        help="Use bidirectional layer in encoder (with outputs summed)",
    )
    parser.add_argument(
        "--attention",
        choices=["luong", "bahdanau"],
        default="luong",
        help="Add an attention model",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Set batch-size")
    parser.add_argument(
        "--num-units", type=int, default=512, help="Number of units in each LSTM cell"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Size of LSTM stack in the encoder and decoder",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=32,
        help="Size of source and target embedding",
    )
    parser.add_argument(
        "--host-embeddings", action="store_true", help="Use host embeddings."
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Size of input length (by padding or truncating)",
    )
    parser.add_argument(
        "--ckpt", action="store_true", help="load weights from latest checkpoint"
    )
    parser.add_argument("--seed", type=int, default=1984, help="Random seed")
    parser.add_argument(
        "--interact",
        action="store_true",
        help="Perform inference on values entered from the command line",
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save the graph to './logs' to be viewed by TensorBoard",
    )
    parser.add_argument(
        "--batches-per-step", type=int, default=100, help="Number of batches per steps."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of iterations to complete in training",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    if args.interact:
        args.batch_size = 1
        args.batches_per_step = 1
        args.infer = True

    if args.host_embeddings and args.infer:
        raise NotImplementedError("Host embeddings cannot be used with inference.")

    print(
        "NMT {}.\n Batch size: {}.  Hidden units: {}.  Layers: {}.".format(
            "Inference" if args.infer else "Training",
            args.batch_size,
            args.num_units,
            args.num_layers,
        )
    )

    n = Nmt(args)
    if args.infer:
        n.infer()
    else:
        n.train()
