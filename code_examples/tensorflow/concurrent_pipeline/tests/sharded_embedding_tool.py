# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import numpy as np
from numpy.core.fromnumeric import transpose
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
import jax
import haiku as hk
import jax.numpy as jnp
from utils import ClosureInitializer
from custom_ops import sharded


class TiedProjection(hk.Module):
    def __init__(self, embedding, **kwargs):
        self.tied_embedding = embedding
        super().__init__(**kwargs)

    def __call__(self, x):
        return jax.numpy.matmul(x, jnp.transpose(self.tied_embedding.embeddings))


def softmax_cross_entropy(logits, labels):
    nlsm = -jax.nn.log_softmax(logits)
    return jnp.take_along_axis(nlsm.squeeze(), labels, axis=-1)


def loss_fn(features, indices, labels):
    embed = hk.Embed(embedding_matrix=features, lookup_style='ARRAY_INDEX', name="tied_embedding")
    project = TiedProjection(embedding=embed, name="tied_projection")
    network = hk.Sequential([embed, project])
    logits = network(indices)
    ce = softmax_cross_entropy(logits, labels)
    loss = jnp.mean(ce)
    return loss, (jnp.argmax(logits, -1), loss)


def take_last_axis(logits, labels):
    batch_size = labels.get_shape().as_list()[0]
    batch_indices = tf.constant(np.arange(batch_size, dtype=np.int32))
    batch_indices = tf.reshape(batch_indices, [batch_size, 1])
    positions = tf.stack([batch_indices, labels], axis=-1)
    return tf.gather_nd(logits, positions)


def embedding_stage(indices, weight_shape, weight_initialiser):
    # Gather options:
    opts = {
        "availableMemoryProportion": "0.3",
    }
    weight_type = tf.float16 if args.fp16 else tf.float32
    weights = tf.get_variable('embedding_weights', dtype=weight_type, shape=weight_shape, initializer=weight_initialiser)
    weights, indices = sharded.allocate_tied_embedding(weights, indices, opts)
    shardedIndices = sharded.to_all(indices, args.ipus)
    embedded = sharded.embedding(weights, shardedIndices, opts)
    # Normal use would be that next stage is language model, then the
    # projection stage comes at the end but here we just call the
    # projection stage directly so we don't have to setup an optimizer
    # in the test:
    return projection_stage(shardedIndices, weights, embedded)


def projection_stage(shardedIndices, weights, input):
    # Projection matmul options:
    opts = {
        "availableMemoryProportion": "0.3",
    }
    # Projection using the same weights:
    inputSharded = sharded.to_all(input, args.ipus)
    output = sharded.matmul(inputSharded, tf.transpose(weights), opts, name="custom_matmul")
    ce = sharded.log_softmax_cross_entropy(output, shardedIndices)
    loss = tf.reduce_mean(ce)

    if args.no_checks:
        tied_weights_grad = tf.constant([1.0])
    else:
        tied_weights_grad = tf.gradients(loss, [weights])[0]
    return loss, tied_weights_grad


def loss_stage(loss, tied_weights_grad):
    return loss, tied_weights_grad


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()


def pipelined_test(input_indices):

    def bound_embedding(indices):
        embedding_init = ClosureInitializer(lambda: tf.constant(features))
        return embedding_stage(indices, weight_shape=features.shape, weight_initialiser=embedding_init)

    with tf.variable_scope("Test", use_resource=True):
        pipeline_op = ipu.pipelining_ops.pipeline(
            computational_stages=[bound_embedding, loss_stage],
            device_mapping=[ipu.pipelining_ops._ALL_DEVICES, args.ipus - 1],  # Need to reference last IPU
            gradient_accumulation_count=1,
            repeat_count=1,
            inputs=[input_indices],
            infeed_queue=None,
            outfeed_queue=outfeed_queue,
            optimizer_function=None,
            pipeline_schedule=ipu.pipelining_ops.PipelineSchedule.Sequential,
            outfeed_loss=False,
            name="Pipeline")
        return pipeline_op


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=20,
                        help="Vocab size (number of rows in the embedding matrix).")
    parser.add_argument("--feature-size", type=int, default=3,
                        help="Dimension of embedded vector space (number of columns ion the embedding matrix).")
    parser.add_argument("--sequence-length", type=int, default=7,
                        help="Sequence length (number of vectors to gather in embedding stage).")
    parser.add_argument("--ipus", type=int, default=2,
                        help="Number of IPUs (number of shards for matmul).")
    parser.add_argument("--no-checks", action="store_true",
                        help="Don't check results: this avoids large host streams of grads so you can run with larger embeddings.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision for embedding/projection weights (note the weight type is propagated almost everywhere).")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    features_shape = [args.vocab_size, args.feature_size]
    indices_shape = [args.sequence_length, 1]
    dtype = np.float16 if args.fp16 else np.float32

    # Generate random input data for the test.
    #
    # If we normalise every feature then we can check the output projection
    # simply by using argmax which should return the input indices (because
    # the dot product of a normalised vector with itself is maximal).
    rng = np.random.default_rng(seed=284)
    features = rng.normal(0, 1, size=features_shape).astype(dtype)
    features = normalized(features)
    if not np.isfinite(features).any():
        raise ValueError("Random Input is not finite")
    indices = rng.uniform(0, args.vocab_size, size=args.sequence_length).astype(np.int32)
    indices = np.expand_dims(indices, axis=1)

    loss_fn_t = hk.transform(loss_fn)
    loss_fn_t = hk.without_apply_rng(loss_fn_t)

    rng = jax.random.PRNGKey(42)
    params = loss_fn_t.init(rng, features, indices, indices.astype(dtype))

    grads, (argmax_logits, hk_loss) = jax.grad(loss_fn_t.apply, has_aux=True)(params, features, indices, indices)

    hk_embedding_grad = grads['tied_embedding']['embeddings']

    if not np.array_equal(argmax_logits, indices):
        raise RuntimeError("Argmax of logits should match the input indices.")

    # TensorFlow version:
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = args.ipus
    cfg.configure_ipu_system()

    with tf.device("cpu"):
        indices_ph = tf.placeholder(np.int32, indices.shape)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        ipu_graph = ipu.ipu_compiler.compile(pipelined_test, [indices_ph])
        outfeed_op = outfeed_queue.dequeue()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(ipu_graph, feed_dict={indices_ph: indices})
        tf_loss, tf_grad = sess.run(outfeed_op)

    # Tolerances suitable for fp32:
    rtol = 1e-06
    atol = 1e-05

    if args.fp16:
        # Tolerances suitable for fp16:
        rtol = 1e-03
        atol = 1e-02

    print(f"Haiku loss: {hk_loss}\nOur loss: {tf_loss}")
    if not np.allclose(tf_loss, hk_loss, rtol=rtol, atol=atol):
        Error("Loss does not match reference.")

    if args.no_checks:
        print(f"Skipping gradient tests.")
    else:
        if not np.allclose(hk_embedding_grad, tf_grad, rtol=rtol, atol=atol):
            print(f"Max ABS error: {np.max(np.abs(hk_embedding_grad - tf_grad))}")
            raise RuntimeError("Tied embedding grad does not match reference.")
        print("Results match.")
