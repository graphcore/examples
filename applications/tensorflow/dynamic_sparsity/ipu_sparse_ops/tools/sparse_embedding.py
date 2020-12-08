# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import numpy as np
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from ipu_sparse_ops import sparse, layers
import logging
import jax
import jax.numpy as jnp

tf.disable_eager_execution()
tf.disable_v2_behavior()


def get_program_arguments():
    parser = argparse.ArgumentParser(description='Sparse matmul tool.')
    parser.add_argument("--batch-size", type=int,
                        help="Number of vectors in a mini-batch.")
    parser.add_argument("--hidden-size", type=int,
                        help="Dimension of embedding vectors.")
    parser.add_argument("--embedding-size", type=int,
                        help="Number of token -> vector mappings in the embedding table.")
    parser.add_argument("--sequence-size", type=int,
                        help="Number of tokens to produce embeddings vectors for.")
    parser.add_argument("--pattern", type=str,
                        help="Choose how the sparsity pattern is generated.",
                        choices=['fixed'])
    parser.add_argument("--block-size", type=int,
                        help="Size of square non-zero blocks.",
                        choices=[1])
    parser.add_argument("--data-type", type=str,
                        help="Choose the floating point type for the embedding weights.",
                        choices=['fp32', 'fp16'])
    parser.add_argument('--meta-info-oversize', default=0.1, type=float,
                        help="Sets the Popsparse matmul option 'metaInfoBucketOversizeProportion'.")
    parser.add_argument('--pooling-type', default='NONE', choices=['NONE', 'SUM', 'AVG', 'MAX'],
                        help="Select dense gradients block pooling")
    parser.add_argument('--check-projection-grads-only', action='store_true',
                        help="Stop gradients after the embedding so that only projection grads are computed and compared in the test.")
    parser.set_defaults(
        batch_size=1, hidden_size=8, embedding_size=32, sequence_size=4, pattern='fixed',
        density=0.1, block_size=1, data_type='fp32')
    return parser.parse_args()


def make_embedding_layer_and_test_inputs(args):
    input_size = args.embedding_size
    output_size = args.hidden_size
    batch_size = args.batch_size
    weights_type = tf.float16 if args.data_type == 'fp16' else tf.float32
    matmul_opts = {"metaInfoBucketOversizeProportion": args.meta_info_oversize}

    if args.pattern == 'fixed':
        in_blocks = input_size // args.block_size
        out_blocks = output_size // args.block_size
        identity_size = max(in_blocks, out_blocks)
        block_mask = np.identity(identity_size)[0:in_blocks, 0:out_blocks]
        n_blocks = in_blocks
        for r in range(n_blocks):
            block_mask[r, r % out_blocks] = 1

        assert n_blocks == np.count_nonzero(block_mask)
        el_mask = sparse.block_mask_to_element(block_mask, args.block_size)
        n_els = np.count_nonzero(el_mask)
        embedding_weights = np.zeros_like(el_mask, dtype=np.float32)
        embedding_weights[np.nonzero(el_mask)] = np.random.rand(n_els)
        token_ids = np.arange(n_els).astype(np.uint32)
        np.random.shuffle(token_ids)
        token_ids = token_ids[0:args.sequence_size]
        # First create a sparse weight matrix to perform the sparse tied projection.
        # This projects from the hidden size back onto the input embedding size
        # (hence it is the transpose of the embedding matrix defined above):
        triplets = sparse.triplets_from_dense(np.transpose(embedding_weights), args.block_size)
        projection = layers.SparseFcLayer.from_triplets(
            args.embedding_size, [args.sequence_size, args.hidden_size], *triplets,
            matmul_options=matmul_opts,
            name='sparse_projection_from_triplets',
            dtype=weights_type,
            use_bias=False, relu=False, pooling_type=args.pooling_type)
        # Next create the embedding layer from the projection (tying them together):
        layer = layers.SparseTiedEmbedding.from_sparse_projection("tied_embedding", projection)
    else:
        raise RuntimeError("Invalid generator")
    return layer, embedding_weights, token_ids


def no_op(x):
    return x


def tied_embed_fwd(weights, indices, maybe_stop_gradient):
    # Compute the expected embedding esult using a one-hot matmul in numpy:
    one_hot = np.zeros([args.sequence_size, args.embedding_size], dtype=np.float32)
    for r, c in enumerate(indices):
        one_hot[r, c] = 1
    embedded = jnp.dot(one_hot, weights)
    # Project the embedded tokens directly back onto the weights:
    projected = jnp.dot(maybe_stop_gradient(embedded), jnp.transpose(weights))
    reduced = jnp.sum(projected)
    return embedded, projected, reduced


def tied_embed_grad_w_fn():
    # This function gets all the Jacobians using reverse accumulation
    # (saves us having to build more than one fwd function):
    def reduction_only(weights, indices, maybe_stop_gradient):
        return tied_embed_fwd(weights, indices, maybe_stop_gradient)[2]
    return jax.grad(reduction_only, 0)


def calc_reference_results_and_grads(args, weights, indices, maybe_stop_gradient):
    embedded, projected, reduced = tied_embed_fwd(weights, indices, maybe_stop_gradient)
    grads_weights = tied_embed_grad_w_fn()(weights, indices, maybe_stop_gradient)
    return embedded, projected, reduced, grads_weights


def sparse_embedding(maybe_stop_gradient, embed: layers.SparseTiedEmbedding, indices: tf.Tensor):
    embedded = maybe_stop_gradient(embed(indices))
    projected = embed.projection(tf.squeeze(embedded))
    s = tf.reduce_sum(projected)
    proj_grad_w = tf.gradients(s, embed.projection.get_values_var())
    return embedded, projected, proj_grad_w


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    tf.logging.set_verbosity(tf.logging.ERROR)
    np.set_printoptions(linewidth=200, threshold=20000)
    random_seed = 1
    np.random.seed(random_seed)
    args = get_program_arguments()
    data_type = tf.float16 if args.data_type == 'fp16' else tf.float32

    maybe_stop_jax_gradient = jax.lax.stop_gradient if args.check_projection_grads_only else no_op
    maybe_stop_tf_gradient = tf.stop_gradient if args.check_projection_grads_only else no_op

    embedding, embedding_weights, indices = make_embedding_layer_and_test_inputs(args)
    reference_embedded_tokens, reference_projections, ref_reduced, reference_grads_w = calc_reference_results_and_grads(
         args, embedding_weights, indices, maybe_stop_jax_gradient)

    if logger.level == logging.getLevelName("DEBUG"):
        print(f"Projection layer:\n{embedding.projection}")
        print(f"Embedding weights:\n{np.transpose(embedding_weights)}")
        print(f"Indices:\n{indices}")
        print(f"Embedded reference:\n{reference_embedded_tokens}")
        print(f"Projected reference: {reference_projections}")
        print(f"Reduced reference: {ref_reduced}")
        print(f"Reference weight grad:\n{reference_grads_w}")

    # Configure the IPU:
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 1)
    ipu.utils.configure_ipu_system(cfg)

    with tf.device("cpu"):
        # Placeholders for the sparse representation:
        embedding.projection.create_placeholders()
        compute_dense_gradw_ph = tf.placeholder(tf.bool, shape=[])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        indices_fp32 = indices.view(dtype=np.float32)
        indices_var = tf.get_variable(name="indices", dtype=tf.float32, initializer=indices_fp32)
        embedding_fetches = ipu.ipu_compiler.compile(partial(sparse_embedding, maybe_stop_tf_gradient, embedding), [indices_var])

    ipu.utils.move_variable_initialization_to_cpu()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        embedded_tokens, projections, tied_grad_w = sess.run(embedding_fetches)
        if logger.level == logging.getLevelName("DEBUG"):
            print(f"Sparse embedding result:\n{embedded_tokens}")
            print(f"Sparse projection result:\n{projections}")
            print(f"SparseTiedEmbedding grad-w:\n{tied_grad_w}")

    # Set tolerances appropriately as numpy is set for doubles by default:
    if args.data_type == 'fp16':
        rtol = 1e-03
        atol = 1e-04
    else:
        rtol = 1e-05
        atol = 1e-06

    # Check the embedding result:
    if not np.allclose(embedded_tokens, reference_embedded_tokens, rtol=rtol, atol=atol, equal_nan=True):
        print(f"Max abs error: {np.max(np.abs(embedded_tokens-reference_embedded_tokens))}")
        raise RuntimeError("Sparse and reference token embeddings do not match.")

    # Check the projection dding result:
    if not np.allclose(projections, reference_projections, rtol=rtol, atol=atol, equal_nan=True):
        print(f"Max abs error: {np.max(np.abs(projections-reference_projections))}")
        raise RuntimeError("Sparse and reference projections do not match.")

    # Convert the sparse gradient metainfo back to triplets and then use those row and col indices
    # to index the dense reference weight gradient:
    matmul_spec = embedding.projection.weights.spec
    matmul_opts = embedding.projection.weights.matmul_options
    sparse_data = sparse.SparseRepresentation(embedding.projection.weights.get_metainfo(), tied_grad_w[0])
    triplets = sparse.triplets_from_representation(matmul_spec, sparse_data, matmul_opts)
    # Reference grad is transposed with respect to popsparse one (third Jacobian is the reduction gradient wrt. weights):
    ref_grad_reduced = np.transpose(reference_grads_w)
    if args.block_size == 1:
        reference_grad_nzvalues = sparse.values_at_indices(triplets[0], triplets[1], ref_grad_reduced)
    else:
        reference_grad_nzvalues = sparse.blocks_at_indices(triplets[0], triplets[1], args.block_size, ref_grad_reduced)
    # Convert the dense reference weight gradient to a sparse one using the same mask
    # that we used for the weights so we can compare the nzvalues against the sparse grad:
    dense_data = sparse.representation_from_triplets(matmul_spec, triplets[0], triplets[1], reference_grad_nzvalues, matmul_opts)

    if logger.level == logging.getLevelName("DEBUG"):
        print(f"Tied grad-w triplets:\n{triplets}")
        print(f"Tied grad-w dense:\n{np.transpose(sparse.dense_from_triplets(matmul_spec, *triplets))}")
        print(f"Ref grad-w:\n{ref_grad_reduced}")

    if not np.allclose(dense_data.nz_values, tied_grad_w, rtol=rtol, atol=atol, equal_nan=True):
        print(f"Reference weight grad (sparsified):\n{dense_data.nz_values}")
        print(f"Computed sparse weight grad:\n{tied_grad_w[0]}")
        raise RuntimeError("Sparse and reference weight gradients do not match.")

    print("Results match.")
