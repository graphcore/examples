# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
from tensorflow.python.ipu.config import IPUConfig
import numpy as np
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from ipu_sparse_ops import sparse, layers
import logging

tf.disable_eager_execution()
tf.disable_v2_behavior()


def get_program_arguments():
    parser = argparse.ArgumentParser(description='Sparse matmul tool.')
    parser.add_argument("--batch-size", type=int,
                        help="Number of vectors in a mini-batch")
    parser.add_argument("--input-size", type=int,
                        help="Input size (rows of weight matrix)")
    parser.add_argument("--output-size", type=int,
                        help="Output size (cols of weight matrix)")
    parser.add_argument("--pattern", type=str,
                        help="Choose how the sparsity pattern is generated.",
                        choices=['random', 'fixed', 'random_sign_ones', 'random_orthogonal'])
    parser.add_argument("--density", type=float,
                        help="Non-zero density (only used if --pattern is random).")
    parser.add_argument("--block-size", type=int,
                        help="Size of square non-zero blocks.",
                        choices=[1, 4, 8, 16])
    parser.add_argument("--data-type", type=str,
                        help="Choose the floating point type for the weights.",
                        choices=['fp32', 'fp16'])
    parser.add_argument('--meta-info-oversize', default=0.1, type=float,
                        help="Sets the Popsparse matmul option 'metaInfoBucketOversizeProportion'.")
    parser.add_argument('--pooling-type', default='NONE', choices=['NONE', 'SUM', 'AVG', 'MAX'],
                        help="Select dense gradients block pooling")
    parser.set_defaults(
        batch_size=2, input_size=16, output_size=8, pattern='fixed',
        density=0.1, block_size=1,
        data_type='fp32')
    return parser.parse_args()


def random_sign_ones_generator(size=[1]):
    return (2*np.random.randint(0, 2, size=size))-1


def make_fc_layer_and_test_inputs(args):
    input_size = args.input_size
    output_size = args.output_size
    batch_size = args.batch_size
    weights_type = tf.float16 if args.data_type == 'fp16' else tf.float32
    matmul_opts = {"metaInfoBucketOversizeProportion": args.meta_info_oversize}

    if args.pattern == 'fixed':
        in_blocks = input_size // args.block_size
        out_blocks = output_size // args.block_size
        identity_size = max(in_blocks, out_blocks)
        block_mask = np.identity(identity_size)[0:in_blocks, 0:out_blocks]
        block_mask[1, 3] = 1
        block_mask[0, 7] = 1
        n_blocks = np.count_nonzero(block_mask)
        el_mask = sparse.block_mask_to_element(block_mask, args.block_size)
        n_els = np.count_nonzero(el_mask)
        masked_rhs = np.zeros_like(el_mask, dtype=np.float32)
        values = np.random.rand(n_els)
        masked_rhs[np.nonzero(el_mask)] = values

        if args.block_size == 1:
                triplets = sparse.triplets_from_dense(masked_rhs)
        else:
            triplets = sparse.triplets_from_dense(block_mask)
            triplets = sparse.Triplets(
                triplets.row_indices, triplets.col_indices,
                sparse.blocks_at_indices(
                    triplets.row_indices, triplets.col_indices, args.block_size, masked_rhs)
            )
        fc = layers.SparseFcLayer.from_triplets(
            args.output_size, [args.batch_size, args.input_size], *triplets,
            matmul_options=matmul_opts,
            name='sparse_fc_from_triplets',
            dtype=weights_type,
            use_bias=False, relu=False, pooling_type=args.pooling_type)
    elif args.pattern == 'random_sign_ones':
        indices_random_gen = np.random.default_rng(seed=random_seed)
        fc = layers.SparseFcLayer.from_random_generator(
            args.output_size, [args.batch_size, args.input_size],
            args.density, args.block_size,
            random_sign_ones_generator, indices_random_gen,
            matmul_options=matmul_opts,
            name='sparse_fc_from_random_sign_ones', use_bias=False, relu=False,
            pooling_type=args.pooling_type)
        masked_rhs = sparse.dense_from_triplets(fc.weights.spec, *fc.weights.triplets)
    elif args.pattern == "random_orthogonal":
        if args.input_size != args.output_size:
            raise ValueError(
                "random_orthogonal pattern requires square matrix")
        ortho_rng = np.random.default_rng(seed=random_seed)
        matrix, max_non_zeros = sparse.gen_sparse_rand_orthog_mat(
            args.output_size,  args.density, args.block_size, ortho_rng)
        triplets = sparse.triplets_from_dense(matrix, args.block_size)

        fc = layers.SparseFcLayer.from_triplets(
            args.output_size,
            [args.batch_size, args.input_size],
            *triplets,
            matmul_options=matmul_opts,
            name='sparse_fc_random_orthogonal',
            dtype=weights_type,
            use_bias=False, relu=False,
            pooling_type=args.pooling_type)

        masked_rhs = sparse.dense_from_triplets(fc.weights.spec, *fc.weights.triplets)
    else:
        random_gen = np.random.default_rng(seed=random_seed)
        indices_random_gen = np.random.default_rng(seed=random_seed)
        fc = layers.SparseFcLayer.from_random_generator(
            args.output_size, [args.batch_size, args.input_size],
            args.density, args.block_size,
            random_gen.standard_normal, indices_random_gen,
            matmul_options=matmul_opts,
            name='sparse_fc_from_random',
            dtype=weights_type,
            use_bias=False, relu=False,
            pooling_type=args.pooling_type)
        masked_rhs = fc.weights.extract_dense()
    return fc, masked_rhs.astype(weights_type.as_numpy_dtype())

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.getLevelName("DEBUG"),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    tf.logging.set_verbosity(tf.logging.ERROR)
    np.set_printoptions(linewidth=200, threshold=20000)
    random_seed = 1
    np.random.seed(random_seed)

    # Make input data and dense and sparse weights:
    args = get_program_arguments()
    fc, masked_rhs = make_fc_layer_and_test_inputs(args)
    data_type = fc.get_data_type()
    np_dtype = data_type.as_numpy_dtype()

    if args.pattern == 'random_sign_ones':
        lhs_values = random_sign_ones_generator(size=[args.batch_size, args.input_size])
    else:
        lhs_values = np.random.rand(args.batch_size, args.input_size)

    if args.pattern == 'random_orthogonal':
        lhs_values = lhs_values / np.linalg.norm(lhs_values, axis=1, keepdims=True)

    # Configure the IPU:
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with tf.device("cpu"):
        # Placeholders for the dense matmul operands:
        lhs = tf.placeholder(shape=lhs_values.shape, dtype=data_type)
        rhs = tf.placeholder(shape=masked_rhs.shape, dtype=data_type)
        # Placeholders for the sparse representation:
        fc.create_placeholders()
        compute_dense_grad_w = tf.placeholder(tf.bool, shape=[])


    def matmul_with_grad(input, weights):
        z = tf.matmul(input, weights)
        s = tf.reduce_sum(z)
        input_grad = tf.gradients(s, input)
        weights_grad = tf.gradients(s, weights)
        return z, input_grad, weights_grad[0]


    def sparse_matmul_with_grad(fc, compute_dense_gradw, input):
        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE, use_resource=True):
            z = fc(input, compute_dense_gradw)
            s = tf.reduce_sum(z)
            input_grad = tf.gradients(s, input)
            weights_grad = tf.gradients(s, fc.get_values_var())
            return z, input_grad, weights_grad, fc.get_dense_grad_w(s)


    # Build the IPU graphs:
    sparse_matmul = partial(sparse_matmul_with_grad, fc)
    with ipu.scopes.ipu_scope("/device:IPU:0"):
        reference_fetches = ipu.ipu_compiler.compile(matmul_with_grad, [lhs, rhs])
        sparse_fetches = ipu.ipu_compiler.compile(sparse_matmul, [compute_dense_grad_w, lhs])
        with tf.variable_scope("fc", reuse=tf.AUTO_REUSE, use_resource=True):
            sparse_data_update_op = fc.update_sparsity_op()

    ipu.utils.move_variable_initialization_to_cpu()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        reference_result, reference_input_grad, reference_weight_grad = sess.run(
            reference_fetches,
            feed_dict={lhs: lhs_values, rhs: masked_rhs})

        sparse_result, sparse_input_grad, sparse_weight_grad, dense_grad_w = sess.run(
            sparse_fetches, feed_dict={lhs: lhs_values, compute_dense_grad_w: True})

    # Check all the results:

    # Convert the sparse gradient metainfo back to triplets and then use those row and col indices
    # to index the dense reference weight gradient:
    sparse_data = sparse.SparseRepresentation(fc.weights.get_metainfo(), sparse_weight_grad[0])
    triplets = sparse.triplets_from_representation(fc.weights.spec, sparse_data, fc.weights.matmul_options)
    if args.block_size == 1:
        reference_grad_nzvalues = sparse.values_at_indices(triplets[0], triplets[1], reference_weight_grad)
    else:
        reference_grad_nzvalues = sparse.blocks_at_indices(triplets[0], triplets[1], args.block_size, reference_weight_grad)
    # Convert the dense reference weight gradient to a sparse one using the same mask
    # that we used for the weights so we can compare the nzvalues against the sparse grad:
    dense_data = sparse.representation_from_triplets(fc.weights.spec, triplets[0], triplets[1], reference_grad_nzvalues, fc.weights.matmul_options)


    # Set tolerances appropriately as numpy is set for doubles by default:
    if args.pattern == 'random_sign_ones':
        rtol = 0
        atol = 0
    elif args.data_type == 'fp16':
        rtol = 1e-03
        atol = 1e-02
    elif args.pattern == 'random_orthogonal':
        rtol = 1e-07
        atol = 1e-06
    else:
        rtol = 1e-05
        atol = 1e-06

    if not np.allclose(reference_result, sparse_result, rtol=rtol, atol=atol, equal_nan=True):
        print(f"Reference result:\n{reference_result}")
        print(f"Sparse result:\n{sparse_result}")
        diff = reference_result-sparse_result
        print(f"Difference:\n{diff}")
        diff_triplet = sparse.triplets_from_dense(diff)
        print(f"Difference triplets:\nrows: {diff_triplet[0]}\ncols: {diff_triplet[1]}\nvalues: {diff_triplet[2]}")
        raise RuntimeError("Sparse and reference results do not match.")

    if not np.allclose(reference_input_grad, sparse_input_grad, rtol=rtol, atol=atol, equal_nan=True):
        raise RuntimeError("Sparse and reference input gradients do not match.")

    if not np.allclose(dense_data.nz_values, sparse_weight_grad, rtol=rtol, atol=atol, equal_nan=True):
        print(f"Reference weight grad (dense):\n{reference_weight_grad}")
        print(f"Reference weight grad (sparse):\n{dense_data.nz_values}")
        print(f"Sparse weight grad:\n{sparse_weight_grad[0]}")
        raise RuntimeError("Sparse and reference weight gradients do not match.")

    if not np.array_equal(dense_grad_w, reference_weight_grad):
        print(f"Reference weight grad (dense):\n{reference_weight_grad}")
        print(f"Dense grad W from custom op:\n{dense_grad_w}")
        raise ValueError("Custom Op's Dense Weight Grad does not match reference.")

    print("Results match.")
