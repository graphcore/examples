# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf

from utilities.constants import AdjacencyForm
from utilities.options import ALLOWED_ADJACENCY_TRANSFORM
import utilities.sparse_mat_ops as mat_ops
from utilities.sparse_mat_ops import dtype_, shape_


class AdjacencyProcessing(tf.keras.layers.Layer):
    """
    Adjacency processing layer that process the adjacency matrix before applying the convolution.
    Four alternatives are considered:
    "normalised_regularised" implements Eq. (1) from paper: A_tilde = A',
        where A' is the normalised and regularised adjacency.
    "self_connections_scaled_by_degree" implements Eq. (10): A_tilde = (D + I)^(-1) @ (A + I).
    "normalised_regularised_self_connections_scaled_by_degree" implements Eqs. (9) and (10):
        A_tilde = (D + I)^(-1) @ (A' + I), where A' is the normalised and regularised adjacency.
    "self_connections_scaled_by_degree_with_diagonal_enhancement" implements Eqs. (10) + (11):
        A_tilde = A_1 + lambda * diag(A_1), where A_1 = (D + I)^(-1) @ (A + I)
    See https://arxiv.org/pdf/1905.07953 Sec 3.3 for more details.
    """

    def __init__(
            self,
            num_nodes,
            transform_mode,
            diag_lambda=None,
            regularisation=None,
            adjacency_form=AdjacencyForm.DENSE,
            adjacency_dtype=tf.float32,
            **kwargs
    ):
        """
        Arguments:
        :param num_nodes: Number of nodes in the current adjacency matrix.
        :param transform_mode: String indicating the way the adjacency matrix should be transformed
            before applying the convolution.
        :param diag_lambda: Diagonal enhancement value, float.
        :param reg_factor: Regularisation factor value, float.
        :param adjacency_form: Integer encoding whether the adjacency is expressed
            as a dense tensor, a sparse tensor, or a tuple.
        :param cast_model_inputs_to_dtype: TF dtype to cast the values of the adjacency
            matrix when it is . If the matrix is dense, then this parameter is not used.
        """
        super().__init__(**kwargs)

        self.num_nodes = num_nodes
        self.transform_mode = transform_mode
        self.diag_lambda = diag_lambda
        self.regularisation = regularisation
        self.adjacency_form = adjacency_form
        self.adjacency_dtype = adjacency_dtype

    @staticmethod
    def normalise_adjacency(adjacency):
        const_one = tf.constant(1.0, dtype=dtype_(adjacency))
        row_sum = mat_ops.reduce_sum_rows(adjacency)
        norm_factor = const_one / tf.maximum(const_one, row_sum)
        adj_norm = mat_ops.diag_matmul(norm_factor, adjacency)
        return adj_norm

    @staticmethod
    def regularise_adjacency(adjacency, regularisation, adjacency_form):
        eye = mat_ops.eye(shape_(adjacency)[0], dtype_(adjacency), adjacency_form)
        reg_eye = mat_ops.scale_by_constant(eye, regularisation)
        adj_reg = mat_ops.add(adjacency, reg_eye)
        return adj_reg

    @staticmethod
    def get_degree_plus_eye_inverse(adjacency):
        const_one = tf.constant(1.0, dtype=dtype_(adjacency))
        row_sum = mat_ops.reduce_sum_rows(adjacency)
        row_sum_plus_eye_vec = row_sum + const_one
        inv_deg_plus_eye_vec = tf.math.divide_no_nan(const_one, row_sum_plus_eye_vec)
        return inv_deg_plus_eye_vec

    def get_normalised_adjacency_with_self_connections(self, adjacency, adjacency_form):
        eye = mat_ops.eye(shape_(adjacency)[0], dtype_(adjacency), adjacency_form)
        adj_self_loops = mat_ops.add(adjacency, eye)
        inv_deg_plus_eye_vec = self.get_degree_plus_eye_inverse(adjacency)
        norm_adj_self_conn = mat_ops.diag_matmul(inv_deg_plus_eye_vec, adj_self_loops)
        return norm_adj_self_conn

    def call(self, adjacency):
        if self.transform_mode not in ALLOWED_ADJACENCY_TRANSFORM:
            raise ValueError(f"Not valid 'adjacency_mode', "
                             f"it must be one of {ALLOWED_ADJACENCY_TRANSFORM}.")

        # If adjacency matrix is a tf.Tensor, cast from bool to the chosen float precision.
        if isinstance(adjacency, tf.Tensor):
            adjacency = tf.cast(adjacency, dtype=self.adjacency_dtype)

        # If adjacency matrix is a tuple, add shape.
        if isinstance(adjacency, tuple):
            indices, values = adjacency
            shape = tf.TensorShape((self.num_nodes, self.num_nodes))
            adjacency = (indices, values, shape)

        if self.transform_mode == "normalised":
            # Option 1: Eq. (1) from paper.
            # A' denotes A normalised.
            # A_tilde = A'
            adj_norm = self.normalise_adjacency(adjacency)
            return adj_norm

        if self.transform_mode == "normalised_regularised":
            # Option 1: Eq. (1) from paper.
            # A' denotes A normalised and regularised.
            # A_tilde = A'
            adj_norm = self.normalise_adjacency(adjacency)
            adj_reg_norm = self.regularise_adjacency(
                adj_norm,
                self.regularisation,
                self.adjacency_form
            )
            return adj_reg_norm

        if self.transform_mode == "self_connections_scaled_by_degree":
            # Option 2: Eq. (10) from paper.
            # A_tilde = (D + I)^(-1) @ (A + I)
            eye = mat_ops.eye(shape_(adjacency)[0], dtype_(adjacency), self.adjacency_form)
            adj_self_loops = mat_ops.add(adjacency, eye)
            inv_deg_plus_eye_vec = self.get_degree_plus_eye_inverse(adjacency)
            adjacency_tilde = mat_ops.diag_matmul(
                inv_deg_plus_eye_vec,
                adj_self_loops
            )
            return adjacency_tilde

        if self.transform_mode == "normalised_regularised_self_connections_scaled_by_degree":
            # Option 3: Eqs. (9) and (10) from paper.
            # A' denotes A normalised and regularised.
            # A_tilde = (D + I)^(-1) @ (A' + I)
            inv_deg_plus_eye_vec = self.get_degree_plus_eye_inverse(adjacency)
            adj_norm = self.normalise_adjacency(adjacency)
            adj_reg_norm = self.regularise_adjacency(
                adj_norm,
                self.regularisation,
                self.adjacency_form
            )
            eye = mat_ops.eye(
                shape_(adjacency)[0],
                dtype_(adjacency),
                self.adjacency_form
            )
            adj_reg_norm_self_loops = mat_ops.add(adj_reg_norm, eye)
            adjacency_tilde = mat_ops.diag_matmul(inv_deg_plus_eye_vec, adj_reg_norm_self_loops)
            return adjacency_tilde

        if self.transform_mode == "self_connections_scaled_by_degree_with_diagonal_enhancement":
            # Option 4: Eqs. (10) + (11).
            # A_1 = (D + I)^(-1) @ (A + I)
            # A_tilde = A_1 + lambda * diag(A_1)
            eye = mat_ops.eye(shape_(adjacency)[0], dtype_(adjacency), self.adjacency_form)
            adj_self_loops = mat_ops.add(adjacency, eye)
            inv_deg_plus_eye_vec = self.get_degree_plus_eye_inverse(adjacency)
            adjacency_tilde = mat_ops.diag_matmul(inv_deg_plus_eye_vec, adj_self_loops)
            adjacency_tilde_diag_part = mat_ops.diag_part(adjacency_tilde)
            diag_adjacency_tilde = mat_ops.diag(adjacency_tilde_diag_part, self.adjacency_form)

            scaled_diag_adjacency_tilde = mat_ops.scale_by_constant(
                diag_adjacency_tilde,
                self.diag_lambda
            )
            adjacency_tilde = mat_ops.add(adjacency_tilde, scaled_diag_adjacency_tilde)
            return adjacency_tilde
