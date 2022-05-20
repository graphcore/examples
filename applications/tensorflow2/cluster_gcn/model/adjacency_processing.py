# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import tensorflow as tf

from utilities.options import ALLOWED_ADJACENCY_MODE


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

    def __init__(self, transform_mode, diag_lambda=None, regularisation=None, **kwargs):
        """
        Arguments:
        :param transform_mode: String indicating the way the adjacency matrix should be transformed
            before applying the convolution.
        :param diag_lambda: Diagonal enhancement value, float.
        :param reg_factor: Regularisation factor value, float.
        """
        super().__init__(**kwargs)

        self.transform_mode = transform_mode
        self.diag_lambda = diag_lambda
        self.regularisation = regularisation

    @staticmethod
    def normalise_adjacency(adjacency):
        const_one = tf.constant(1.0, dtype=adjacency.dtype)
        row_sum = tf.math.reduce_sum(adjacency, axis=-1)
        norm_factor = const_one / (tf.maximum(const_one, row_sum))
        normaliser_mat = tf.linalg.diag(norm_factor)
        adj_norm = tf.linalg.matmul(normaliser_mat, adjacency, a_is_sparse=True, b_is_sparse=True)
        return adj_norm

    @staticmethod
    def regularise_adjacency(adjacency, regularisation):
        adj_reg = adjacency + regularisation * tf.eye(adjacency.shape[0], dtype=adjacency.dtype)
        return adj_reg

    @staticmethod
    def get_degree_plus_eye_inverse(adjacency):
        const_one = tf.constant(1.0, dtype=adjacency.dtype)
        row_sum = tf.math.reduce_sum(adjacency, axis=-1)
        row_sum_plus_eye = row_sum + const_one
        inv_deg_plus_eye = tf.math.divide_no_nan(const_one, row_sum_plus_eye)
        inv_deg_plus_eye_mat = tf.linalg.diag(inv_deg_plus_eye)
        return inv_deg_plus_eye_mat

    def get_normalised_adjacency_with_self_connections(self, adjacency):
        inv_deg_plus_eye_mat = self.get_degree_plus_eye_inverse(adjacency)
        adj_self_loops = adjacency + tf.eye(adjacency.shape[0], dtype=adjacency.dtype)
        norm_adj_self_conn = tf.linalg.matmul(
            inv_deg_plus_eye_mat,
            adj_self_loops,
            a_is_sparse=True,
            b_is_sparse=True
        )
        return norm_adj_self_conn

    def call(self, adjacency):
        if self.transform_mode not in ALLOWED_ADJACENCY_MODE:
            raise ValueError(f"Not valid 'adjacency_mode', "
                             f"it must be one of {ALLOWED_ADJACENCY_MODE}.")

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
            adj_reg_norm = self.regularise_adjacency(adj_norm, self.regularisation)
            return adj_reg_norm

        if self.transform_mode == "self_connections_scaled_by_degree":
            # Option 2: Eq. (10) from paper.
            # A_tilde = (D + I)^(-1) @ (A + I)
            inv_deg_plus_eye_mat = self.get_degree_plus_eye_inverse(adjacency)
            adj_self_loops = adjacency + tf.eye(adjacency.shape[0], dtype=adjacency.dtype)
            adjacency_tilde = tf.linalg.matmul(inv_deg_plus_eye_mat, adj_self_loops)
            return adjacency_tilde

        if self.transform_mode == "normalised_regularised_self_connections_scaled_by_degree":
            # Option 3: Eqs. (9) and (10) from paper.
            # A' denotes A normalised and regularised.
            # A_tilde = (D + I)^(-1) @ (A' + I)
            adj_norm = self.normalise_adjacency(adjacency)
            adj_reg_norm = self.regularise_adjacency(adj_norm, self.regularisation)
            inv_deg_plus_eye_mat = self.get_degree_plus_eye_inverse(adjacency)
            adj_reg_norm_self_loops = adj_reg_norm + tf.eye(adjacency.shape[0], dtype=adjacency.dtype)
            adjacency_tilde = tf.linalg.matmul(
                inv_deg_plus_eye_mat,
                adj_reg_norm_self_loops,
                a_is_sparse=True,
                b_is_sparse=True
            )
            return adjacency_tilde

        if self.transform_mode == "self_connections_scaled_by_degree_with_diagonal_enhancement":
            # Option 4: Eqs. (10) + (11).
            # A_1 = (D + I)^(-1) @ (A + I)
            # A_tilde = A_1 + lambda * diag(A_1)
            inv_deg_plus_eye_mat = self.get_degree_plus_eye_inverse(adjacency)
            adj_self_loops = adjacency + tf.eye(adjacency.shape[0], dtype=adjacency.dtype)
            adjacency_tilde = tf.linalg.matmul(
                inv_deg_plus_eye_mat,
                adj_self_loops,
                a_is_sparse=True,
                b_is_sparse=True
            )
            diag_adjacency_tilde = tf.linalg.diag(tf.linalg.diag_part(adjacency_tilde))
            adjacency_tilde += self.diag_lambda * diag_adjacency_tilde
            return adjacency_tilde
