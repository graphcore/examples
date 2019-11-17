# Copyright 2019 Graphcore Ltd.
import tensorflow as tf

from util import Modes


class MLPModel():
    def __init__(self, opts, mode=Modes.TRAIN, *args, **kwargs):
        self.EMBEDDING_SIZE = 10
        self.dtype, self.master_dtype = [getattr(tf, x) for x in opts.dtypes]
        self.is_training = mode == Modes.TRAIN
        # Use of Embeddings creates a model <-> data dependency
        self.data = opts.training_data if Modes.TRAIN else opts.validation_data
        self.embedding_initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, dtype=self.dtype)
        self.weights_initializer = tf.contrib.layers.xavier_initializer(dtype=self.dtype)

    def _get_variable(self, name, shape, init):
        var = tf.get_variable(
            name,
            shape,
            initializer=init,
            dtype=self.master_dtype)
        if self.master_dtype != self.dtype:
            var = tf.cast(var, dtype=self.dtype)
        return var

    def _build_embeddings(self, input_x):
        with tf.variable_scope('model', use_resource=True, reuse=tf.AUTO_REUSE):
            # Add embeddings for categorical input variables
            combined_list = []
            for (i, l) in enumerate(self.data.VOCAB_LENS):
                categorical_feature = tf.cast(input_x[:, i], tf.int32)
                embedding_var = self._get_variable(
                        "embedding_{}".format(i + 1),
                        shape=[l, self.EMBEDDING_SIZE],
                        init=self.embedding_initializer)
                one_hot = tf.one_hot(indices=categorical_feature, depth=l, dtype=self.dtype)
                embedded_col = tf.matmul(one_hot, embedding_var)
                combined_list.append(embedded_col)
            embedded_all = tf.concat(combined_list, 1, name="embedded_all")
        return embedded_all


    def _build_graph(self, input_x):
        # Build model graph
        with tf.variable_scope('model', use_resource=True, reuse=tf.AUTO_REUSE):
            # Generate the embeddings from the categorical features
            embedded_features = self._build_embeddings(input_x)

            # Generate the continuous features into a batch norm
            continuous_features = input_x[:, self.data.NUM_CATEGORICAL:(self.data.NUM_CATEGORICAL + self.data.NUM_CONTINUOUS)]
            continuous_features = tf.layers.batch_normalization(continuous_features, fused=True, center=True, scale=True,
                                                                training=self.is_training, trainable=self.is_training,
                                                                momentum=0.990, epsilon=1e-3, name='batch_norm')

            # Concatenate input variables
            x = tf.concat([embedded_features, continuous_features], 1, name="all_features")

            # Tag all layers to l2 regularise with '_l2tag'
            x = tf.layers.dense(x, 1000, activation=tf.nn.relu, kernel_initializer=self.weights_initializer, name='d0_l2tag')
            x = tf.layers.dense(x, 1000, activation=tf.nn.relu, kernel_initializer=self.weights_initializer, name='d1_l2tag')
            x = tf.layers.dense(x, 1000, activation=tf.nn.relu, kernel_initializer=self.weights_initializer, name='d2_l2tag')
            x = tf.layers.dense(x, 500,  activation=tf.nn.relu, kernel_initializer=self.weights_initializer, name='d3_l2tag')
            # Only dropout on training
            if self.is_training:
                x = tf.nn.dropout(x, rate=0.5, name='dropout')
            x = tf.layers.dense(x, 1, kernel_initializer=self.weights_initializer, name='df')
            # Scale output by log and 20%
            x = tf.nn.sigmoid(x) * self.data._log_max_sales * 1.2
        return x

    def __call__(self, x):
        return self._build_graph(x)
