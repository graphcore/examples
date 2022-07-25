# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# Copyright (c) Microsoft Corporation.
# Original code: https://github.com/vsyrgkanis/adversarial_gmm
"""
TensorFlow2 implementation of the Adversarial Generalized Method of Moments.

... as described at https://arxiv.org/abs/1803.07164

In its core, we optimize::

    min over modeler h max over critic f of E[((y-h(w))*f(x))**2]
"""
import argparse
import numpy as np
import time as time
import yaml
import tensorflow as tf
import tensorflow_probability as tfp
try:
    from tensorflow.python import ipu
except:
    pass
# set up logging
import logging_util
logger = logging_util.get_basic_logger("AdGMoM")

tfd = tfp.distributions


# function examples
def tau_fn_2dpoly(x): return -1.5 * x + .9 * (x**2)


def tau_fn_3dpoly(x): return -1.5 * x + .9 * (x**2) + x**3


def tau_fn_abs(x): return np.abs(x)


def tau_fn_linear(x): return x


def tau_fn_sigmoid(x): return 2 / (1 + np.exp(-2 * x))


def tau_fn_sin(x): return np.sin(x)


def tau_fn_step(x): return 1.0 * np.array(x < 0) + 2.5 * np.array(x >= 0)


def regression(y, w, h): return y - h.predict(w)


# data generating functions
TARGET_FUNCTION = {
    "2dpoly": tau_fn_2dpoly,
    "3dpoly": tau_fn_3dpoly,
    "abs": tau_fn_abs,
    "linear": tau_fn_linear,
    "sigmoid": tau_fn_sigmoid,
    "sin": tau_fn_sin,
    "step": tau_fn_step
}
# problem definitions
RHO = {"regression": regression}


def get_tf_data(n_samples, n_instruments, iv_strength, tau_fn):
    """Construct TF2 dataset for DGP1 from the paper."""
    e = tfd.Normal(0, 2).sample([n_samples, 1])
    x = tfd.Normal(0, 2).sample([n_samples, n_instruments])
    x1 = tf.reshape(x[:, 0], [n_samples, 1])
    w = (1-iv_strength) * x1 + iv_strength * e +\
        tfd.Normal(0, 0.1).sample([n_samples, 1])
    y = tau_fn(w) + e + tfd.Normal(0, 0.1).sample([n_samples, 1])
    return tf.constant(x), tf.constant(w), tf.constant(y)


def get_data_clustering(
    data_z, data_p, n_instruments, n_critics=50, cluster_type="kmeans",
        num_trees=5, min_cluster_size=50, critic_type="Gaussian"):
    """Return the centers, precisions, and normalizers of a data cover.
    """
    if cluster_type == "forest":
        from sklearn.ensemble import RandomForestRegressor
        dtree = RandomForestRegressor(
            n_estimators=num_trees,
            max_leaf_nodes=n_critics,
            min_samples_leaf=min_cluster_size)
        dtree.fit(data_z, data_p)
        cluster_labels = dtree.apply(data_z)
        cluster_ids = [np.unique(cluster_labels[:, c])
                       for c in range(cluster_labels.shape[1])]
    elif cluster_type == "kmeans":
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_critics).fit(data_z)
        cluster_labels = kmeans.labels_.reshape(-1, 1)
        cluster_ids = [np.unique(cluster_labels)]
    elif cluster_type == "random_points":
        center_ids = np.random.choice(
            np.arange(data_z.shape[0]), size=n_critics, replace=False)
        cluster_labels = np.zeros((data_z.shape[0], n_critics))
        cluster_ids = np.ones((n_critics, 1))
        for it, center in enumerate(center_ids):
            distances = np.linalg.norm(data_z - data_z[center], axis=1)
            cluster_members = np.argsort(
                distances)[:min_cluster_size]
            cluster_labels[cluster_members, it] = 1
    else:
        raise Exception("Unknown option {}".format(cluster_type))

    if critic_type == "Gaussian":
        # We put a symmetric gaussian encompassing
        # all the data points of each cluster of each clustering
        center_grid = []
        precision_grid = []
        normalizers = []

        data_z = np.array(data_z)

        for tree in range(cluster_labels.shape[1]):
            for leaf in cluster_ids[tree]:
                center = np.mean(
                    data_z[cluster_labels[:, tree].flatten() == leaf, :],
                    axis=0)
                distance = np.linalg.norm(
                    data_z - center, axis=1) / data_z.shape[1]
                precision = 1. / (
                    np.sqrt(2) * (np.sort(distance)[min_cluster_size]))
                normalizer = (precision ** n_instruments) * np.sum(
                    np.exp(- (precision * distance) ** 2)) / (
                                       np.power(2. * np.pi, n_instruments / 2.))
                normalizers.append(normalizer)
                center_grid.append(center)
                precision_grid.append(precision)
        # The proposed normalizing constant results in too small function values
        # which result in too small losses and respective lack of scaling
        # when using the exp function for the weights update.
        # The code is kept for future fixes but overwritten
        # with the following command.
        # TODO: Explore normalizers and normalization of f(x)
        normalizers = np.ones(len(center_grid), dtype="float32")
        normalizers = np.array(normalizers, dtype="float32")
        center_grid = np.array(center_grid, dtype="float32")
        precision_grid = np.array(precision_grid, dtype="float32")

        normalizers = tf.constant(normalizers, name="normalizers")
        center_grid = tf.constant(center_grid, name="centers")
        precision_grid = tf.constant(precision_grid, name="precisions")
    else:
        raise NotImplementedError("Uniform functions not supported.")

    return normalizers, precision_grid, center_grid


class KerasModeler:
    """Map modeler to keras API."""
    def __init__(
            self, layer_dims, output_dims, optimizer,
            reg_params={}):
        kernel_regularizer = None
        use_l1_reg = ("l1" in reg_params and reg_params["l1"])
        use_l2_reg = ("l2" in reg_params and reg_params["l2"])
        if use_l1_reg and use_l2_reg:
            kernel_regularizer = tf.keras.regularizers.l1_l2(
                l1=reg_params["l1"], l2=reg_params["l2"])
        elif use_l1_reg:
            kernel_regularizer = tf.keras.regularizers.l1(reg_params["l1"])
        elif use_l2_reg:
            kernel_regularizer = tf.keras.regularizers.l2(reg_params["l2"])
        self._optimizer = optimizer
        sequence = []
        for i, layer_dim in enumerate(layer_dims):
            sequence.append(tf.keras.layers.Dense(
                layer_dim, activation="relu", name="layer_{}".format(i + 1),
                kernel_regularizer=kernel_regularizer))
        sequence.append(tf.keras.layers.Dense(
            output_dims, name="output", kernel_regularizer=kernel_regularizer
        ))
        self.model = tf.keras.Sequential(sequence, name="modeler")
        self.model.compile()

    def predict(self, data_w):
        """Wrapper to keras model predict that enables backpropagation."""
        # With model.predict no gradient would be available.
        with tf.name_scope("KerasModelerPredict"):
            return self.model(data_w)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def optimizer(self):
        return self._optimizer


class GaussianVectorizedCritic:
    """Provide critic parameters and calculation.

    Based on original GaussianCritic but compressing the list.
    There is only one adam optimizer for all parameters.
    """
    def __init__(
        self, optimizer, center, precision, normalizer, n_instruments, n_critics,
        learning_rate_hedge=0.01, num_reduced_dims=2,
    ):
        self.size_k = n_critics  # number of critics
        with tf.name_scope("CriticInit"):
            self.num_reduced_dims = min(num_reduced_dims, n_instruments)
            self._center = tf.Variable(
                center, name="center", trainable=True,
                constraint=lambda x: tf.clip_by_value(
                    x, center-0.4*tf.abs(center), center+0.4*tf.abs(center))
            )
            self._precision = tf.Variable(
                precision, name="precision", trainable=True,
                constraint=lambda x: tf.clip_by_value(
                    x, 0.2*precision, 1.2*precision)
            )
            self._normalizer = normalizer  # not trainable
            self._translation = tf.Variable(
                tf.random.normal(
                    [self.size_k, n_instruments, self.num_reduced_dims],
                    0, 1, tf.float32
                ),
                name="translation", trainable=True)

            self._optimizer = optimizer
            self._center_trainable = True
            self._precision_trainable = True
            self._translation_trainable = True
            self.learning_rate_hedge = learning_rate_hedge

            # critic weights for hedge update
            self.weights = tf.Variable(
                tf.ones([n_critics], dtype=tf.dtypes.float32) / n_critics,
                name="critic_weights", trainable=False)

    def predict(self, data_x):
        """Cast arrays to the right shapes and forward to Gaussian kernel."""
        with tf.name_scope("CriticPredict"):
            n_instruments = data_x.shape[1]
            size_n = data_x.shape[0]
            size_d = n_instruments
            size_k = self.size_k

            translation = tf.reshape(
                self._translation,
                [1, size_k, n_instruments, self.num_reduced_dims])

            # V parameter is in paper with W=V*V^T
            normalized_translation = tf.nn.l2_normalize(
                translation, 2, epsilon=1e-7
            )

            # data preprocessing to arrange for batch operation
            # The first two dimensions become batch dimension:
            # One for the data batch and the other for the critics batch.
            # We do it this way because `tf.matmul` treats all dimensions
            # except for the last two in each tensor as batch dimensions.
            t_x = tf.reshape(data_x, [size_n, 1, 1, size_d])
            casted_center = tf.reshape(tf.transpose(self._center),
                                       [1, size_k, 1, size_d])

            # data broadcasting for batch operation
            t_x = tf.broadcast_to(t_x, tf.constant([size_n, size_k, 1, size_d]))
            casted_center = tf.broadcast_to(
                casted_center, tf.constant([size_n, size_k, 1, size_d]))
            t_normalized_translation = tf.broadcast_to(
                normalized_translation,
                tf.constant(
                    [size_n, size_k, n_instruments, self.num_reduced_dims]))

            # Note that (a^T V)((a^T V)^T) = a^T*VV^T*a = a^T*W*a
            # where `a = x - center`
            # thus allowing us to calculate the W-matrix weight norm
            # in `gaussian_kernel`
            norm_input = tf.matmul(
                t_x - casted_center, t_normalized_translation)

            batch_kernel_data = tf.reshape(
                norm_input,
                [size_n, size_k, self.num_reduced_dims])

            output = gaussian_kernel(
                batch_kernel_data,
                precision=self._precision,
                normalizer=self._normalizer
            )
            return output

    # the main variables
    @property
    def center(self):
        return self._center

    @property
    def precision(self):
        return self._precision

    @property
    def translation(self):
        return self._translation

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def trainable_variables(self):
        vars = []
        if self._center_trainable:
            vars.append(self._center)
        if self._precision_trainable:
            vars.append(self._precision)
        if self._translation_trainable:
            vars.append(self._translation)
        return vars


def gaussian_kernel(x, precision, normalizer):
    """A multi-dimensional symmetric gaussian kernel."""
    with tf.name_scope("GaussianKernel"):
        dimension = x.get_shape().as_list()[-1]
        last = tf.pow(2. * np.pi, dimension / 2.)
        pre_last = tf.reduce_sum(tf.pow(x, 2), axis=-1, keepdims=True)
        pre_last_reduced = tf.reshape(pre_last, x.get_shape().as_list()[:-1])
        y = tf.math.multiply(tf.pow(precision, 2) / 2., pre_last_reduced)
        w = tf.exp(-y) / last
        t = tf.pow(tf.abs(precision), dimension)
        kernel = tf.math.multiply(t, w)
        return tf.math.multiply(1. / normalizer, kernel)


def expected_values(data, modeler, critic):
    """Calculate one expected value per critic."""
    with tf.name_scope("ExpectedValue"):
        result = tf.math.reduce_mean(tf.multiply(
            RHO[conf.rho](data["Y"], data["W"], modeler), critic.predict(data["X"])),
            axis=0)
        return result


@tf.function(experimental_compile=True)
def training_loop(modeler, critic):
    """Main IPU training loop with inner computational graph."""
    def computational_graph(max_loss, s1, s2, s3):
        weights = critic.weights
        with tf.name_scope("Exp1"):
            e1 = tf.stop_gradient(expected_values(s1, modeler, critic))

        with tf.name_scope("Exp3"):
            u_ft = tf.pow(
                tf.stop_gradient(expected_values(
                    s3, modeler, critic)), 2)

        variables = modeler.trainable_variables + critic.trainable_variables

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(variables)
            with tf.name_scope("Exp2"):
                e2 = expected_values(s2, modeler, critic)

            with tf.name_scope("ModellerSummation"):
                total_modeler_sum = 2 * tf.reduce_sum(tf.multiply(
                    tf.multiply(
                        e2, tf.stop_gradient(weights)), e1
                ))
                # add kernel regularization loss
                regularization = modeler.model.losses
                if regularization:
                    total_modeler_sum += tf.math.add_n(regularization)
            with tf.name_scope("CriticsSummation"):
                total_critics_sum = - 2 * tf.reduce_sum(tf.multiply(e2, e1))

        with tf.name_scope("ModelerGradientTape"):
            modeler_gradients = tape.gradient(
                total_modeler_sum, modeler.trainable_variables)

        with tf.name_scope("CriticGradientTape"):
            critic_gradients = tape.gradient(
                total_critics_sum, critic.trainable_variables)

        del tape

        with tf.name_scope("ModelerGradientUpdate"):
            modeler.optimizer.apply_gradients(
                zip(modeler_gradients, modeler.trainable_variables))

        with tf.name_scope("CriticGradientUpdate"):
            critic.optimizer.apply_gradients(
                zip(critic_gradients, critic.trainable_variables))

        with tf.name_scope("CriticWeightsUpdate"):
            learning_rate_hedge = critic.learning_rate_hedge
            unscaled_weights = tf.multiply(
                tf.exp(learning_rate_hedge * u_ft), weights)
            scale = tf.reduce_sum(unscaled_weights)
            scaled_weights = unscaled_weights / scale
            critic.weights.assign(scaled_weights)

        loss = tf.reduce_max(u_ft)

        return tf.math.maximum(loss, max_loss)
    max_loss = -1e6
    max_loss = ipu.loops.repeat(
        conf.iterations_per_loop, computational_graph, [max_loss], infeed_queue)
    return max_loss


def cpu_graph_wrapper(modeler, critic):
    """Decorator to map modeler and critic appropriately."""
    @tf.function(experimental_compile=True)
    def cpu_computational_graph_copy(max_loss, s1, s2, s3):
        """Copy of computational_graph to use on CPU."""
        weights = critic.weights
        with tf.name_scope("Exp1"):
            e1 = tf.stop_gradient(expected_values(s1, modeler, critic))

        with tf.name_scope("Exp3"):
            u_ft = tf.pow(
                tf.stop_gradient(expected_values(
                    s3, modeler, critic)), 2)

        variables = modeler.trainable_variables + critic.trainable_variables

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(variables)
            with tf.name_scope("Exp2"):
                e2 = expected_values(s2, modeler, critic)

            with tf.name_scope("ModellerSummation"):
                total_modeler_sum = 2 * tf.reduce_sum(tf.multiply(
                    tf.multiply(
                        e2, tf.stop_gradient(weights)), e1
                ))
            with tf.name_scope("CriticsSummation"):
                total_critics_sum = - 2 * tf.reduce_sum(tf.multiply(e2, e1))

        with tf.name_scope("ModelerGradientTape"):
            modeler_gradients = tape.gradient(
                total_modeler_sum, modeler.trainable_variables)

        with tf.name_scope("CriticGradientTape"):
            critic_gradients = tape.gradient(
                total_critics_sum, critic.trainable_variables)

        del tape

        with tf.name_scope("ModelerGradientUpdate"):
            modeler.optimizer.apply_gradients(
                zip(modeler_gradients, modeler.trainable_variables))

        with tf.name_scope("CriticGradientUpdate"):
            critic.optimizer.apply_gradients(
                zip(critic_gradients, critic.trainable_variables))

        with tf.name_scope("CriticWeightsUpdate"):
            learning_rate_hedge = critic.learning_rate_hedge
            unscaled_weights = tf.multiply(
                tf.exp(learning_rate_hedge * u_ft), weights)
            scale = tf.reduce_sum(unscaled_weights)
            scaled_weights = unscaled_weights / scale
            critic.weights.assign(scaled_weights)

        loss = tf.reduce_max(u_ft)

        return tf.math.maximum(loss, max_loss)
    return cpu_computational_graph_copy


def add_conf_args():
    """Define the argument parser object."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_conf_yaml", type=str, default="AdGMoM_conf_default.yaml",
        help="Path to yaml file containing model configuration params.")
    return parser


def get_conf(parser, print_model_conf=True):
    """Parse the arguments and set the model configuration parameters."""
    conf = parser.parse_args()

    with open(conf.model_conf_yaml, "r") as f:
        model_conf = yaml.safe_load(f)
    for k in model_conf.keys():
        setattr(conf, k, model_conf[k])

    conf.iv_strength = np.array([conf.iv_strength],
                                dtype=np.float32).reshape(-1, 1)
    conf.learning_rate_modeler = tf.constant(conf.learning_rate_modeler,
                                             name="learning_rate_modeler")
    conf.learning_rate_hedge = tf.constant(conf.learning_rate_hedge,
                                           name="learning_rate_hedge")
    conf.learning_rate_critic_gradient = tf.constant(
        conf.learning_rate_critic_gradient,
        name="learning_rate_critic_gradient")

    if print_model_conf:
        logger.info("Model configuration params:")
        logger.info("\n\n" + yaml.dump(model_conf))

    return conf


def configure_ipu(conf):
    """Reserve IPUs, setup profiling, and comment on debug flags."""
    if conf.use_ipu:
        # debug flags
        import os
        if conf.gen_report:
            os.environ.pop("TF_POPLAR_FLAGS")

        if conf.dry_run:
            os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

        if conf.synthetic:
            # Test (only roughly) if data pipeline is slowing down processing.
            # This method might still be faster than having the full data on
            # device.
            os.environ["TF_POPLAR_FLAGS"] += \
                " --use_synthetic_data --synthetic_data_initializer=random"

        if conf.xla_dump:
            # Dump xla files for debugging.
            os.environ["XLA_FLAGS"] = \
                "--xla_dump_to=XLA_graphs " \
                "--xla_dump_hlo_pass_re=forward-allocation " \
                "--xla_hlo_graph_sharding_color " \
                "--xla_dump_hlo_as_text"

        if not conf.dry_run:
            # Configure the IPU system
            cfg = ipu.config.IPUConfig()
            cfg.auto_select_ipus = conf.replication_factor
            cfg.matmuls.poplar_options = {
                'availableMemoryProportion': conf.availableMemoryProportion
            }
            cfg.configure_ipu_system()

    return


if __name__ == "__main__":
    # parse arguments and configure IPU
    parser = add_conf_args()
    conf = get_conf(parser)
    configure_ipu(conf)

    # Generate data
    data_x, data_w, data_y = get_tf_data(
        conf.n_samples, conf.n_instruments, conf.iv_strength,
        TARGET_FUNCTION[conf.target_function])
    normalizers, precision_grid, center_grid = get_data_clustering(
        data_x, data_w, conf.n_instruments, n_critics=conf.n_critics,
        cluster_type=conf.cluster_type)
    # for expected value calculation (just forward pass)
    first_dataset = tf.data.Dataset.from_tensor_slices(
        {"Y": data_y, "W": data_w, "X": data_x}
    ).shuffle(data_y.shape[0]).batch(
        conf.batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE).cache().repeat()
    # for expected value of gradient for critic and modeler
    second_dataset = tf.data.Dataset.from_tensor_slices(
        {"Y": data_y, "W": data_w, "X": data_x}
    ).shuffle(data_y.shape[0]).batch(
        conf.batch_size, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE).cache().repeat()
    # potentially bigger dataset for critic weights
    third_dataset = tf.data.Dataset.from_tensor_slices(
        {"Y": data_y, "W": data_w, "X": data_x}
    ).shuffle(data_y.shape[0]).batch(
        conf.batch_size_hedge,
        drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE).cache().repeat()
    # merge datasets
    full_dataset = tf.data.Dataset.zip(
        (first_dataset, second_dataset, third_dataset))

    # create distribution strategy
    if conf.use_ipu:
        # Create an IPU distribution strategy
        strategy = ipu.ipu_strategy.IPUStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        # modeler
        keras_modeler_opt = tf.keras.optimizers.Adam(
            learning_rate=conf.learning_rate_modeler)
        keras_modeler = KerasModeler(
            conf.hidden_layers,
            conf.n_outcomes,
            optimizer=keras_modeler_opt,
            reg_params={"l1": conf.l1_regularization,
                        "l2": conf.l2_regularization}
        )

        # critic
        critic = GaussianVectorizedCritic(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=conf.learning_rate_critic_gradient),
            center=center_grid,
            precision=precision_grid,
            normalizer=normalizers,
            n_instruments=conf.n_instruments,
            n_critics=conf.n_critics,
            num_reduced_dims=2,
            learning_rate_hedge=conf.learning_rate_hedge)

        # IPU setup
        if conf.use_ipu:
            ipu.utils.move_variable_initialization_to_cpu()
            # IPU input data setup
            infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(full_dataset)
            infeed_queue.initializer
        else:
            # CPU input data setup
            infeed_queue = iter(full_dataset)

        start_time = time.time()
        header = False
        # train the model
        if conf.use_ipu:
            for loop_ind in range(conf.n_steps // conf.iterations_per_loop + 1):
                loss = strategy.run(
                    training_loop, args=[keras_modeler, critic])

                # repetition does not benefit report
                if conf.gen_report:
                    break
                end_time = time.time()
                if not header:
                    print()
                    print("Iterations", "\t", "duration/repl. factor", "\t",
                          "max violation")
                    header = True
                # report
                # The warm up run at first iteration run can be ignored.
                print(loop_ind * conf.iterations_per_loop, "\t\t",
                      # With replication, we process a multiple of the
                      # batches and hence more data. This is corrected here.
                      (end_time - start_time) / conf.replication_factor,
                      "\t", float(loss))
                start_time = end_time
        else:
            # CPU init
            cpu_graph = cpu_graph_wrapper(keras_modeler, critic)
            for loop_ind, (s1, s2, s3) in enumerate(infeed_queue):
                loss = -1e6
                loss = cpu_graph(
                    loss, s1, s2, s3)
                if not header:
                    print()
                    print("Iterations", "\t", "duration", "\t", "max violation")
                    header = True
                # report
                # The warm up run at iteration 0 can be ignored.
                if loop_ind % conf.iterations_per_loop == 0:
                    end_time = time.time()
                    print(loop_ind, "\t\t",
                          end_time - start_time, "\t", float(loss))
                    start_time = end_time
                # done
                if loop_ind > conf.n_steps:
                    break
