# Copyright 2019 Graphcore Ltd.
# coding=utf-8
"""
Standard Variational Autoencoder
"""
from functools import partial
import json
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from models.base import BaseModel
from machinable.config_map import ConfigMap
from machinable.utils import serialize
from models.vae.architectures import encoders, decoders
from utils.ipu_utils import loops_repeat

tfd = tfp.distributions


N_IWAE_SAMPLES_TEST_DEFAULT = 5000
IWAE_SAMPLES_TEST_BATCH_SIZE_DEFAULT = 100


class VAE(BaseModel):
    """
    Vanilla VAE.
    Specific encoder and decoder architectures should be specified as strings in the config (set in machinable.yaml),
    which are keys to the functions in models/vae/architectures/__init__.py
    """

    def on_create(self):
        self.graph = tf.Graph()
        self.set_device(self.config['device'])
        self.set_network_config()
        self._t_start = np.nan
        self.latest_checkpoint_path = None

    def build_networks(self):
        """Construct encoder/decoder networks"""
        self.set_network_config()
        with self.graph.as_default():
            with self.device_config['scoper']():
                self.tfp_likelihood = tfd.Bernoulli
                self.encoder = self.get_encoder(self.config.network.encoder)
                self.decoder = self.get_decoder(self.config.network.decoder)

    def config_get_seed(self):
        return self.experiment.flags['SEED']

    def get_encoder(self, encoder_config):
        """Retrieves architecture from config, which contains name which is key to dict in
        models/vae/architectures/__init__.py"""
        kwargs = encoder_config.get('kwargs', ConfigMap()).toDict()
        return partial(encoders[encoder_config.name],
                       Z_dim=self.Z_dim,
                       dtype=self.experiment.dtype,
                       **kwargs)

    def get_decoder(self, decoder_config):
        """Retrieves architecture from config, which contains name which is key to dict in
        models/vae/architectures/__init__.py"""
        kwargs = decoder_config.get('kwargs', ConfigMap()).toDict()
        return partial(decoders[decoder_config.name],
                       output_dims=self.experiment.data_meta['data_dims'],
                       dtype=self.experiment.dtype,
                       **kwargs)

    def set_network_config(self):
        """Assign some of the config regarding the network hyperparams to internal variables"""
        self.Z_dim = self.config.network.z_dim    # Latent dimension

    def set_training_config(self):
        super().set_training_config()
        self.train_output_labels = ('ELBO',)

    def set_test_config(self):
        super().set_test_config()
        self.n_iwae_samples_te = self.config.testing.get('n_iwae_samples_te', N_IWAE_SAMPLES_TEST_DEFAULT)
        self.iwae_samples_te_batch_size = self.config.testing.get('iwae_samples_te_batch_size',
                                                                  IWAE_SAMPLES_TEST_BATCH_SIZE_DEFAULT)
        assert self.n_iwae_samples_te % self.iwae_samples_te_batch_size == 0,\
            'Number of IWAE samples must be integer multiple of iwae_samples_te_batch_size'

    def network(self, X_in):
        """End-to-end network from encoder input to decoder output"""
        # Calculate q(Z|X)
        Z_cond_X_mean, Z_cond_X_std = self.encoder(X_in)

        # Reparameterisation trick: convert samples from standard normal to samples from posterior
        Z_cond_X_samples = self.reparameterised_samples(Z_cond_X_mean, Z_cond_X_std,
                                                        [tf.shape(X_in)[0], self.Z_dim])

        # Pass through decoder to estimate p(X|Z)
        p_X_cond_Z = self.p_X_cond_Z_params(Z_cond_X_samples)

        return p_X_cond_Z, Z_cond_X_mean, Z_cond_X_std, Z_cond_X_samples

    def p_X_cond_Z_params(self, Z_cond_X_samples):
        """Estimate parameters of output distribution given the latent posterior params.
        If on IPU need to force reshape, even if not needed, otherwise get XLA-poplar mismatch error"""

        # Decode the posterior samples to calculate logit(p_{model}(X|Z))
        net_out = self.decoder(Z_cond_X_samples)

        return net_out

    def _get_loss_function(self):

        def loss(X_in, logits_out, z_cond_x_mean, z_cond_x_std, Z_cond_X_samples):
            elbo = self.elbo(X_in, logits_out, z_cond_x_mean, z_cond_x_std,
                             Z_cond_X_samples)
            return -tf.reduce_sum(elbo)   # Aggregate by summing over batch, as with VCD loss
        return loss

    def build_loss_function(self):
        self.loss = self._get_loss_function()

    def gauss_ll(self, samples, mean, std):
        # With tfp
        gaussian = tfd.MultivariateNormalDiag(mean, std)
        return gaussian.log_prob(samples)

    def stochastic_kl(self, Z_cond_X_mean, Z_cond_X_std, Z_cond_X_samples):
        log_q_Z_cond_X = self.gauss_ll(Z_cond_X_samples, Z_cond_X_mean, Z_cond_X_std)
        log_p_Z = self.gauss_ll(Z_cond_X_samples, tf.zeros_like(Z_cond_X_mean), tf.ones_like(Z_cond_X_std))
        return -(log_p_Z - log_q_Z_cond_X)

    def elbo(self, X_in, logits_out, z_cond_x_mean, z_cond_x_std, Z_cond_X_samples):
        """
        Evidence Lower Bound
        """
        log_p_X_cond_Z = self.log_likelihood(X_in, logits_out)
        kl = self.stochastic_kl(z_cond_x_mean, z_cond_x_std, Z_cond_X_samples)

        return log_p_X_cond_Z - kl

    def log_likelihood(self, X_in, network_out):
        """Log-likelihood at VAE output. Currently only Bernoulli output supported"""

        # With tfp
        logit_p_X_cond_Z = self.tfp_likelihood(**network_out)
        lls = logit_p_X_cond_Z.log_prob(X_in, name='log_p_x_cond_z')

        # Sum over the pixels in each image
        ll = tf.reduce_sum(lls, [-1])

        return ll

    def iwae_elbo(self, X_b, proposal_sampler=None, proposal_log_prob=None):
        """
        More memory-efficient implementation of importance-weighted ELBO which accumulates IWELBO
        scores over latent samples (rather than storing all values and then doing log-sum-exp).
        :param X_b: input tensor to calculate model IWELBO for
        :param proposal_sampler: function, which takes input of the required shape of tensor with samples, and returns
        samples to evaluate decoder output for
        :param proposal_log_prob: function, which takes same z_samples as input and returns their
        log-probability under proposal
        """
        # Get parameters of approximate latent posterior q(Z|X)
        Z_cond_X_mu, Z_cond_X_sigma = self.encoder(tf.identity(X_b))

        # How many samples to process in each loop?
        iwae_batch_size = self.iwae_samples_te_batch_size
        n_iwae_loops = int(self.n_iwae_samples_te / iwae_batch_size)

        # If no proposal specified use the default (approx posterior)
        def _default_proposal_log_prob(z_samples, mean, std):
            return self.gauss_ll(z_samples, mean, std)

        def _default_proposal_sampler(mean, std, z_samples_shape):
            return self.reparameterised_samples(mean, std, z_samples_shape)

        proposal_sampler = proposal_sampler or _default_proposal_sampler
        proposal_log_prob = proposal_log_prob or _default_proposal_log_prob

        def online_logsumexp_update(sum_exp_accum, to_add, current_max):
            """
            Updates accumulator for logsumexp given a batch. For more details see
            http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
            :param sum_exp_accum: tf.Tensor, length equal to batch size, vector of accumulated sumexps
            :param to_add: tf.Tensor, length equal to batch size, vector of sumexps to add on
            :param current_max: tf.Tensor, length equal to batch size, online max for each element in batch,
            may be updated within function
            :return: 2-tuple, (updated sum-exps, updated max-value normaliser)
            """
            # Are any of new arguments larger than current max?
            max_new = tf.maximum(tf.reduce_max(to_add, axis=0), current_max)

            # Update accumulator
            exp_accumulator = tf.exp(to_add - max_new) + sum_exp_accum * tf.exp(current_max - max_new)
            return exp_accumulator, max_new

        def iwae_elbo_loop(sample_index):
            """While loop, getting imporance-weighted ELBO for a sample in each iteration"""
            def _cond(sample_id, exp_elbo_accumulator, max_elbo):
                return tf.less(sample_id, n_iwae_loops)

            def _body(sample_id, exp_elbo_accumulator, max_elbo):
                # Reparameterisation trick: draw samples from standard normal and convert them to samples from q(Z|X)
                Z_cond_X_samples = proposal_sampler(Z_cond_X_mu, Z_cond_X_sigma,
                                                    (iwae_batch_size,
                                                     tf.shape(X_b)[0],
                                                     self.Z_dim))

                # Find log probability of posterior samples according to prior
                log_p_Z = self.gauss_ll(Z_cond_X_samples,
                                        tf.zeros_like(Z_cond_X_mu),
                                        tf.ones_like(Z_cond_X_sigma))

                # Find log prob of samples according to posterior
                log_q_Z_cond_X = proposal_log_prob(Z_cond_X_samples, Z_cond_X_mu, Z_cond_X_sigma)

                # Pass posterior samples through generator network
                ll_params = self.p_X_cond_Z_params(Z_cond_X_samples)

                # Find log-likelihood at output
                log_p_X_cond_Z = self.log_likelihood(X_b, ll_params)

                # Evaluate elbo for the current sample
                elbo_batch = log_p_X_cond_Z + log_p_Z - log_q_Z_cond_X

                # Apply update to sum-exp accumulator
                exp_elbo_accumulator, max_elbo = online_logsumexp_update(exp_elbo_accumulator, elbo_batch, max_elbo)

                sample_id += 1
                return [sample_id, exp_elbo_accumulator, max_elbo]

            # Initialise accumulator and maximum tracker
            exp_elbo_acc = tf.zeros((iwae_batch_size, tf.shape(X_b)[0],), self.experiment.dtype)

            # Set tracker to minimum possible value (so it will be updated in first step)
            most_negative_float = np.finfo(self.experiment.dtype_np).min
            max_elbo_tracker = tf.ones((tf.shape(X_b)[0],), self.experiment.dtype) * most_negative_float

            # Accumulate exp(ELBO) terms
            _, sum_exp_elbo, max_elbo_offset = \
                tf.while_loop(_cond, _body,
                              loop_vars=[sample_index, exp_elbo_acc, max_elbo_tracker],
                              maximum_iterations=n_iwae_loops,
                              back_prop=False)

            # Normalise log(sum_i(exp(x_i)/exp(x_max)) -> log(mean_i(exp(x_i)))
            log_n_cast = tf.log(tf.cast(self.n_iwae_samples_te, self.experiment.dtype))
            log_sum_exp_elbo = tf.log(tf.reduce_sum(sum_exp_elbo, axis=0))
            return log_sum_exp_elbo - log_n_cast + max_elbo_offset

        # Execute while loop to get IW-ELBO scores
        sample_idx = tf.constant(0)
        return iwae_elbo_loop(sample_idx),

    def reparameterised_samples(self, mu, sigma, samples_shape):
        """
        To backpropagate through sampling op, this function implements the "reparameterisation trick".
        Samples from an arbitrary normal distribution are drawn by first drawing samples
        from a standard normal and scaling them by sigma, shifting them by mu.
        Thus gradient only flows through mu and sigma, rather than through the random sampling.
        See Section 2.4 of Kingma and Welling's original paper: https://arxiv.org/pdf/1312.6114.pdf
        :param mu: Tensor, mean of normal distribution to sample from
        :param sigma: Tensor, standard deviation of normal distribution to sample from
        :param samples_shape: iterable[int], dimensions of tensor of samples to be generated
        :return: Tensor of shape samples_shape
        """
        # Draw from standard normal
        standard_norm_samples = tf.random.normal(samples_shape,
                                                 name='standard_gauss_samples_stateful',
                                                 dtype=self.experiment.dtype)

        # Rescale and translate
        new_distribution_samples = mu + sigma * standard_norm_samples

        return new_distribution_samples

    def train_ops(self, X_b):
        """Single training update"""
        loss = self.network_loss(X_b)
        ops = self.get_grad_ops(loss)
        with tf.control_dependencies(ops):
            return loss

    def get_grad_ops(self, loss):
        _learning_rate = self.get_current_learning_rate()
        opt_kwargs = self.optimiser_kwargs.copy()

        def _apply_grads_op(scope_loss, v_scope, learn_rate):
            opt = self.optimiser_type(learn_rate, **opt_kwargs)
            vars_in_scope = tf.trainable_variables(v_scope)
            var_grads = opt.compute_gradients(scope_loss, vars_in_scope, colocate_gradients_with_ops=True)
            return opt.apply_gradients(var_grads)

        def _loss_scope(scope_str=None):
            if scope_str is None or not isinstance(loss, dict):
                return loss
            else:
                return loss[scope_str]

        # Different learning rates for different variable scopes (e.g. encoder/decoder)
        tr_ops = []
        for scope, lr in _learning_rate.items():
            tr_ops.append(_apply_grads_op(_loss_scope(scope.split('/')[0]), scope, lr))

        return tr_ops

    def network_loss(self, X_batch):
        """Loss of full network. Helper method so as not to require loads of inputs in self.train_ops()"""
        network_out = self.network(tf.identity(X_batch))
        return self.loss(X_batch, *network_out)

    def validation_ops(self, X_b):
        """Evaluate network performance: KL, log-likelihood, ELBO"""
        X_b_out, Z_cond_X_mu, Z_cond_X_sigma, Z_cond_X_samples = self.network(X_b)
        kl = self.analytic_kl(Z_cond_X_mu, Z_cond_X_sigma, Z_cond_X_samples)
        ll = self.log_likelihood(X_b, X_b_out)
        return kl, ll, ll - kl

    def get_train_ops(self, graph_ops, infeed_queue, i_tr, X_b_tr, y_b_tr):
        """Add training operations to the graph"""
        possible_xla = self.device_config['maybe_xla_compile']

        # Need to close over scope of `self` for GPU XLA
        def train_op(loss, i, X, y):
            return tr(X)

        def tr(X):
            return self.train_ops(X)

        def tr_infeed():
            loss = tf.zeros(self.loss_shape, self.experiment.dtype)
            return loops_repeat(self.device_config['device'],
                                self.iters_per_sess_run,
                                train_op,
                                [loss],
                                infeed_queue,
                                maybe_xla=possible_xla)

        with self.graph.as_default():
            graph_ops['incr_global_step'] = tf.assign_add(self.global_step, self.iters_per_sess_run)
            with self.device_config['scoper']():
                if self.experiment.config.training:
                    if self.use_infeed:
                        graph_ops['train'] = tr_infeed()
                    else:
                        graph_ops['train'] = possible_xla(tr, [X_b_tr])
                    graph_ops['lr'] = self.get_current_learning_rate()
                    graph_ops['epochs'] = self.get_epoch()
        return graph_ops

    def get_validation_ops(self, graph_ops, i_val, X_b_val, y_b_val):
        """Add validation operations to the graph"""
        with self.graph.as_default():
            with self.device_config['scoper']():
                possible_xla = self.device_config['maybe_xla_compile']

                def val(X):
                    return self.validation_ops(X)

                def iwelbo(X):
                    return self.iwae_elbo(X)

                if self.experiment.config.validation:
                    graph_ops['validation'] = possible_xla(val, [X_b_val])
                    graph_ops['iwae_elbo_val'] = possible_xla(iwelbo, [X_b_val])
        return graph_ops

    def get_test_ops(self, graph_ops, i_te, X_b_te, y_b_te):
        """Add validation operations to the graph"""
        with self.graph.as_default():
            with self.device_config['scoper']():
                possible_xla = self.device_config['maybe_xla_compile']

                def iwelbo(X):
                    return self.iwae_elbo(X)

                if self.experiment.config.testing:
                    graph_ops['iwae_elbo_test'] = possible_xla(iwelbo, [X_b_te])
        return graph_ops

    def validation(self):
        """Calculate evaluation metrics of model on validation set"""
        # KL, LL and ELBO
        n_val_batches = int(np.ceil(self.config.data.n_validation / self.batch_size))

        # Importance-weighted ELBO
        op_names = {'iwae_elbo_val': ['iwae_elbo_val']}
        record_iwae = self.evaluation_scores(ops_sets_names=op_names,
                                             iters_to_init=('validation',),
                                             n_batches=n_val_batches)
        record = {}
        record.update(record_iwae)
        self.experiment.save_record(record, scope='validation')

    def test(self, max_iwae_batches=None):
        """Find model performance on full train and test sets"""
        # Test set LL, KL, ELBO
        self.experiment.log.info('Getting test accuracies for trained model...\n')
        n_te_batches = int(np.ceil(self.experiment.data_meta['test_size'] / self.batch_size_te))

        # Test set importance-weighted ELBO
        self.experiment.log.info('Finding IWELBO on test set...\n')
        op_names = {'iwae_elbo_test': ['te_iwae_elbo']}
        n_te_batches = max_iwae_batches or n_te_batches
        record_iwae_te = self.evaluation_scores(ops_sets_names=op_names,
                                                iters_to_init=('test',),
                                                n_batches=n_te_batches,
                                                verbose=True)
        self.experiment.log.info('...done.\n')

        # Print and save results
        self.experiment.log.info(f'Test results:\n{json.dumps(record_iwae_te, indent=4, default=serialize)}\n\n')
        self.experiment.save_record(record_iwae_te, scope='test')
        # self.experiment.save_record(record_tr, scope='train')

        # Save test results
        self.experiment.observer.store(f'test_results_{self.iters}_iters.json', record_iwae_te)
