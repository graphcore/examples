# Copyright 2019 Graphcore Ltd.
# coding=utf-8
"""
VAE trained with the Variational Contrastive Divergence loss proposed by Ruiz and Titsias, ICML 2019. See paper:
https://arxiv.org/pdf/1905.04062.pdf. This approach combines a standard VAE with some MCMC in the latent space.
"""
import json
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from machinable.utils import serialize

from models.hmc import HamiltonianMonteCarlo
from utils.ipu_utils import get_device_scope_call, loops_repeat, get_ipu_tf
from models.vae.vae_base import VAE

tfd = tfp.distributions
ipu = get_ipu_tf()


class VCDVAE(VAE):

    def build_loss_function(self):

        def g(x, logits, z, mu_z, sigma_z):
            """
            g(z) \triangleq \log p(x,z) + \frac{1}{2}(z - \mu)^T\Sigma^{-1}(z-\mu) from Appendix C (equation (24))
            As in Appendix C of the paper.
            """
            log_p_X_Z = self.log_likelihood(x, logits) + self.gauss_ll(z, tf.zeros_like(mu_z), tf.ones_like(sigma_z))
            return log_p_X_Z + tf.reduce_sum(0.5 * ((z - mu_z) / sigma_z) ** 2, axis=-1)

        def loss(X_in, idx, control_var, logits_out, logits_out_hmc, z_cond_x_mean, z_cond_x_std,
                 Z_cond_X_samples, Z_cond_X_samples_hmc, step_size):
            """Loss-function as per Appendix C - Particularizations of the Gradient"""
            log_pz0 = self.gauss_ll(Z_cond_X_samples,
                                    tf.zeros_like(z_cond_x_mean),
                                    tf.ones_like(z_cond_x_std))
            log_pzt = self.gauss_ll(Z_cond_X_samples_hmc,
                                    tf.zeros_like(z_cond_x_mean),
                                    tf.ones_like(z_cond_x_std))
            log_qz0_cond_x_stop_z = self.gauss_ll(tf.stop_gradient(Z_cond_X_samples),
                                                  z_cond_x_mean,
                                                  z_cond_x_std)
            log_px_cond_z0 = self.log_likelihood(X_in, logits_out)
            log_px_cond_zt = self.log_likelihood(X_in, logits_out_hmc)

            log_pxz0 = log_px_cond_z0 + log_pz0
            g_zT_stop_z = g(X_in, logits_out_hmc, tf.stop_gradient(Z_cond_X_samples_hmc), z_cond_x_mean, z_cond_x_std)

            # Combine loss terms.
            # NOTE - must sum over batch, not average! Verified empirically to be correct
            enc_loss = tf.reduce_sum(-log_pxz0 + g_zT_stop_z +
                                     tf.stop_gradient(g_zT_stop_z - control_var) * log_qz0_cond_x_stop_z)
            dec_loss = -tf.reduce_sum(log_px_cond_zt)

            # Calculate control variate update - execute update it if not on ipu and not using local cv
            # Also if using infeed, save update until after all infeed loops finished
            control_var_update = tf.stop_gradient(g_zT_stop_z)
            use_global_cv = self.use_control_variate and not self.use_local_control_variate
            on_cpu_or_gpu = not self.device_config['on_ipu']
            if not self.use_infeed and (use_global_cv or on_cpu_or_gpu):
                cvar = self.get_control_var(idx)[0]     # Gets tf variable to update
                control_var_update = self.maybe_update_control_variate(cvar, idx, control_var_update,
                                                                       decay=self.control_var_decay)

            # Calculate some diagnostics
            diagnostics = {}
            Z_dim_float = tf.cast(self.Z_dim, self.experiment.dtype)
            entropy = 0.5 * Z_dim_float * tf.log(tf.cast(2 * np.pi, self.experiment.dtype)) +\
                tf.reduce_sum(tf.log(z_cond_x_std), -1) + Z_dim_float * 0.5
            diagnostics['stoch_vcd'] = \
                -(tf.reduce_mean(log_px_cond_z0 + log_pz0) + 0.5 * Z_dim_float) + tf.reduce_mean(g_zT_stop_z)
            diagnostics['elbo'] = tf.reduce_mean(log_px_cond_z0 + log_pz0 + entropy)
            diagnostics['log_lik_qt'] = tf.reduce_mean(log_px_cond_zt + log_pzt)
            diagnostics['log_lik_q'] = tf.reduce_mean(log_px_cond_z0 + log_pz0)
            diagnostics['control_var_mean'] = tf.reduce_mean(control_var)

            return [enc_loss, dec_loss], control_var_update, step_size, diagnostics

        # Check we're actually using MCMC - if not revert to standard VAE loss
        no_hmc = self.hmc_train.n_post_burn_steps + self.hmc_train.n_burn_in_steps == 0
        if no_hmc:
            self.experiment.log.warn('Number of HMC and burn in steps both 0. Reverting to standard VAE.\n')

            # Get standard VAE (negative-ELBO) loss function
            neg_elbo_loss = super()._get_loss_function()

            # Adjust so it has consistent inputs/outputs with VCD loss
            def loss_vae(X_in, idx, control_var, logits_out,
                         logits_out_hmc, z_cond_x_mean, z_cond_x_std,
                         Z_cond_X_samples, Z_cond_X_samples_hmc, step_size):
                enc_loss = neg_elbo_loss(X_in, logits_out, z_cond_x_mean, z_cond_x_std, Z_cond_X_samples)
                dec_loss = enc_loss
                cv_update = tf.zeros((self.batch_size,), dtype=self.experiment.dtype)
                step_size = 0.
                diagnostics = {}
                return [enc_loss, dec_loss], cv_update, step_size, diagnostics

            self.loss = loss_vae

        else:
            # Otherwise use the VCD loss
            self.loss = loss

    def get_log_p_X_Z_fn(self, X, Z):
        """Returns function to evaluate joint log-prob over observed and latents,
        for a fixed X, given input Z. Used for HMC target."""
        # Standard normal latent prior
        p_z = tfd.MultivariateNormalDiag(tf.zeros_like(Z), tf.ones_like(Z))

        def _log_p_x_cond_z(Z_t):
            net_out = self.p_X_cond_Z_params(Z_t)
            o = self.log_likelihood(X, net_out)
            return o

        def _unnormalised_target(z_t):
            return p_z.log_prob(z_t) + _log_p_x_cond_z(z_t)
        return _unnormalised_target

    def set_training_config(self):

        # Control variate config
        self.use_control_variate = self.config.training.control_variate.get('use_control_variate', True)
        self.use_local_control_variate = self.config.training.control_variate.get('use_local_control_variate',
                                                                                  self.use_control_variate)
        self.use_local_control_variate = self.use_local_control_variate if self.use_control_variate else False
        self.control_var_decay = self.config.training.control_variate.get('decay', 0.9)
        self.control_var_independent_iters = self.config.training.control_variate.get('independent_iterations', 3000)

        # Set function to update control variate
        on_ipu_or_cpu = self.device_config['on_ipu'] or 'cpu' in self.device_config['device'].lower()
        if self.use_local_control_variate:
            self.control_var_device = get_device_scope_call('/device:CPU:0' if on_ipu_or_cpu else '/device:GPU:0')
            self.maybe_update_control_variate = self._update_local_control_variate
        elif self.use_control_variate:
            self.control_var_device = self.device_config['scoper']
            self.maybe_update_control_variate = self._update_global_control_variate
        else:
            self.control_var_device = self.device_config['scoper']

            def dont_update_cv(control_variate, idx, elbo_hmc, assign=True, decay=0.9):
                """For consistent inputs/outputs with self._update_global_control_variate()
                and self._update_local_control_variate()"""
                return tf.zeros((), dtype=self.experiment.dtype)

            self.maybe_update_control_variate = dont_update_cv

        # HMC config
        self.hmc_train = HamiltonianMonteCarlo(self.config.training.mcmc, self.config.batch_size)

        # Set config for normal VAE stuff
        super().set_training_config()
        self.loss_shape = (2,)
        self.train_output_labels = ('Loss [encoder, decoder]',
                                    'Average control variate',
                                    'Average Unnorm. ELBO (HMC)',
                                    'HMC step size')

    def set_test_config(self):
        super().set_test_config()

        self.hmc_test = HamiltonianMonteCarlo(self.config.testing.mcmc,
                                              self.batch_size_te,
                                              testing=True,
                                              keep_samples=True)

    def _update_local_control_variate(self, control_variate, idx, elbo_hmc, assign=True, decay=0.9):
        """
        Moving average update of control variate if using a control variate for each point in training set.
        Assumes that for first few (self.control_var_independent_iters) iterations, use global control variate,
        and then remaining iterations use local one. Note that sometimes we want to update the control variate,
        sometimes we don't. This is controlled by the `assign` argument.
        """
        # Find update for relevant elements in control_variate vector
        # and calculate their updated values
        control_variate_local_update = decay * tf.gather(control_variate, idx) + (1. - decay) * elbo_hmc

        # Find update if using using global control variate at current iteration
        control_variate_global_update = decay * control_variate[0] + (1. - decay) * tf.reduce_mean(elbo_hmc)
        if assign:
            # control_variate is a variable
            control_variate_local = tf.scatter_update(control_variate, idx, control_variate_local_update)
        else:
            # control_variate is a tensor
            control_variate_local = tf.tensor_scatter_nd_update(control_variate,
                                                                tf.expand_dims(idx, 1),
                                                                control_variate_local_update)

        # Tile scalar control variate to match shape of local control variate
        control_variate_global = tf.fill(control_variate.get_shape(), control_variate_global_update)
        new_cv_value = tf.where(self.global_step > self.control_var_independent_iters,
                                control_variate_local, control_variate_global)
        if assign:
            cv_update = control_variate.assign(new_cv_value)
            return cv_update
        else:
            return new_cv_value

    def _update_global_control_variate(self, control_variate, idx, elbo_hmc, assign=True, decay=0.9):
        """Moving average update for global control variate. If assign is True will also update
        value of control_variate variable"""
        new_cv_value = decay * control_variate + (1. - decay) * tf.reduce_mean(elbo_hmc)
        if assign:
            return control_variate.assign(new_cv_value)
        else:
            return new_cv_value

    def get_control_var(self, i_tr):
        """Get the control_variate variable and, if using local control_variate, the elements indexed by i_tr"""
        cv_shp = (self.experiment.data_meta['train_size'],) if self.use_local_control_variate else ()
        with self.control_var_device():
            with tf.variable_scope('cv_scope', reuse=tf.AUTO_REUSE, use_resource=True):
                cv_var = tf.get_variable('control_variate',
                                         shape=cv_shp,
                                         dtype=self.experiment.dtype,
                                         initializer=tf.zeros_initializer(),
                                         use_resource=True,
                                         trainable=False)
            cv_idxed = tf.gather(cv_var, i_tr) if self.use_local_control_variate else cv_var
            return cv_var, cv_idxed

    def get_train_ops(self, graph_ops, infeed_queue, i_tr, X_b_tr, y_b_tr):
        with self.graph.as_default():
            # Global step counter update
            graph_ops['incr_global_step'] = tf.assign_add(self.global_step, self.iters_per_sess_run)

            # Whether to use XLA
            possible_xla = self.device_config['maybe_xla_compile']

            def infeed_train_op(loss, cvar, elbo, stepsize, i, X, y):
                """Run the training update with IPU infeeds or, for other hardware,
                run multiple training updates in a tf.while_loop()"""
                with self.device_config['scoper']():   # TODO: could this scope be removed?
                    # If using local control variate and on GPU, retrieve full variable
                    # (cvar fn arg is actually a tensor after being passed as arg into xla.compile)
                    retrieve_cv_as_var = self.use_local_control_variate and \
                                         'gpu' in self.device_config['device'].lower()
                    if retrieve_cv_as_var:
                        cvar, _ = self.get_control_var(i)

                    # Index elems of control var corresponding to input data, if using local cvar
                    cv_batch = tf.gather(cvar, i) if self.use_local_control_variate else cvar

                    # Do train update with given control variate(s)
                    tr_loss, elbo_hmc, hmc_step_size, diagnost = train_op_batch(X, i, cv_batch)

                    # If using the control variate variable update it, else update
                    # the corresponding tensor which will be fed into next loop
                    update_cv = self.maybe_update_control_variate(cvar,
                                                                  i,
                                                                  elbo_hmc,
                                                                  assign=retrieve_cv_as_var,
                                                                  decay=self.control_var_decay)
                with tf.control_dependencies([tr_loss, update_cv]):
                    return [tf.identity(tr_loss), update_cv, elbo_hmc, hmc_step_size]

            def train_op_batch(X, idx_tr, cvar):
                with self.device_config['scoper']():   # TODO: could this scope be removed?
                    return self.vcd_train_ops(X, idx_tr, cvar)

            def tr_infeed(cvar):

                if self.device_config['on_ipu'] or not self.device_config['do_xla']:
                    batch_size = X_b_tr.get_shape()[0]
                else:
                    # If using XLA on GPU/CPU, infeed queue is replaced by stacked tensor,
                    # which is iterated over in loops_repeat()
                    batch_size = X_b_tr.get_shape()[0] // self.iters_per_sess_run

                # Initialise loop inputs
                tr_loss = tf.zeros(self.loss_shape, self.experiment.dtype, name='loss')
                inputs = [tr_loss,
                          cvar,
                          tf.zeros(batch_size, self.experiment.dtype, name='elbo'),
                          tf.constant(0., self.experiment.dtype, name='stepsize')]

                loop_out = loops_repeat(self.device_config['device'],
                                        self.iters_per_sess_run,
                                        infeed_train_op,
                                        inputs,
                                        infeed_queue,
                                        maybe_xla=possible_xla)
                return loop_out

            if self.experiment.config.training:
                # retrieve control variate (on correct device)
                cv_var, cv = self.get_control_var(i_tr)
                if self.use_infeed:
                    with self.device_config['scoper']():
                        loss, cv_update, hmc_elbo, hmc_step_size = possible_xla(tr_infeed, [cv_var])
                        cv_update = cv_var.assign(cv_update) if self.use_control_variate else tf.no_op()
                        graph_ops['train'] = [loss,
                                              tf.reduce_mean(cv_update),
                                              tf.reduce_mean(hmc_elbo),
                                              hmc_step_size]
                else:
                    loss, hmc_elbo, hmc_step_size, diags = possible_xla(train_op_batch, [X_b_tr, i_tr, cv])
                    if self.use_local_control_variate and self.device_config['on_ipu']:
                        # If using global control variate, or not using IPU, CV update is already
                        # executed within training loop. Otherwise, it is updated here
                        with self.control_var_device():
                            cv_update = self.maybe_update_control_variate(cv_var,
                                                                          i_tr,
                                                                          hmc_elbo,
                                                                          decay=self.control_var_decay)
                    else:
                        cv_update = diags['control_var_mean']

                    graph_ops['train'] = [loss,
                                          tf.reduce_mean(cv_update),
                                          tf.reduce_mean(hmc_elbo),
                                          hmc_step_size]

                # For diagnostics
                graph_ops['lr'] = self.get_current_learning_rate()
                graph_ops['epochs'] = self.get_epoch()
        return graph_ops

    def get_log_likelihood_ops(self, indices, X, y):
        """
        Returns the ops needed to estimate the log-likelihood of the model,
        via importance sampling with three Gaussian proposals, with mean and var:
            1. Of the approx posterior q(z|x), but with std = 1.2 * std(q)
            2. Mean from running 300 samples of HMC (+ 300 burn in)
            starting at the mean of approx posterior. Same std as in 1.
            3. Mean as in 2., but std also from 300 HMC samples, slightly
            overdispersed: std = std(HMC samples) * 1.2

        For more details see "Evaluation" paragraph in section 4.2 of paper
        (https://arxiv.org/pdf/1905.04062.pdf)
        """
        possible_xla = self.device_config['maybe_xla_compile']

        def hmc(X_in):
            means, _ = self.encoder(X_in)
            hmc_samples, _ = self.hmc_test.run_hmc(z_sample_init=means,
                                                   unnorm_log_prob_target=self.get_log_p_X_Z_fn(X_in, means))
            return hmc_samples

        samples_hmc = possible_xla(hmc, [X])
        hmc_ax = 1 if self.device_config['do_xla'] else 0
        means_hmc = tf.reduce_mean(samples_hmc, axis=hmc_ax)      # Mean over the 300 samples
        stds_hmc = tf.math.reduce_std(samples_hmc, axis=hmc_ax)   # std over the 300 samples

        # clip to avoid std of 0 - as done in matlab implementation
        stds_hmc = tf.clip_by_value(stds_hmc, 1e-4, tf.reduce_max(stds_hmc))

        def iwelbo_1(X):
            """
            Proposal 1:
                - mean and std are approximate posterior, std increased by factor of 1.2
            """
            def p1_sample(means, stds, shape):
                proposal_1 = tfd.MultivariateNormalDiag(means, 1.2 * stds)
                return proposal_1.sample((self.iwae_samples_te_batch_size,))

            def p1_log_prob(samples, means, stds):
                proposal_1 = tfd.MultivariateNormalDiag(means, 1.2 * stds)
                return proposal_1.log_prob(samples)

            return self.iwae_elbo(X, p1_sample, p1_log_prob)

        def iwelbo_2(X, mu_hmc):
            """
            Proposal 2:
                - mean is that of 300 samples, generated through HMC from approx posterior mean,
                    after 300 burn in steps
                - std is that of approximate posterior, with stddev increased by factor of 1.2
            """
            def p2_sample(means, stds, shape):
                proposal_2 = tfd.MultivariateNormalDiag(mu_hmc, 1.2 * stds)
                samps = proposal_2.sample(self.iwae_samples_te_batch_size)
                return tf.reshape(samps, (self.iwae_samples_te_batch_size, tf.shape(X)[0], self.Z_dim))

            def p2_log_prob(samples, means, stds):
                proposal_2 = tfd.MultivariateNormalDiag(mu_hmc, 1.2 * stds)
                return proposal_2.log_prob(samples)

            return self.iwae_elbo(X, p2_sample, p2_log_prob)

        def iwelbo_3(X, mu_hmc, sigma_hmc):
            """
            Proposal 3:
                - mean is that of 300 samples, generated through HMC from approx posterior mean,
                    after 300 burn in steps
                - std is of the same 300 samples, overdispersed by a factor of 1.2
            """
            def p3_sample(means, stds, shape):
                proposal_3 = tfd.MultivariateNormalDiag(mu_hmc, 1.2 * sigma_hmc)
                samps = proposal_3.sample(self.iwae_samples_te_batch_size)
                return tf.reshape(samps, (self.iwae_samples_te_batch_size, tf.shape(X)[0], self.Z_dim))

            def p3_log_prob(samples, means, stds):
                proposal_3 = tfd.MultivariateNormalDiag(mu_hmc, 1.2 * sigma_hmc)
                return proposal_3.log_prob(samples)

            return self.iwae_elbo(X, p3_sample, p3_log_prob)

        return [possible_xla(iwelbo_1, [X]),
                possible_xla(iwelbo_2, [X, means_hmc]),
                possible_xla(iwelbo_3, [X, means_hmc, stds_hmc])]

    def get_test_ops(self, graph_ops, i_te, X_b_te, y_b_te):
        """Add model testing operations to the graph"""

        with self.graph.as_default():

            if self.experiment.config.testing:
                with self.device_config['scoper']():
                    graph_ops['iwae_elbos_test'] = self.get_log_likelihood_ops(i_te, X_b_te, y_b_te)

            if not self.experiment.config.training:
                # To avoid KeyError when testing loaded model
                graph_ops['epochs'] = self.get_epoch()
                graph_ops['lr'] = self.get_current_learning_rate()
        return graph_ops

    def get_validation_ops(self, graph_ops, i_te, X_b_te, y_b_te):
        """Add model validation operations to the graph"""
        with self.graph.as_default():
            if self.experiment.config.validation:
                with self.device_config['scoper']():
                    graph_ops['iwae_elbos_val'] = self.get_log_likelihood_ops(i_te, X_b_te, y_b_te)

            if not self.experiment.config.training:
                # To avoid KeyError when testing loaded model
                graph_ops['epochs'] = self.get_epoch()
                graph_ops['lr'] = self.get_current_learning_rate()
        return graph_ops

    def test(self, max_iwae_batches=None):
        """Find model performance on full train and test sets"""
        # Test set LL, KL, ELBO
        n_te_batches = int(np.ceil(self.experiment.data_meta['test_size'] / self.batch_size_te))

        # Test set importance-weighted ELBO
        op_name_dict = {'iwae_elbos_test': ['te_iwae_elbo_overdisp',
                                            'te_iwae_elbo_hmc_mean_overdisp',
                                            'te_iwae_elbo_hmc_mean_std_overdisp']}

        self.experiment.log.info(f'Running test IWAE for log-likelihood estimation...')
        record_iwae_te = self.evaluation_scores(ops_sets_names=op_name_dict,
                                                iters_to_init=('test',),
                                                n_batches=n_te_batches,
                                                verbose=True)
        self.experiment.log.info('...done\n')

        # Print and save results
        self.experiment.log.info(f'Test results:\n{json.dumps(record_iwae_te, indent=4, default=serialize)}\n\n')
        self.experiment.save_record(record_iwae_te, scope='test')
        self.experiment.observer.store(f'test_results_{self.iters}_iters.json', record_iwae_te)

    def validation(self):
        """Calculate evaluation metrics of model on validation set"""
        self.experiment.log.info('Running model validation...\n')
        n_val_batches = int(np.ceil(self.experiment.data_meta['validation_size'] / self.batch_size_te))

        # Validation set importance-weighted ELBO
        op_name_dict = {'iwae_elbos_val': ['val_iwae_elbo_overdisp',
                                           'val_iwae_elbo_hmc_mean_overdisp',
                                           'val_iwae_elbo_hmc_mean_std_overdisp']}

        self.experiment.log.info(f'Running validation IWAE for log-likelihood estimation...')
        record_iwae_val = self.evaluation_scores(ops_sets_names=op_name_dict,
                                                 iters_to_init=('test',),
                                                 n_batches=n_val_batches,
                                                 verbose=True)
        self.experiment.log.info('...done\n')

        # Print and save results
        self.experiment.log.info(f'Test results:\n{json.dumps(record_iwae_val, indent=4, default=serialize)}\n\n')
        self.experiment.save_record(record_iwae_val, scope='validation')
        self.experiment.observer.store(f'test_results_{self.iters}_iters.json', record_iwae_val)

    def vcd_train_ops(self, X_b, i_b, control_var):
        """Single training update"""
        [enc_loss, dec_loss], elbo_hmc, step_size, diagnostics = self.vcd_network_loss(X_b, i_b, control_var)
        losses = {'encoder': enc_loss, 'decoder': dec_loss}
        ops = self.get_grad_ops(losses)
        with tf.control_dependencies(ops):    # Update VAE params
            return [tf.convert_to_tensor([enc_loss, dec_loss], self.experiment.dtype),
                    elbo_hmc,
                    step_size,
                    diagnostics]

    def vcd_network_loss(self, X_in, i_in, control_var):
        network_out = self.network(X_in)

        # Need tf.identity() around input - stops bug on ipu
        return self.loss(X_in, i_in, control_var, *network_out)

    def network(self, X_in):
        # Calculate q(Z|X)
        Z_cond_X_mean, Z_cond_X_std = self.encoder(X_in)

        # Reparameterisation trick: convert samples from standard normal to samples from posterior - z_0
        Z_cond_X_samples = self.reparameterised_samples(Z_cond_X_mean, Z_cond_X_std,
                                                        samples_shape=(tf.shape(X_in)[0], self.Z_dim))

        # Estimate params of p(X|Z_0)
        net_out_z_0 = self.p_X_cond_Z_params(Z_cond_X_samples)

        if self.hmc_train.n_burn_in_steps + self.hmc_train.n_post_burn_steps > 0:
            # Run some MCMC on the posterior samples - z_T
            Z_cond_X_samples_mcmc, step_size = self.hmc_train.run_hmc(z_sample_init=Z_cond_X_samples,
                                                                      unnorm_log_prob_target=self.get_log_p_X_Z_fn(X_in, Z_cond_X_samples))

            # Estimate params of p(X|Z_T)
            net_out_z_T = self.p_X_cond_Z_params(Z_cond_X_samples_mcmc)
        else:
            # Standard VAE - these objects will not be used
            Z_cond_X_samples_mcmc = Z_cond_X_samples
            net_out_z_T = net_out_z_0
            step_size = 0.

        return net_out_z_0, net_out_z_T, Z_cond_X_mean, Z_cond_X_std, Z_cond_X_samples, Z_cond_X_samples_mcmc, step_size
