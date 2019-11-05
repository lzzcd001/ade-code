from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import tensorflow as tf
import sonnet as snt
import numpy as np


class NoneNet(snt.AbstractModule):
    def __init__(self, args, dim, x_min=None, x_max=None, name='none_net'):
        super(NoneNet, self).__init__(name=name)

    def _build(self, U, q):
        return q


class HMCNet(snt.AbstractModule):
    def __init__(self, args, dim, x_min=None, x_max=None, name='hmc_net'):
        super(HMCNet, self).__init__(name=name)
        self.dim = dim
        self.use_mh = args.use_mh
        self.hmc_clip = args.hmc_clip
        self.hmc_steps = args.mcmc_steps
        self.hmc_adaptive_mode = args.hmc_adaptive_mode
        self.hmc_step_size = args.hmc_step_size
        self.hmc_p_sigma = args.hmc_p_sigma
        self.hmc_inner_steps = args.hmc_inner_steps
        self.use_2nd_order_grad = args.use_2nd_order_grad
        self.x_min = x_min
        self.x_max = x_max
        self.moment_penalty = args.moment_penalty
        if x_min is not None:
            print('clip value in [%.4f, %.4f]' % (x_min, x_max))
        self.eps_list = []
        with self._enter_variable_scope():
            for i in range(self.hmc_steps):
                self.eps_list.append(tf.get_variable('hmc_eps%d' % i, 
                                                     shape=(1, dim), 
                                                     initializer=tf.constant_initializer(args.hmc_step_size)))

    def _clip(self, vec):
        if self.hmc_clip <= 0:
            return vec
        return tf.clip_by_norm(vec, self.hmc_clip, axes=[1])

    def _euler_p(self, u, q, p, eps):
        dq = tf.reshape(tf.gradients(u, q), [-1, self.dim])
        if not self.use_2nd_order_grad:
            dq = tf.stop_gradient(dq)
        p_new = p - eps * self._clip(dq)
        return p_new

    def _euler_q(self, q, p, eps, M):
        p = tf.reshape(p, tf.shape(q))
        q_new = q + eps * self._clip(p) / M
        return q_new

    def _build(self, U, q, init_p=None, log_q_p=None):
        '''
        Args:
            U: potential function
            q: initial z
        '''
        def V(p, M):
            if tf.__version__ == '1.4.1':
                v = 0.5 * tf.reduce_sum(p ** 2, axis=1, keep_dims=True) / M
            else:
                v = 0.5 * tf.reduce_sum(p ** 2, axis=1, keepdims=True) / M
            return v
        
        q_current = q
        L = self.hmc_inner_steps
        p = init_p
        if self.hmc_adaptive_mode != 'auto':
            eps = self.hmc_step_size
        M = self.hmc_p_sigma ** 2
        p_norms = 0
        for step in range(self.hmc_steps):
            if self.hmc_adaptive_mode == 'auto':
                eps = self.eps_list[step]
            if init_p is None:
                p = tf.random_normal(shape=tf.shape(q))
            q = q_current
            u = U(q)
            H_current = u + V(p, M)

            p = self._euler_p(u, q, p,  eps / 2.0)
            for i in range(L):
                q = self._euler_q(q, p, eps, M)
                
                if i + 1 < L:
                    p = self._euler_p(U(q), q, p,  eps)
            u = U(q)
            p = self._euler_p(u, q, p, eps / 2.0)
            cur_pnorm = 0.5 * tf.reduce_sum(p**2, axis=-1, keepdims=True)
            if self.x_min is not None:
                q = tf.clip_by_value(q, self.x_min, self.x_max)
            H_new = u + V(p, M)

            # rejection sampling
            if self.use_mh:
                ratio = tf.stop_gradient(tf.squeeze(tf.exp(H_current - H_new)))
                if tf.__version__ == '1.4.1':
                    is_accept = tf.random_uniform(shape=tf.shape(ratio)) < ratio
                else:
                    is_accept = tf.random.uniform(shape=tf.shape(ratio)) < ratio
                
                q_current = tf.where(is_accept, q, q_current)
                cur_pnorm = cur_pnorm * tf.reshape(tf.cast(is_accept, tf.float32), shape=tf.shape(cur_pnorm))
                accept_rate = tf.reduce_mean(tf.to_float(is_accept))

                if self.hmc_adaptive_mode == 'human':
                    eps = tf.cond(accept_rate < 0.6, lambda: eps / 1.1, lambda: eps)
                    eps = tf.cond(accept_rate > 0.7, lambda: eps * 1.1, lambda: eps)
            else:
                q_current = q
            p_norms = p_norms + cur_pnorm
        if log_q_p is None:
            return q_current
        else:
            if self.moment_penalty > 0:
                log_q_p = log_q_p + self.moment_penalty * p_norms
            return q_current, log_q_p


class GeneralHmcNet(HMCNet):
    def __init__(self, args, dim, x_min=None, x_max=None, name='general_hmc_net'):
        super(GeneralHmcNet, self).__init__(args, dim, x_min, x_max, name=name)

        with self._enter_variable_scope():
            dims = [args.nn_hidden_size, dim]
            self.gv = snt.nets.MLP(output_sizes=dims,
                                    activation=tf.nn.relu)
            self.gx = snt.nets.MLP(output_sizes=dims,
                                    activation=tf.nn.relu)

    def _euler_p(self, u, q, p, eps):
        dq = tf.reshape(tf.gradients(u, q), [-1, self.dim])
        if not self.use_2nd_order_grad:
            dq = tf.stop_gradient(dq)
        t = self.gv(tf.concat([dq, q], axis=1))
        p_new = p - eps * self._clip(t)
        return p_new

    def _euler_q(self, q, p, eps, M):
        p = tf.reshape(p, tf.shape(q))
        q_new = q + eps * self._clip(self.gx(p)) / M
        return q_new


class ResGeneralHmcNet(HMCNet):
    def __init__(self, args, dim, name='res_general_hmc_net'):
        super(ResGeneralHmcNet, self).__init__(args, dim, name=name)

        with self._enter_variable_scope():
            dims = [args.nn_hidden_size, dim]
            self.gv = snt.nets.MLP(output_sizes=dims,
                                    activation=tf.nn.leaky_relu)
            self.gx = snt.nets.MLP(output_sizes=dims,
                                    activation=tf.nn.leaky_relu)

    def _euler_p(self, u, q, p, eps):
        dq = tf.reshape(tf.gradients(u, q), [-1, self.dim])
        if not self.use_2nd_order_grad:
            dq = tf.stop_gradient(dq)
        t = self.gv(tf.concat([dq, q], axis=1))
        p_new = p - eps * (1 + tf.clip_by_norm(t, 1))
        return p_new

    def _euler_q(self, q, p, eps, M):
        p = tf.reshape(p, tf.shape(q))
        q_new = q + eps * (1 + tf.clip_by_norm(self.gx(p), 1)) / M
        return q_new


class SGLDNet(snt.AbstractModule):
    def __init__(self, args, dim, name='sgld_net'):
        super(SGLDNet, self).__init__(name=name)
        self.dim = dim
        self.use_mh = args.use_mh
        self.hmc_steps = args.mcmc_steps
        self.hmc_adaptive_mode = args.hmc_adaptive_mode
        self.hmc_step_size = args.hmc_step_size
        # self.hmc_step_size = args.hmc_step_size
        self.hmc_p_sigma = args.hmc_p_sigma
        self.hmc_inner_steps = args.hmc_inner_steps
        self.use_2nd_order_grad = args.use_2nd_order_grad
        self.sgld_clip_value = args.sgld_clip_value
        self.sgld_clip_mode = args.sgld_clip_mode
        self.sgld_noise_std = args.sgld_noise_std

        self.eps_list = []
        with self._enter_variable_scope():
            for i in range(self.hmc_steps):
                self.eps_list.append(tf.get_variable('hmc_eps%d' % i, 
                                                     shape=(1, dim), 
                                                     initializer=tf.constant_initializer(args.hmc_step_size)))

    def _euler_q(self, u, q, eps):
        dq = tf.reshape(tf.gradients(u, q), [-1, self.dim])
        if not self.use_2nd_order_grad:
            dq = tf.stop_gradient(dq)
        if self.sgld_clip_value is not None:
            self.sgld_clip_value = np.abs(self.sgld_clip_value)
            if self.sgld_clip_mode == 'value':
                dq = tf.clip_by_value(dq, -self.sgld_clip_value, self.sgld_clip_value)
            elif self.sgld_clip_mode == 'norm':
                dp = tf.clip_by_norm(dq, self.sgld_clip_value)
            else:
                raise NotImplementedError
        q_new = q - eps * dq
        return q_new

    def _build(self, U, q):
        '''
        Args:
            U: potential function
            q: initial z
        '''
        
        q_current = q
        L = self.hmc_inner_steps

        for step in range(self.hmc_steps):
            if self.hmc_adaptive_mode == 'auto':
                eps = self.eps_list[step]
            else:
                eps = self.hmc_step_size
            if tf.__version__ == '1.4.1':
                p = tf.random_normal(shape=tf.shape(q)) * self.sgld_noise_std
            else:
                p = tf.random.normal(shape=tf.shape(q)) * self.sgld_noise_std
            u = U(q_current)
            q_current = self._euler_q(u, q, eps) + p

        return q_current
