from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import sonnet as snt

from ade.common.tf_utils import MLP, get_gamma


class GaussianEnergy(snt.AbstractModule):
    def __init__(self, dim, init_mu=None, init_sigma=None, name='gauss_energy'):
        super(GaussianEnergy, self).__init__(name=name)
        self.name = name
        self.dim = dim
        with self._enter_variable_scope():
            self.mu = tf.get_variable('u', shape=(dim,), 
                                      initializer=tf.initializers.random_normal() if init_mu is None else tf.constant_initializer(init_mu))
            self.sigma = tf.get_variable('sigma', shape=(dim,), 
                                         initializer=tf.constant_initializer(1.0 if init_sigma is None else init_sigma))

    def _build(self, x):
        variance = self.sigma * self.sigma
        if tf.__version__ == '1.4.1':
            f = tf.reduce_sum(((x - self.mu) ** 2) / 2.0 / variance, axis=1, keep_dims=True)
        else:
            f = tf.reduce_sum(((x - self.mu) ** 2) / 2.0 / variance, axis=1, keepdims=True)
        return f

    def get_params(self, sess):
        mu, sigma = sess.run([self.mu, self.sigma])
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        return np.concatenate((mu, sigma))

    def get_true_params(self, mu=-1, sigma=0.1):
        mu = [mu] * self.dim
        sigma = [sigma] * self.dim
        p = np.array(mu + sigma, dtype=np.float32)
        return p


class MLPEnergy(snt.AbstractModule):
    def __init__(self, dim, hidden_dim, depth, output_dim=1, act_hidden=tf.nn.relu, act_out=None, sp_iters=0, mlp=None, name='mlp_energy'):
        super(MLPEnergy, self).__init__(name=name)
        self.act_out = act_out
        with self._enter_variable_scope():
            if mlp is None:
                self.mlp = MLP(dim, hidden_dim, depth, output_dim, act_hidden, sp_iters)
            else:
                self.mlp = mlp

    def _build(self, x):
        score = self.mlp(x)
        if self.act_out is not None:
            score = self.act_out(score)
        return score
