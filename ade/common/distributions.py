from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import numpy as np
import tensorflow as tf


class DiagMvn(object):

    def __init__(self, mu, sigma):
        assert isinstance(mu, list) and isinstance(sigma, list)
        self.tf_mu = tf.constant(mu, shape=[1, len(mu)], dtype=tf.float32)
        self.tf_sigma = tf.constant(sigma, shape=[1, len(sigma)], dtype=tf.float32)

        mu = np.array(mu, dtype=np.float32).reshape(1, -1)        
        sigma = np.array(sigma, dtype=np.float32).reshape(1, -1)
        assert mu.shape[1] == sigma.shape[1]
        self.mu = mu
        self.sigma = sigma
        self.gauss_dim = mu.shape[1]
        
    def get_log_pdf(self, x):
        return DiagMvn.log_pdf(x, self.tf_mu, self.tf_sigma)

    def get_samples(self, num_samples):
        return np.random.randn(num_samples, self.gauss_dim).astype(np.float32) * self.sigma + self.mu

    @staticmethod
    def log_pdf(x, mu=None, sigma=None):
        if mu is None:
            mu = 0.0
        t = -(x - mu) ** 2 / 2.0
        if sigma is not None:
            t = t / ((sigma + 1e-6) ** 2)
            t = t - 0.5 * tf.log(2 * np.pi * sigma * sigma + 1e-6)
        else:
            t = t - 0.5 * np.log(2 * np.pi)
        if tf.__version__ == '1.4.1':
            return tf.reduce_sum(t, axis=-1, keep_dims=True)
        else:
            return tf.reduce_sum(t, axis=-1, keepdims=True)
