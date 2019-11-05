from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import tensorflow as tf
import sonnet as snt
import numpy as np

from ade.common.tf_utils import MLP


class MLPGaussian(snt.AbstractModule):
    def __init__(self, dim, hidden_dim, depth, output_dim, act_hidden=tf.nn.relu, sp_iters=0, mlp=None, name='mlp_gauss'):
        super(MLPGaussian, self).__init__(name=name)
        with self._enter_variable_scope():
            if mlp is None:
                self.mlp = MLP(dim, hidden_dim, depth - 1, hidden_dim, act_hidden, sp_iters)
            else:
                self.mlp = mlp
            self.w_mu = tf.get_variable("w_mu", shape=[hidden_dim, output_dim])
            self.b_mu = tf.get_variable("b_mu", shape=[1, output_dim])

            self.w_logsig = tf.get_variable("w_logsig", shape=[hidden_dim, output_dim])
            self.b_logsig = tf.get_variable("b_logsig", shape=[1, output_dim])

    def _build(self, x, logp):
        hidden = self.mlp(x)
        mu = tf.matmul(hidden, self.w_mu) + self.b_mu
        log_sigma = tf.matmul(hidden, self.w_logsig) + self.b_logsig

        eps = tf.random.normal(shape=tf.shape(mu), mean=0, stddev=1, dtype=tf.float32)
        sigma = tf.exp(log_sigma)
        z = mu + sigma * eps
        
        t = -eps ** 2 / 2.0 - 0.5 * tf.log(2 * np.pi) - log_sigma
        if tf.__version__ == '1.4.1':
            ll = tf.reduce_sum(t, axis=-1, keep_dims=True)
        else:
            ll = tf.reduce_sum(t, axis=-1, keepdims=True)

        return z, logp + ll


class PlanarFlow(snt.AbstractModule):
    """PlanarFlow"""

    def __init__(self, dim, name='planar_flow_layer'):
        super(PlanarFlow, self).__init__(name=name)
        self.name = name
        self.dim = dim
        self.h = tf.tanh
        with self._enter_variable_scope():
            self.u = tf.get_variable('u', shape=(dim,))
            self.w = tf.get_variable('w', shape=(dim,))
            self.b = tf.get_variable('b', shape=(1,))
            self.u = tf.reshape(self.u, (dim, 1))
            self.w = tf.reshape(self.w, (dim, 1))

    def _build(self, z, logp):
        a = self.h(tf.matmul(z, self.w) + self.b)
        psi = tf.matmul(1 - a ** 2, tf.transpose(self.w))

        x = tf.matmul(tf.transpose(self.w), self.u)
        m = -1 + tf.nn.softplus(x)
        u_h = self.u + (m - x) * self.w / (tf.matmul(tf.transpose(self.w), self.w))

        logp = logp - tf.squeeze(tf.log(1 + tf.matmul(psi, u_h)))
        z = z + tf.matmul(a, tf.transpose(u_h))

        return z, logp


class NormFlow(snt.AbstractModule):
    """Normalizing flow"""

    def __init__(self, dim, num_layers, name="norm_flow"):
        super(NormFlow, self).__init__(name=name)
        self.name = name
        self.dim = dim
        self.num_layers = num_layers

        self.planar_flows = []
        with self._enter_variable_scope():
            for i in range(self.num_layers):
                flow = PlanarFlow(dim, name='planar_flow_layer_%d' % i)
                self.planar_flows.append(flow)

    def _build(self, z, logp):
        for flow in self.planar_flows:
            z, logp = flow(z, logp)
        return z, logp


class IdentityFlow(snt.AbstractModule):
    """IdentityFlow"""

    def __init__(self, name='IdentityFlow'):
        super(IdentityFlow, self).__init__(name=name)
        self.name = name

    def _build(self, z, logp):
        return z, logp