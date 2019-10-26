from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import tensorflow as tf
import numpy as np

class HMCSampler():
    def __init__(self, dim, name='hmc_net'):
        if isinstance(dim, int):
            dim = (dim,)
        self.dim = dim
        self.shape = (-1, ) + dim
        self.outer_hmc_steps = 1
        self.hmc_adaptive_mode = 'auto'
        self.hmc_p_sigma = 1.0 
        self.hmc_inner_steps = 30
        self.use_2nd_order_grad = False

        self.eps_list = []
        self.md_list = []
        with tf.variable_scope('hmc', reuse=tf.AUTO_REUSE):
            for i in range(self.outer_hmc_steps):
                self.eps_list.append(tf.get_variable('hmc_eps_%d' % i, 
                                                        shape=(1,) + dim, 
                                                        initializer=tf.constant_initializer(1e1)))

            self.md_list.append(tf.constant(0.5))

        
        self.placeholder = tf.get_variable('hmc_eps', shape=(1,), 
            initializer=tf.constant_initializer(0.0),
            trainable=False) ### Not used

    def _euler_p(self, u, q, p, eps, md):
        dq = tf.reshape(tf.gradients(u, q), self.shape)
        if not self.use_2nd_order_grad:
            dq = tf.stop_gradient(dq)
        p_new = md * p + eps * tf.clip_by_value(dq, -0.01, 0.01)
        return p_new

    def _euler_q(self, q, p, eps, M, mask):
        p = tf.reshape(p, tf.shape(q))
        q_new = q + p / M * mask
        return q_new

    def sample(self, U, q, reuse_flag=True, mask=None):
        '''
        Args:
            U: potential function
            q: initial z
        '''
        if mask is None:
            mask = tf.constant(1.0)

        def V(p, M):
            p = tf.reshape(p, (-1, np.prod(self.dim)))
            if tf.__version__ == '1.4.1':
                v = 0.5 * tf.reduce_sum(p ** 2, reduction_indices=[1], keep_dims=True) / M
            else:
                v = 0.5 * tf.reduce_sum(p ** 2, reduction_indices=[], keepdims=True) / M
            return v

        L = self.hmc_inner_steps

        eps_list = []
        M = self.hmc_p_sigma ** 2
        for step in range(self.outer_hmc_steps):
            print("hmc step %d" % step)
            eps = self.eps_list[step]

            if tf.__version__ == '1.4.1':
                p = tf.random_normal(shape=tf.shape(q)) 
            else:
                p = tf.random.normal(shape=tf.shape(q))

            std_p = 1e-4
            p *= std_p
 
            u = U(q, reuse=reuse_flag)
            reuse_flag = True

            p = self._euler_p(u, q, p,  eps / 2.0, self.md_list[step]) 
            for i in range(L):
                q = self._euler_q(q, p, eps, M, mask)
                q = tf.clip_by_value(q, -1.0, 1.0)
       
                if i + 1 < L:
                    p = self._euler_p(U(q, reuse=reuse_flag), q, p,  eps, self.md_list[step])
            u = U(q, reuse=reuse_flag)
            p = self._euler_p(u, q, p, eps / 2.0, self.md_list[step])

        log_prob = - self.hmc_inner_steps * tf.reduce_sum(tf.log(self.md_list[step] + 1e-6))
        return q, p, log_prob

    def __call__(self, U, q):
        return self._build(U, q)
