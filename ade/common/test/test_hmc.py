from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ade.common.cmd_args import cmd_args
from ade.common.f_family import GaussianEnergy
from ade.common.distributions import DiagMvn
import ade.common.mcmc_net as mcmc_net


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    tf.random.set_random_seed(cmd_args.seed)

    true_mean = -1.0
    true_std = 0.1

    with tf.variable_scope('energy_func'):
        energy_func = GaussianEnergy(dim=cmd_args.gauss_dim, init_mu=true_mean, init_sigma=true_std)

    with tf.variable_scope('generator'):
        mcmc = mcmc_net.HMCNet(cmd_args, dim=cmd_args.gauss_dim)
    
    ph_z0 = tf.placeholder(tf.float32, shape=(1000, cmd_args.gauss_dim))
    zt = mcmc(energy_func, ph_z0)
    init_dist = DiagMvn(mu=[0.0] * cmd_args.gauss_dim,
                        sigma=[1.0] * cmd_args.gauss_dim)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        z0 = init_dist.get_samples(num_samples=1000)
        z_samples = sess.run(zt, feed_dict={ph_z0: z0})

        print(np.mean(z_samples, axis=0))
        print(np.std(z_samples, axis=0))