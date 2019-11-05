from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import random
import numpy as np
import tensorflow as tf

from ade.common.cmd_args import cmd_args
from ade.experiments.train_utils import train_loop, build_model
from ade.experiments.mvn_unimodal import MvnGaussDataset

true_mean = -1.0
true_std = 0.1


def eval_callback(epoch, sess, energy_func):
    mu, sigma = sess.run([energy_func.mu, energy_func.sigma])
    print('epoch %d, mean:' % epoch, mu)
    print('epoch %d, sigma:' % epoch, sigma)
    return 0.0


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    if tf.__version__ == '1.4.1':
        tf.set_random_seed(cmd_args.seed)
    else:
        tf.random.set_random_seed(cmd_args.seed)


    db = MvnGaussDataset(mu=true_mean, sigma=true_std, dim=cmd_args.gauss_dim)

    energy_func, flow, _, _ = build_model(cmd_args, db)

    train_loop(cmd_args, db, energy_func, flow, 
               eval_callback=lambda e, s: eval_callback(e, s, energy_func))
