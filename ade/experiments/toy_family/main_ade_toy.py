from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import random
import numpy as np
import tensorflow as tf
from ade.common.distributions import DiagMvn
from ade.common.cmd_args import cmd_args
from ade.experiments.train_utils import train_loop, build_model, build_graph, build_ema_f
from ade.experiments.toy_family import OnlineToyDataset, KernelToyDataset
from ade.common.data_utils.dataset import ToyDataset
from ade.common.plot_utils.plot_2d import plot_samples, plot_heatmap, plot_joint
from ade.common.tf_utils import get_gamma, MMD


def eval_callback(epoch, sess, save_dir, init_dist, ph_z0, x_samples, ph_x, pdf_dict, pdf_ema_dict, plt_size, true_data, eval_metric):
    z0 = init_dist.get_samples(num_samples=1000)
    samples = sess.run(x_samples, feed_dict={ph_z0: z0})
    output_path = os.path.join(save_dir, 'samples-%d.pdf' % epoch)
    plot_samples(samples, output_path)

    output_path = os.path.join(save_dir, 'joint-%d.pdf' % epoch)
    plot_joint(true_data, samples, output_path)

    for key in pdf_dict:
        pdf_func = lambda x: sess.run(pdf_dict[key], feed_dict={ph_x: x})
        output_path = os.path.join(save_dir, 'heat-%d-%s.pdf' % (epoch, str(key)))
        plot_heatmap(pdf_func, output_path, size=plt_size)

        pdf_func = lambda x: sess.run(pdf_ema_dict[key], feed_dict={ph_x: x})
        output_path = os.path.join(save_dir, 'heat-ema-%d-%s.pdf' % (epoch, str(key)))
        plot_heatmap(pdf_func, output_path, size=plt_size)

    dist = eval_metric(samples)
    print('mmd:', dist)
    return dist


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    if tf.__version__ == '1.4.1':
        tf.set_random_seed(cmd_args.seed)
    else:
        tf.random.set_random_seed(cmd_args.seed)

    if cmd_args.data_name == 'two_moons':
        db = ToyDataset(dim=2, data_file=cmd_args.data_dump)
    elif cmd_args.data_name[0].isupper():
        db = KernelToyDataset(data_name=cmd_args.data_name)
    else:
        db = OnlineToyDataset(data_name=cmd_args.data_name)

    energy_func, flow, xmin, xmax = build_model(cmd_args, db)
    plt_size = 3
    t = np.ceil(max(np.abs(xmin), np.abs(xmax)))
    plt_size = max(plt_size, t)
    print('plot size:', plt_size)
    # plot true data
    data_gen = db.data_gen(batch_size=1000, auto_reset=False)
    true_data = next(data_gen)
    output_path = os.path.join(cmd_args.save_dir, 'ground_truth.pdf')
    plot_samples(true_data, output_path)

    gamma = get_gamma(true_data, cmd_args.mmd_bd)
    eval_metric = lambda x: MMD(true_data, x, gamma)

    graph_list = build_graph(cmd_args, db, energy_func, flow)
    ema = graph_list[-1]
    ema_energy_func = build_ema_f(cmd_args, db, ema)

    # computation graph for getting samples
    init_dist = DiagMvn(mu=[0.0] * db.dim,
                        sigma=[1.0] * db.dim)
    ph_z0 = tf.placeholder(tf.float32, shape=(None, db.dim))
    log_pz0 = init_dist.get_log_pdf(ph_z0)
    x_samples, _ = flow(ph_z0, log_pz0)

    # computation graph for getting pdf

    factors = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 5]
    pdf_dict = {}
    ph_x = tf.placeholder(tf.float32, shape=(None, db.dim))
    x_energy = tf.reshape(-energy_func(ph_x), shape=[-1])
    for key in factors:
        pdf_dict[key] = tf.nn.softmax(key * x_energy)
    
    x_energy_ema = tf.reshape(-ema_energy_func(ph_x), shape=[-1])
    pdf_ema_dict = {}
    for key in factors:
        pdf_ema_dict[key] = tf.nn.softmax(key * x_energy_ema)

    train_loop(cmd_args, db, energy_func, flow,
               eval_callback=lambda e, s: eval_callback(e, s, cmd_args.save_dir, init_dist, ph_z0, x_samples, ph_x, pdf_dict, pdf_ema_dict, plt_size, true_data, eval_metric),
               graph_elements=graph_list[:-1])
