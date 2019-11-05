from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import random
import numpy as np
import tensorflow as tf
from ade.common import fp_flow
from tqdm import tqdm

from ade.common.distributions import DiagMvn
import ade.common.mcmc_net as mcmc_net
from ade.common.f_family import GaussianEnergy, MLPEnergy
from ade.common.flow_family import NormFlow, IdentityFlow, MLPGaussian
from ade.common.tf_utils import tf_optimizer, MLP


def get_gen_loss(args, x, z0, flow, energy_func, init_dist):
    if args.flow_type == 'identity':
        return None, None
    opt_gen = tf_optimizer(learning_rate=args.learning_rate, 
        beta1=args.beta1, beta2=args.beta2)

    log_pz0 = init_dist.get_log_pdf(z0)
    xfake, ll = flow(z0, log_pz0)
    f_sampled_x = -energy_func(xfake)
    loss = -tf.reduce_mean(f_sampled_x) + args.ent_lam * tf.reduce_mean(ll)

    gvs = opt_gen.compute_gradients(loss, var_list=tf.trainable_variables(scope='generator'))
    gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val) for grad,val in gvs if grad is not None]
    train_gen = opt_gen.apply_gradients(gvs)
    return loss, train_gen


def get_disc_loss(args, x, z0, flow, energy_func, init_dist):
    opt_disc = tf_optimizer(learning_rate=args.learning_rate, 
        beta1=args.beta1, beta2=args.beta2)

    # log_pz0 = init_dist.get_log_pdf(z0)
    x_fake, ll = flow(z0, 0)

    fx = -energy_func(x)
    f_fake_x = -energy_func(tf.stop_gradient(x_fake))
    f_loss = tf.reduce_mean(-fx + f_fake_x)
    if args.moment_penalty > 0:
        f_loss += tf.reduce_mean(ll)

    loss = f_loss
    if args.gp_lambda > 0:  # add gradient penalty
        if tf.__version__ == '1.4.1':
            alpha = tf.random_uniform(shape=(tf.shape(x)[0], 1))
        else:
            alpha = tf.random.uniform(shape=(tf.shape(x)[0], 1))
        x_hat = alpha * x + (1 - alpha) * x_fake
        d_hat = energy_func(x_hat)
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0)) * args.gp_lambda
        loss = loss + ddx
    gvs = opt_disc.compute_gradients(loss, var_list=tf.trainable_variables(scope='energy_func'))
    gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val) for grad,val in gvs if grad is not None]
    train_disc = opt_disc.apply_gradients(gvs)
    return f_loss, train_disc


def build_ema_f(cmd_args, db, ema):
    data_dim = db.dim
    vars = tf.trainable_variables(scope='energy_func')
    with tf.variable_scope('energy_func'):
        ema_vars = [ema.average(v) for v in vars]
        mlp = MLP(input_dim=data_dim,
                  hidden_dim=cmd_args.nn_hidden_size,
                  depth=cmd_args.f_depth,
                  output_dim=1,
                  act_hidden=tf.nn.relu,
                  vars=ema_vars,
                  sp_iters=cmd_args.sp_iters)
        if cmd_args.energy_type == 'mlp':
            energy_func = MLPEnergy(dim=data_dim,
                                    hidden_dim=cmd_args.nn_hidden_size,
                                    depth=cmd_args.f_depth,
                                    sp_iters=cmd_args.sp_iters,
                                    mlp=mlp)
        else:
            raise NotImplementedError
    return energy_func


def build_model(cmd_args, db):
    data_dim = db.dim
    with tf.variable_scope('energy_func'):
        if cmd_args.energy_type == 'gauss':
            energy_func = GaussianEnergy(dim=data_dim)
        elif cmd_args.energy_type == 'mlp':
            energy_func = MLPEnergy(dim=data_dim,
                                    hidden_dim=cmd_args.nn_hidden_size,
                                    depth=cmd_args.f_depth,
                                    sp_iters=cmd_args.sp_iters)
        else:
            raise NotImplementedError

    with tf.variable_scope('generator'):
        if cmd_args.flow_type == 'norm':
            base_flow = NormFlow(dim=data_dim, num_layers=cmd_args.gen_depth)
        elif cmd_args.flow_type == 'identity':
            base_flow = IdentityFlow()
        elif cmd_args.flow_type == 'mlp':
            base_flow = MLPGaussian(dim=data_dim, hidden_dim=cmd_args.nn_hidden_size, depth=cmd_args.gen_depth, 
                                    output_dim=data_dim)
        else:
            raise NotImplementedError
        x_min = None
        x_max = None
        data_gen = db.data_gen(batch_size=cmd_args.batch_size, auto_reset=False)

        for i in range(10):
            x_input = next(data_gen)
            cur_min = np.min(x_input)
            if x_min is None or cur_min < x_min:
                x_min = cur_min
            cur_max = np.max(x_input)
            if x_max is None or cur_max > x_max:
                x_max = cur_max
        net_type = getattr(mcmc_net, cmd_args.mcmc_type + 'Net')
        if cmd_args.clip_samples:
            mcmc = net_type(cmd_args, dim=data_dim, x_min=x_min, x_max=x_max)
        else:
            mcmc = net_type(cmd_args, dim=data_dim, x_min=None, x_max=None)

    fn_gen = lambda z, log_z: fp_flow(energy_func, base_flow, mcmc, z, log_z)
    return energy_func, fn_gen, x_min, x_max


def build_graph(cmd_args, db, energy_func, flow):
    ph_real_data = tf.placeholder(tf.float32, shape=(None, db.dim))
    ph_z0 = tf.placeholder(tf.float32, shape=(None, db.dim))

    init_dist = DiagMvn(mu=[0.0] * db.dim,
                        sigma=[1.0] * db.dim)

    gen_loss, train_gen = get_gen_loss(cmd_args, ph_real_data, ph_z0, flow, energy_func, init_dist)
    disc_loss, train_disc = get_disc_loss(cmd_args, ph_real_data, ph_z0, flow, energy_func, init_dist)
    ema = tf.train.ExponentialMovingAverage(decay=cmd_args.ema_decay)
    with tf.control_dependencies([train_disc]):
        vars = tf.trainable_variables(scope='energy_func')
        disc_train_op = ema.apply(vars)

    train_disc = disc_train_op
    return ph_real_data, ph_z0, init_dist, gen_loss, train_gen, disc_loss, train_disc, ema


def train_loop(cmd_args, db, energy_func, flow, eval_callback=lambda x: None, graph_elements=None):
    if graph_elements is None:
        ph_real_data, ph_z0, init_dist, gen_loss, train_gen, disc_loss, train_disc, _ = build_graph(cmd_args, db, energy_func, flow)
    else:
        ph_real_data, ph_z0, init_dist, gen_loss, train_gen, disc_loss, train_disc = graph_elements

    config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1
             )
    config.gpu_options.allow_growth = True
    best_dist = None
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())        
        data_gen = db.data_gen(batch_size=cmd_args.batch_size, auto_reset=True)
        for epoch in range(cmd_args.num_epochs):
            pbar = tqdm(range(cmd_args.iters_per_eval), unit='batch')

            for pos in pbar:
                x_input = next(data_gen)
                # optimize discriminator
                z0 = init_dist.get_samples(num_samples=x_input.shape[0])
                _, np_disc_loss = sess.run([train_disc, disc_loss], feed_dict={ph_real_data: x_input, ph_z0: z0})

                # optimize generator
                if cmd_args.flow_type != 'identity':
                    for i in range(cmd_args.g_iter):
                        z0 = init_dist.get_samples(num_samples=x_input.shape[0])
                        _, np_gen_loss = sess.run([train_gen, gen_loss], feed_dict={ph_real_data: x_input, ph_z0: z0})
                else:
                    np_gen_loss = 0.0
                pbar.set_description('disc_loss: %.4f, gen_loss: %.4f' % (np_disc_loss, np_gen_loss))
            cur_dist = eval_callback(epoch, sess)
            if best_dist is None or cur_dist < best_dist:
                best_dist = cur_dist
                print('saving with best model')
                with open('%s/best-model.txt' % cmd_args.save_dir, 'w') as f:
                    f.write('dist %.10f\n' % best_dist)
                    f.write('epoch %d\n' % epoch)
                saver.save(sess, '%s/best-model.ckpt' % cmd_args.save_dir)
