import timeit

import os
import numpy as np
import tensorflow as tf

from libs.input_helper import Cifar10
from libs.utils import save_images, mkdir
from net import Generator, Discrminator

from utils import get_is_score
from hmc import HMCSampler

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_iter', 200000, '')
flags.DEFINE_integer('snapshot_interval', 1000, 'interval of snapshot')
flags.DEFINE_integer('evaluation_interval', 5000, 'interval of evalution')
flags.DEFINE_integer('display_interval', 100, 'interval of displaying log to console')
flags.DEFINE_float('adam_alpha', 1e-4, 'learning rate')
flags.DEFINE_float('adam_beta1', 0.0, 'beta1 in Adam')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam')
flags.DEFINE_integer('n_dis', 1, 'n discrminator train')

mkdir('results')
mkdir('results/tmp')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
INCEPTION_FILENAME = 'inception_score.pkl'
config = FLAGS.__flags

config = {k: FLAGS[k]._value for k in FLAGS}
generator = Generator(**config)
discriminator = Discrminator(**config)
data_set = Cifar10(batch_size=FLAGS.batch_size)

global_step = tf.Variable(0, name="global_step", trainable=False)
increase_global_step = global_step.assign(global_step + 1)
is_training = tf.placeholder(tf.bool, shape=())
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_prehat, logprob = generator(z, is_training=is_training)

hmc = HMCSampler((32,32,3))
x_hat, hmc_p, delta_lp = hmc.sample(lambda x, reuse: discriminator(x, update_collection="NO_OPS"), x_prehat, reuse_flag=False)
logprob += delta_lp

logprob_loss = tf.reduce_mean(logprob)
hmc_p_loss = tf.reduce_mean(tf.reduce_sum(hmc_p**2, reduction_indices=[1,2,3]))

x = tf.placeholder(tf.float32, shape=x_hat.shape)

d_fake = discriminator(x_hat, update_collection=None)
d_prefake = discriminator(x_prehat, update_collection="NO_OPS")

# Don't need to collect on the second call, put NO_OPS
d_real = discriminator(x, update_collection="NO_OPS")


# Softplus at the end as in the official code of author at chainer-gan-lib github repository
d_loss = tf.reduce_mean(d_fake - d_real)
g_loss = tf.reduce_mean(-d_fake)
logprob_loss = tf.reduce_mean(logprob)
d_loss_summary_op = tf.summary.scalar('d_loss', d_loss)
g_loss_summary_op = tf.summary.scalar('g_loss', g_loss)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('results/snapshots')

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
hmc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hmc')

optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
hmc_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2, epsilon=1e-5)

d_gvs = optimizer.compute_gradients(d_loss, var_list=d_vars)
g_gvs = optimizer.compute_gradients(g_loss + 1e-5 * hmc_p_loss + 1e-5 * logprob_loss, var_list=g_vars)
hmc_gvs = hmc_optimizer.compute_gradients(g_loss + 1e-5 * hmc_p_loss + 1e-5 * logprob_loss, var_list=hmc_vars)

d_solver = optimizer.apply_gradients(d_gvs)
g_solver = optimizer.apply_gradients(g_gvs)
hmc_solver = hmc_optimizer.apply_gradients(hmc_gvs)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
best_saver = tf.train.Saver()


np.random.seed(1337)
sample_noise = generator.generate_noise()
np.random.seed()
iteration = sess.run(global_step)


is_img = tf.placeholder(tf.float32, shape=[None, None, None, 3])
is_feat = tf.nn.softmax(tf.contrib.gan.eval.run_inception(tf.image.resize_bilinear(is_img, [299, 299])))

if tf.train.latest_checkpoint('results/snapshots') is not None:
  saver.restore(sess, tf.train.latest_checkpoint('results/snapshots'))

iteration = sess.run(global_step)
start = timeit.default_timer()
best_inception = 0.0

is_start_iteration = False
inception_scores = []

while iteration < FLAGS.max_iter:
  _, __, g_loss_curr, logprob_v = sess.run(
    [g_solver, hmc_solver, g_loss, logprob_loss], 
    feed_dict={z: generator.generate_noise(), is_training: True})
  for _ in range(FLAGS.n_dis):
    _, d_loss_curr, summaries = sess.run([d_solver, d_loss, merged_summary_op],
                          feed_dict={x: data_set.get_next_batch(), z: generator.generate_noise(), is_training: True})

  sess.run(increase_global_step)
  if (iteration + 1) % FLAGS.display_interval == 0 and not is_start_iteration:
    summary_writer.add_summary(summaries, global_step=iteration)
    stop = timeit.default_timer()
    logstr = ('Iter {}: d_loss = {:4f}, g_loss = {:4f}, logprob = {:4f}, time = {:2f}s'.format(
      iteration, d_loss_curr, g_loss_curr, logprob_v, stop - start))
    print(logstr)
    with open('results/txtlog.txt', 'a') as f:
      f.write(logstr + '\n')
    start = stop

  if (iteration + 1) % FLAGS.snapshot_interval == 0 and not is_start_iteration:
    saver.save(sess, 'results/snapshots/model.ckpt', global_step=iteration)
    sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'results/tmp/{:06d}.png'.format(iteration))

  if (iteration + 1) % FLAGS.evaluation_interval == 0:
    sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'results/tmp/{:06d}.png'.format(iteration))

    # Sample images for evaluation
    print("Evaluating...")
    num_images_to_eval = 10000
    eval_images = []
    num_batches = num_images_to_eval // FLAGS.batch_size + 1
    print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
    np.random.seed(0)
    for _ in range(num_batches):
      images = sess.run(x_hat, feed_dict={z: generator.generate_noise(), is_training: False})
      eval_images.append(images)
    np.random.seed()
    eval_images = np.vstack(eval_images)
    eval_images = eval_images[:num_images_to_eval]

    eval_images = np.clip(eval_images, -1.0, 1.0)
    inception_score_mean, inception_score_std = get_is_score(is_feat, is_img, eval_images)
    print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))


    if inception_score_mean > best_inception:
      best_inception = inception_score_mean
      best_saver.save(sess, 'results/snapshots/best_model.ckpt')
    with open('results/is.txt', 'a') as f:
      f.write('iter: %d, mean: %1.6f, std: %1.6f\n' % (iteration, inception_score_mean, inception_score_std))

  iteration += 1
  is_start_iteration = False
