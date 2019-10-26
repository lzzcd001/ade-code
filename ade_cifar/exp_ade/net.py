import numpy as np

from libs.ops import *

class Generator(object):

  def __init__(self, hidden_dim=128, batch_size=64, hidden_activation=tf.nn.relu, 
      output_activation=tf.identity, z_distribution='normal', scope='generator', **kwargs):
    self.hidden_dim = hidden_dim
    self.batch_size = batch_size
    self.hidden_activation = hidden_activation
    self.output_activation = output_activation
    self.z_distribution = z_distribution
    self.scope = scope

  def __call__(self, z, is_training=True, **kwargs):
    with tf.variable_scope(self.scope):
      l0  = self.hidden_activation(batch_norm(
        linear(z, 4 * 4 * 512, name='l0', stddev=0.02),
        name='bn0', is_training=is_training))
      l0  = tf.reshape(l0, [self.batch_size, 4, 4, 512])
      dc1 = self.hidden_activation(batch_norm(
        deconv2d( l0, [self.batch_size,  8,  8, 256], name='dc1', stddev=0.02),
        name='bn1', is_training=is_training))
      dc2 = self.hidden_activation(batch_norm(
        deconv2d(dc1, [self.batch_size, 16, 16, 128], name='dc2', stddev=0.02),
        name='bn2', is_training=is_training))
      dc3 = self.hidden_activation(batch_norm(
        deconv2d(dc2, [self.batch_size, 32, 32,  64], name='dc3', stddev=0.02),
        name='bn3', is_training=is_training))
      dc4 = self.output_activation(
        deconv2d(dc3, [self.batch_size, 32, 32, 3], 3, 3, 1, 1, name='dc4', stddev=0.02))

      dc1_sigma = self.hidden_activation(batch_norm(
        deconv2d( l0, [self.batch_size,  8,  8, 256], name='dc1_sigma', stddev=0.02),
        name='bn1_sigma', is_training=is_training))
      dc2_sigma = self.hidden_activation(batch_norm(
        deconv2d(dc1_sigma, [self.batch_size, 16, 16, 128], name='dc2_sigma', stddev=0.02),
        name='bn2_sigma', is_training=is_training))
      dc3_sigma = self.hidden_activation(batch_norm(
        deconv2d(dc2_sigma, [self.batch_size, 32, 32,  64], name='dc3_sigma', stddev=0.02),
        name='bn3_sigma', is_training=is_training))
      dc4_sigma = 1e-3 * tf.sigmoid(
        deconv2d(dc3_sigma, [self.batch_size, 32, 32, 3], 3, 3, 1, 1, name='dc4_sigma', stddev=0.02))

      dc4_sigma = tf.abs(dc4_sigma)
      noise = tf.random_normal(tf.shape(dc4)) 
      x = dc4 + dc4_sigma * noise

      logprob = tf.reduce_sum(
        - tf.log(dc4_sigma + 1e-6)
        , reduction_indices=[1,2,3])
    return x, logprob

  def generate_noise(self):
    if self.z_distribution == 'normal':
      return np.random.randn(self.batch_size, self.hidden_dim).astype(np.float32)
    elif self.z_distribution == 'uniform' :
      return np.random.uniform(-1, 1, (self.batch_size, self.hidden_dim)).astype(np.float32)
    else:
      raise NotImplementedError


class Discrminator(object):

  def __init__(self, batch_size=64, hidden_activation=lrelu, output_dim=1, scope='critic', **kwargs):
    self.batch_size = batch_size
    self.hidden_activation = hidden_activation
    self.output_dim = output_dim
    self.scope = scope

  def __call__(self, x, update_collection=tf.GraphKeys.UPDATE_OPS, **kwargs):
    with tf.variable_scope(self.scope):
      feat = self.hidden_activation(conv2d(   x,  64, 3, 3, 1, 1, mhe=False, net_type='d',
        spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_0'))
      feat = self.hidden_activation(conv2d(feat, 128, 3, 3, 1, 1, mhe=False, net_type='d',
        spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_1'))
      feat = tf.nn.avg_pool(feat, [1,2,2,1], [1,2,2,1], 'VALID')

      feat = self.hidden_activation(conv2d(feat, 128, 3, 3, 1, 1, mhe=False, net_type='d',
        spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_0'))
      feat = self.hidden_activation(conv2d(feat, 256, 3, 3, 1, 1, mhe=False, net_type='d',
        spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_1'))
      feat = tf.nn.avg_pool(feat, [1,2,2,1], [1,2,2,1], 'VALID')

      feat = self.hidden_activation(conv2d(feat, 256, 3, 3, 1, 1, mhe=False, net_type='d',
        spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c3_0'))
      feat = tf.nn.avg_pool(feat, [1,7,7,1], [1,7,7,1], 'VALID')

      feat = tf.reshape(feat, [self.batch_size, -1])
      feat = linear(feat, self.output_dim, mhe=False, net_type='d',
        spectral_normed=True, update_collection=update_collection, stddev=0.02, name='l4')
    return tf.reshape(feat, [-1])
