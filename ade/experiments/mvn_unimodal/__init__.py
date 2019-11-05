from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

from ade.common.data_utils.dataset import ToyDataset
from ade.common.distributions import DiagMvn


class MvnGaussDataset(ToyDataset):
    def __init__(self, mu, sigma, dim, static_data=None):
        super(MvnGaussDataset, self).__init__(dim, static_data=static_data)
        self.mu = mu
        self.sigma = sigma
        self.data_dist = DiagMvn(mu=[self.mu] * self.dim,
                                 sigma=[self.sigma] * self.dim)

    def gen_batch(self, batch_size):
        return self.data_dist.get_samples(num_samples=batch_size)

class SizedMvnGaussDataset(ToyDataset):
    def __init__(self, mu, sigma, dim, static_data=None):
        super(SizedMvnGaussDataset, self).__init__(dim, static_data=static_data)
        self.mu = mu
        self.sigma = sigma
        self.data_dist = DiagMvn(mu=[self.mu] * self.dim,
                                 sigma=[self.sigma] * self.dim)
        self.num_samples = 1000
        self.data = self.data_dist.get_samples(num_samples=self.num_samples)
        

    def gen_batch(self, batch_size):
        inds = np.random.choice(np.arange(self.num_samples), size=batch_size)
        return self.data[inds, :].reshape((batch_size,) + self.data.shape[1:])        

