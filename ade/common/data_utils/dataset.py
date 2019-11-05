from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import numpy as np


class ToyDataset(object):

    def __init__(self, dim, data_file=None, static_data=None):
        if data_file is not None:
            self.static_data = np.load(data_file)
        elif static_data is not None:
            self.static_data = static_data
        else:
            self.static_data = None
        self.dim = dim
        # print(self.static_data.shape)

    def gen_batch(self, batch_size):
        raise NotImplementedError

    def data_gen(self, batch_size, auto_reset):
        if self.static_data is not None:
            num_obs = self.static_data.shape[0]
            while True:
                for pos in range(0, num_obs, batch_size):
                    if pos + batch_size > num_obs:  # the last mini-batch has fewer samples
                        if auto_reset:  # no need to use this last mini-batch
                            break
                        else:
                            num_samples = num_obs - pos
                    else:
                        num_samples = batch_size
                    yield self.static_data[pos : pos + num_samples, :]
                if not auto_reset:
                    break
                np.random.shuffle(self.static_data)
        else:
            while True:
                yield self.gen_batch(batch_size)

class SizedToyDataset(object):

    def __init__(self, dim, data_file=None, static_data=None):
        if data_file is not None:
            self.static_data = np.load(data_file)
            inds = np.random.choice(np.arange(self.static_data.shape[0]), size=1000)
            self.static_data = self.static_data[inds]
        elif static_data is not None:
            self.static_data = static_data
        else:
            self.static_data = None
        self.dim = dim
        print(self.static_data.shape)

    def gen_batch(self, batch_size):
        raise NotImplementedError

    def data_gen(self, batch_size, auto_reset):
        if self.static_data is not None:
            num_obs = self.static_data.shape[0]
            while True:
                for pos in range(0, num_obs, batch_size):
                    if pos + batch_size > num_obs:  # the last mini-batch has fewer samples
                        if auto_reset:  # no need to use this last mini-batch
                            break
                        else:
                            num_samples = num_obs - pos
                    else:
                        num_samples = batch_size
                    yield self.static_data[pos : pos + num_samples, :]
                if not auto_reset:
                    break
                np.random.shuffle(self.static_data)
        else:
            while True:
                yield self.gen_batch(batch_size)



