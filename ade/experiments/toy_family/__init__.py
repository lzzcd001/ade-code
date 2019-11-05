from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from ade.common.data_utils.dataset import ToyDataset
from ade.common.data_utils.toy_data_gen import inf_train_gen
import ade.common.data_utils.kernel_toy_datagen as kernel_gen


class OnlineToyDataset(ToyDataset):
    def __init__(self, data_name):
        super(OnlineToyDataset, self).__init__(2)
        self.data_name = data_name
        self.rng = np.random.RandomState()

    def gen_batch(self, batch_size):
        return inf_train_gen(self.data_name, self.rng, batch_size)


class KernelToyDataset(ToyDataset):
    def __init__(self, data_name):
        super(KernelToyDataset, self).__init__(2)
        self.data_name = data_name
        self.cls = getattr(kernel_gen, self.data_name)()

    def gen_batch(self, batch_size):
        return self.cls.sample(batch_size)

class SizedKernelToyDataset(ToyDataset):
    def __init__(self, data_name, data_len=1000):
        super(SizedKernelToyDataset, self).__init__(2)
        self.data_name = data_name
        self.cls = getattr(kernel_gen, self.data_name)()
        self.data_len = data_len
        self.data = self.cls.sample(data_len)

    def gen_batch(self, batch_size):
        inds = np.random.choice(np.arange(self.data_len), size=batch_size)
        return self.data[inds, :].reshape((batch_size,) + self.data.shape[1:])
    


