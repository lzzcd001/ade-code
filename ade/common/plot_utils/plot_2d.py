from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(pdf_func, out_name, size=3):
    w = 100
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx.flatten(), yy.flatten()]).transpose()

    scores = pdf_func(coords)
    a = scores.reshape((w, w))

    plt.imshow(a)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()


def plot_samples(samples, out_name):
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.axis('equal')
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()

def plot_joint(dataset, samples, out_name):
    plt.scatter(dataset[:, 0], dataset[:, 1], c='r', marker='x')
    plt.scatter(samples[:, 0], samples[:, 1], c='b', marker='.')
    plt.legend(['training data', 'ADE sampled'])
    plt.axis('equal')
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()

    fname = out_name.split('/')[-1]
    out_name = '/'.join(out_name.split('/')[:-1]) + '/none-' + fname
    plt.scatter(dataset[:, 0], dataset[:, 1], c='r', marker='x')
    plt.scatter(samples[:, 0], samples[:, 1], c='b', marker='.')
    plt.axis('equal')
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()