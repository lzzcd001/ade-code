from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


def fp_flow(energy_func, base_flow, mcmc, z, log_z):
    x0, ll = base_flow(z, log_z)
    xt, ll = mcmc(energy_func, x0, log_q_p=ll)
    return xt, ll