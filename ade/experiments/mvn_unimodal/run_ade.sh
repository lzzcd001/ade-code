#!/bin/bash

export CUDA_VISIBLE_DEVICES=

mcmc=HMC
mcmc_steps=5

python main_ade.py \
    -energy_type gauss \
    -save_dir scratch \
    -mcmc_type $mcmc \
    -mcmc_steps $mcmc_steps \
    -use_mh True \
    -use_2nd_order_grad True \
    -gen_depth 2 \
    -hmc_adaptive_mode auto \
    -hmc_clip 0.1 \
    -flow_type mlp \
    -moment_penalty 1 \
    $@


