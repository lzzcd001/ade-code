#!/bin/bash

export CUDA_VISIBLE_DEVICES=
export KMP_DUPLICATE_LIB_OK=True

data_name=moons
mcmc=HMC
gp_lambda=0
p_sigma=1
hmc_clip=1
f_depth=3
clip_sample=True
lr=0.001
vc=0

mcmc_steps=5
f_bd=1
gen_depth=10
f_type=mlp
flow_type=norm
ema=0.99

data_path=../../../data/synthetic/${data_name}.npy

save_dir=$HOME/scratch/results/ade/$data_name/mcmc-${mcmc}-mc_step-${mcmc_steps}-f-${f_type}-f_dep-${f_depth}-g-${flow_type}-g_dep-${gen_depth}-gp-${gp_lambda}-p-${p_sigma}-c-${hmc_clip}-cs-${clip_sample}-lr-${lr}-fbd-${f_bd}-ema-${ema}-vc-${vc}

python main_ade_toy.py \
    -save_dir $save_dir \
    -moment_penalty $vc \
    -clip_sample $clip_sample \
    -learning_rate $lr \
    -ema_decay $ema \
    -gp_lambda $gp_lambda \
    -hmc_p_sigma $p_sigma \
    -hmc_clip $hmc_clip \
    -data_name $data_name \
    -energy_type $f_type \
    -f_bd $f_bd \
    -data_dump $data_path \
    -mcmc_type $mcmc \
    -f_depth $f_depth \
    -mcmc_steps $mcmc_steps \
    -use_2nd_order_grad True \
    -gen_depth $gen_depth \
    -hmc_adaptive_mode auto \
    -num_epochs 100 \
    $@
