#!/bin/sh
env="JSBSim"
task="singlecombat"
exp="VsBaseline_OriginReward"
seed=1

echo "env is ${env}, task is ${scenario}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python try_train_jsbsim.py --env ${env} --task ${task} --exp ${exp} \
    --num-parallel-env 60 --num-train 1000 --num-eval 60 --gpu-id 0 --seed ${seed}
