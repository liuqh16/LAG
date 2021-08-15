#!/bin/sh
env="heading"
task="heading"
exp="originReward"
seed=1

echo "env is ${env}, task is ${scenario}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python try_train_selfplay.py --env ${env} --task ${task} --exp ${exp} \
    --num-parallel-env 60 --num-train 2000 --num-eval 60 --gpu-id 0 --seed ${seed}
