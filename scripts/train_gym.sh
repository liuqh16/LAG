#!/bin/sh
env="gym"
task="CartPole-v1"
seed=1
num_agents=1
algo="ppo"
exp="CartPole-v1"
echo "env is ${env}, task is ${task}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=2 python train/train_gym.py \
    --env-name ${env} --algorithm-name ${algo} --task-name ${task} --experiment-name ${exp} --num-agents ${num_agents} \
    --user-name "jyh" \
    --seed 1 --n-training-threads 1 --n-rollout-threads 4 \
    --num-mini-batch 5 --num-env-steps 1e6 \
    --buffer-size 3000