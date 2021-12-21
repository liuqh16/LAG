#!/bin/sh
env="gym"
scenario="HalfCheetah-v2"
algo="ppo"
exp="test"
seed=1
echo "env is ${env}, task is ${task}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python train/train_gym.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --user-name "jyh" \
    --seed 1 --n-training-threads 1 --n-rollout-threads 8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --num-mini-batch 5 --num-env-steps 5e6 \
    --buffer-size 3000 --cuda \