#!/bin/sh
env="SingleCombat"
scenario="1v1/Missile/Selfplay"
num_agents=2
algo="ppo"
exp="test"
seed=1

# export PYTHONPATH=$PYTHONPATH:$(cd ../ && pwd)

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --seed 1 --n-training-threads 1 --n-rollout-threads 1 --cuda\
    --num-mini-batch 5 --buffer-size 2700 --episode-length 900 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" \
    --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8