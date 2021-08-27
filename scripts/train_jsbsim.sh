#!/bin/sh
env="JSBSim"
scenario="singlecombat_vsbaseline"
num_agents=1
algo="ppo"
exp="debug"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} --num-agents ${num_agents} \
    --use-wandb --user-name "liuqh16" --wandb-name "liuqh16" \
    --seed 1 --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --num-mini-batch 5 --buffer-size 2700 --episode-length 900 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" \
    --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8