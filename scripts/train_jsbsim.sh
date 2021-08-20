#!/bin/sh
env="SingleControl"     # JSBSim or SingleControl
task="heading_altitude_task"     # singlecombat_task or heading_task or heading_altitude_task.py
num_agents=1
algo="ppo"
exp="heading_0820"
seed=1

echo "env is ${env}, task is ${task}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --task-name ${task} --experiment-name ${exp} --num-agents ${num_agents} \
    --user-name "jyh" --wandb-name "jyh" --use-wandb\
    --seed 1 --n-training-threads 1 --n-rollout-threads 30 --cuda \
    --num-mini-batch 5 --episode-length 3000 --num-env-steps 1000000 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" \
    --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 10
